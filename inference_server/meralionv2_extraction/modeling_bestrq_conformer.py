import os
import torch
import math
from torch import nn
from typing import Optional, Tuple, Union

from transformers.modeling_utils import PreTrainedModel
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutput, Wav2Vec2BaseModelOutput, CausalLMOutput
from safetensors.torch import load_file

from .configuration_bestrq_conformer import MeralionBestRqConformerConfig


_HIDDEN_STATES_START_POSITION = 2


def lengths_to_padding_mask(lens: torch.LongTensor) -> torch.BoolTensor:
    """
    Create a boolean padding mask from a tensor of sequence lengths.

    Args:
        lens (`torch.LongTensor`):
            A tensor of shape `(batch_size,)` containing the length of each sequence in the batch.

    Returns:
        `torch.BoolTensor`:
            A boolean mask of shape `(batch_size, max_len)` where `True` indicates a padded position.
    """
    bsz, max_lens = lens.size(0), torch.max(lens).item()
    mask = torch.arange(max_lens).to(lens.device).view(1, max_lens)
    mask = mask.expand(bsz, -1) >= lens.view(bsz, 1).expand(-1, max_lens)
    return mask


def make_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    """Make mask tensor containing indices of padded part.

    See description of make_non_pad_mask.

    Args:
        lengths (torch.Tensor): Batch of lengths (B,).
    Returns:
        torch.Tensor: Mask tensor containing indices of padded part.

    Examples:
        >>> lengths = [5, 3, 2]
        >>> make_pad_mask(lengths)
        masks = [[0, 0, 0, 0 ,0],
                 [0, 0, 0, 1, 1],
                 [0, 0, 1, 1, 1]]
    """
    batch_size = lengths.size(0)
    max_len = max_len if max_len > 0 else lengths.max().item()
    seq_range = torch.arange(0,
                             max_len,
                             dtype=torch.int64,
                             device=lengths.device)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_length_expand = lengths.unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand
    return mask


class Conv2dSubsampling(nn.Module):
    """
    Convolutional 2D subsampling (to 1/4 length)
    For feature extraction/downsampling of input mel spectrogram

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution

    Inputs:
        inputs (batch, time, dim): Tensor containing sequence of inputs
        input_lengths (batch): Tensor containing input_length for each item in batch

    Returns:
        outputs (batch, time, dim): Tensor produced by the convolution
        output_lengths (batch): Tensor containing output_length for each item in batch
    """
    def __init__(self, config):
        super().__init__()
        self.sequential = nn.Sequential(
            nn.Conv2d(config.input_channels, config.hidden_size, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(config.hidden_size, config.hidden_size, kernel_size=3, stride=2),
            nn.ReLU(),
        )

    def forward(self, inputs: torch.Tensor, input_lengths: torch.Tensor):
        _, max_seq_len, _ = inputs.size()
        outputs = self.sequential(inputs.unsqueeze(1))
        batch_size, channels, subsampled_lengths, sumsampled_dim = outputs.size()

        outputs = outputs.permute(0, 2, 1, 3)
        outputs = outputs.contiguous().view(batch_size, subsampled_lengths, channels * sumsampled_dim)

        subsampling_factor = int(max_seq_len * 1.0 / subsampled_lengths + 0.5)
        input_len_0 = (input_lengths.float() / subsampling_factor).ceil().long()
        input_len_1 = outputs.size(1) * torch.ones([input_lengths.size(0)]).long().to(
            input_len_0.device
        )
        output_lengths = torch.min(input_len_0, input_len_1)

        return outputs, output_lengths


class ConformerRelPositionalEmbedding(nn.Module):
    """Relative positional encoding module (new implementation).

    Args:
        d_model: Embedding dimension.
        dropout_rate: Dropout rate.
        max_len: Maximum input length.
    """
    def __init__(self, config):
        super().__init__()
        self.max_len = config.max_source_positions
        self.d_model = config.hidden_size
        self.register_buffer("pe", None) 
        self.extend_pe(torch.tensor(0.0).expand(1, self.max_len))

    def extend_pe(self, x):
        """Reset the positional encodings."""
        if self.pe is not None:
            # self.pe contains both positive and negative parts
            # the length of self.pe is 2 * input_len - 1
            if self.pe.size(1) >= x.size(1) * 2 - 1:
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return
        # Suppose `i` means to the position of query vector and `j` means the
        # position of key vector. We use position relative positions when keys
        # are to the left (i>j) and negative relative positions otherwise (i<j).
        pe_positive = torch.zeros(x.size(1), self.d_model)
        pe_negative = torch.zeros(x.size(1), self.d_model)
        position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.d_model)
        )
        pe_positive[:, 0::2] = torch.sin(position * div_term)
        pe_positive[:, 1::2] = torch.cos(position * div_term)
        pe_negative[:, 0::2] = torch.sin(-1 * position * div_term)
        pe_negative[:, 1::2] = torch.cos(-1 * position * div_term)

        # Reserve the order of positive indices and concat both positive and
        # negative indices. This is used to support the shifting trick
        # as in https://arxiv.org/abs/1901.02860
        pe_positive = torch.flip(pe_positive, [0]).unsqueeze(0)
        pe_negative = pe_negative[1:].unsqueeze(0)
        pe = torch.cat([pe_positive, pe_negative], dim=1)
        self.pe = pe.to(device=x.device, dtype=x.dtype)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor):
        """Add positional encoding.
        Args:
            x : Input tensor T X B X C.
        Returns:
            torch.Tensor: Encoded tensor T X B X C.

        """
        x = x.transpose(0, 1)  # Change TBC to BTC
        self.extend_pe(x)
        pos_emb = self.pe[
            :,
            self.pe.size(1) // 2 - x.size(1) + 1 : self.pe.size(1) // 2 + x.size(1),
        ]
        pos_emb = pos_emb.transpose(0, 1)  # change to TBC
        return pos_emb


class ConformerRotaryPositionalEmbedding(nn.Module):
    """Rotary positional embedding
    Reference : https://blog.eleuther.ai/rotary-embeddings/ Paper: https://arxiv.org/pdf/2104.09864.pdf
    """

    def __init__(self, config):
        super().__init__()
        dim = config.hidden_size // config.num_attention_heads
        base = config.rotary_embedding_base

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.cached_sequence_length = None
        self.cached_rotary_positional_embedding = None

    def forward(self, hidden_states):
        sequence_length = hidden_states.shape[1]

        if sequence_length == self.cached_sequence_length and self.cached_rotary_positional_embedding is not None:
            return self.cached_rotary_positional_embedding

        self.cached_sequence_length = sequence_length
        # Embeddings are computed in the dtype of the inv_freq constant
        time_stamps = torch.arange(sequence_length).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", time_stamps, self.inv_freq)
        embeddings = torch.cat((freqs, freqs), dim=-1)

        cos_embeddings = embeddings.cos()[:, None, None, :]
        sin_embeddings = embeddings.sin()[:, None, None, :]
        # Computed embeddings are cast to the dtype of the hidden state inputs
        self.cached_rotary_positional_embedding = torch.stack([cos_embeddings, sin_embeddings]).type_as(hidden_states)
        return self.cached_rotary_positional_embedding


class ConformerInputFeatureProjection(nn.Module):
    """
    Projects the input features to the hidden size of the Conformer model. This layer is applied after the
    convolutional subsampling.
    """
    def __init__(self, config):
        super().__init__()
        subsample_embed_dim = config.hidden_size * (((config.input_dim - 1) // 2 - 1) // 2)
        #self.layer_norm = nn.LayerNorm(config.conv_dim[-1], eps=config.layer_norm_eps)
        self.projection = nn.Linear(subsample_embed_dim, config.hidden_size)
        self.dropout = nn.Dropout(config.feat_proj_dropout)

    def forward(self, hidden_states):
        """
        Args:
            hidden_states: Input Tensor of shape  T X B X C
        Returns:
            Tensor of shape T X B X C
        """
        # non-projected hidden states are needed for quantization
        #norm_hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.projection(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class ConformerFeedForward(nn.Module):
    """Positionwise feed forward layer used in conformer"""
    def __init__(self, config):
        super().__init__()

        #self.layer_norm = torch.nn.LayerNorm(config.hidden_size, eps=1e-5, elementwise_affine=True)

        self.intermediate_dropout = nn.Dropout(config.activation_dropout)

        self.intermediate_dense = nn.Linear(config.hidden_size, config.ffn_dim)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

        self.output_dense = nn.Linear(config.ffn_dim, config.hidden_size)
        self.output_dropout = nn.Dropout(config.hidden_dropout)

    def forward(self, hidden_states):
        """
        Args:
            x: Input Tensor of shape  T X B X C
        Returns:
            Tensor of shape T X B X C
        """
        hidden_states = self.intermediate_dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.intermediate_dropout(hidden_states)
        hidden_states = self.output_dense(hidden_states)
        hidden_states = self.output_dropout(hidden_states)
        return hidden_states


class ConformerConvolutionModule(nn.Module):
    """Convolution block used in the conformer block"""

    def __init__(self, config):
        super().__init__()
        if (config.conv_depthwise_kernel_size - 1) % 2 == 1:
            raise ValueError("`config.conv_depthwise_kernel_size` should be a odd number for 'SAME' padding")
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.pointwise_conv1 = nn.Conv1d(
            config.hidden_size,
            2 * config.hidden_size,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.glu = nn.GLU(dim=1)
        self.depthwise_conv = nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            config.conv_depthwise_kernel_size,
            stride=1,
            padding=(config.conv_depthwise_kernel_size - 1) // 2,
            groups=config.hidden_size,
            bias=False,
        )
        self.batch_norm = nn.BatchNorm1d(config.hidden_size)
        self.activation = ACT2FN[config.hidden_act]
        self.pointwise_conv2 = nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.dropout = nn.Dropout(config.conformer_conv_dropout)

    def forward(self, hidden_states):
        """
        Args:
            hidden_states: Input of shape B X T X C
        Returns:
            Tensor of shape B X T X C
        """
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = hidden_states.transpose(1, 2)

        # GLU mechanism
        # => (batch, 2*channel, dim)
        hidden_states = self.pointwise_conv1(hidden_states)
        # => (batch, channel, dim)
        hidden_states = self.glu(hidden_states)

        # 1D Depthwise Conv
        hidden_states = self.depthwise_conv(hidden_states)
        hidden_states = self.batch_norm(hidden_states)
        hidden_states = self.activation(hidden_states)

        hidden_states = self.pointwise_conv2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states.transpose(1, 2)
        return hidden_states


class ConformerSelfAttention(nn.Module):
    """ConformerSelfAttention object.
    Can be enhanced with rotary or relative position embeddings.
    """

    def __init__(self, config):
        super().__init__()

        self.head_size = config.hidden_size // config.num_attention_heads
        self.num_heads = config.num_attention_heads
        self.position_embeddings_type = config.position_embeddings_type

        self.linear_q = nn.Linear(config.hidden_size, config.hidden_size)
        self.linear_k = nn.Linear(config.hidden_size, config.hidden_size)
        self.linear_v = nn.Linear(config.hidden_size, config.hidden_size)
        self.linear_out = nn.Linear(config.hidden_size, config.hidden_size)

        self.dropout = nn.Dropout(p=config.attention_dropout)

        if self.position_embeddings_type == "relative":
            # linear transformation for positional encoding
            self.linear_pos = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
            # these two learnable bias are used in matrix c and matrix d
            # as described in https://arxiv.org/abs/1901.02860 Section 3.3
            self.pos_bias_u = nn.Parameter(torch.Tensor(self.num_heads, self.head_size))
            self.pos_bias_v = nn.Parameter(torch.Tensor(self.num_heads, self.head_size))
            torch.nn.init.xavier_uniform_(self.pos_bias_u) ##
            torch.nn.init.xavier_uniform_(self.pos_bias_v) ##

    def forward(
        self,
        hidden_states: torch.Tensor, #[T, B, C]
        attention_mask: Optional[torch.Tensor] = None,
        relative_position_embeddings: Optional[torch.Tensor] = None, #[T, B, C]
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # self-attention mechanism
        hidden_states = hidden_states.transpose(0, 1) #[B, T, C]
        relative_position_embeddings = relative_position_embeddings.transpose(0, 1) #[B, T, C]
        batch_size, sequence_length, hidden_size = hidden_states.size()

        # make sure query/key states can be != value states
        query_key_states = hidden_states
        value_states = hidden_states

        if self.position_embeddings_type == "rotary":
            if relative_position_embeddings is None:
                raise ValueError(
                    "`relative_position_embeddings` has to be defined when `self.position_embeddings_type == 'rotary'"
                )
            query_key_states = self._apply_rotary_embedding(query_key_states, relative_position_embeddings)

        # project query_key_states and value_states
        query = self.linear_q(query_key_states).view(batch_size, -1, self.num_heads, self.head_size)
        key = self.linear_k(query_key_states).view(batch_size, -1, self.num_heads, self.head_size)
        value = self.linear_v(value_states).view(batch_size, -1, self.num_heads, self.head_size)

        # => (batch, head, time1, d_k)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        if self.position_embeddings_type == "relative":
            if relative_position_embeddings is None:
                raise ValueError(
                    "`relative_position_embeddings` has to be defined when `self.position_embeddings_type =="
                    " 'relative'"
                )
            # apply relative_position_embeddings to qk scores
            # as proposed in Transformer_XL: https://arxiv.org/abs/1901.02860
            scores = self._apply_relative_embeddings(
                query=query, key=key, relative_position_embeddings=relative_position_embeddings
            )
        else:
            scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_size)

        # apply attention_mask if necessary
        if attention_mask is not None:
            scores = scores.masked_fill(
                attention_mask.unsqueeze(1).unsqueeze(2).to(bool),
                float("-inf"),  # (batch, head, time1, time2)
            )

        # => (batch, head, time1, time2)
        probs = torch.softmax(scores, dim=-1)
        probs = self.dropout(probs)

        # => (batch, head, time1, d_k)
        hidden_states = torch.matmul(probs, value)

        # => (batch, time1, hidden_size)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.num_heads * self.head_size)
        hidden_states = self.linear_out(hidden_states)

        # => (time1, batch, hidden_size)
        hidden_states = hidden_states.transpose(0, 1)

        return hidden_states, probs

    def _apply_rotary_embedding(self, hidden_states, relative_position_embeddings):
        batch_size, sequence_length, hidden_size = hidden_states.size()
        hidden_states = hidden_states.view(batch_size, sequence_length, self.num_heads, self.head_size)

        cos = relative_position_embeddings[0, :sequence_length, ...]
        sin = relative_position_embeddings[1, :sequence_length, ...]

        # rotate hidden_states with rotary embeddings
        hidden_states = hidden_states.transpose(0, 1)
        rotated_states_begin = hidden_states[..., : self.head_size // 2]
        rotated_states_end = hidden_states[..., self.head_size // 2 :]
        rotated_states = torch.cat((-rotated_states_end, rotated_states_begin), dim=rotated_states_begin.ndim - 1)
        hidden_states = (hidden_states * cos) + (rotated_states * sin)
        hidden_states = hidden_states.transpose(0, 1)

        hidden_states = hidden_states.view(batch_size, sequence_length, self.num_heads * self.head_size)

        return hidden_states

    def _apply_relative_embeddings(self, query, key, relative_position_embeddings):
        # 1. project positional embeddings
        # => (batch, head, d_k, 2*time1-1)
        proj_relative_position_embeddings = self.linear_pos(relative_position_embeddings)
        proj_relative_position_embeddings = proj_relative_position_embeddings.view(
            relative_position_embeddings.size(0), -1, self.num_heads, self.head_size # (batch, 2*time1-1, head, d_k)
        )
        proj_relative_position_embeddings = proj_relative_position_embeddings.transpose(1, 2) # (batch, head, 2*time1-1, d_k)
        proj_relative_position_embeddings = proj_relative_position_embeddings.transpose(2, 3) # (batch, head, d_k, 2*time1-1)

        # 2. Add bias to query
        # => (batch, head, time1, d_k)
        query = query.transpose(1, 2) # (batch, time1, head, d_k)
        q_with_bias_u = (query + self.pos_bias_u).transpose(1, 2)
        q_with_bias_v = (query + self.pos_bias_v).transpose(1, 2)

        # 3. attention score: first compute matrix a and matrix c
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        # => (batch, head, time1, time2)
        scores_ac = torch.matmul(q_with_bias_u, key.transpose(-2, -1))

        # 4. then compute matrix b and matrix d
        # => (batch, head, time1, 2*time1-1)
        scores_bd = torch.matmul(q_with_bias_v, proj_relative_position_embeddings)

        # 5. shift matrix b and matrix d
        zero_pad = torch.zeros((*scores_bd.size()[:3], 1), device=scores_bd.device, dtype=scores_bd.dtype)
        scores_bd_padded = torch.cat([zero_pad, scores_bd], dim=-1)
        scores_bd_padded_shape = scores_bd.size()[:2] + (scores_bd.shape[3] + 1, scores_bd.shape[2])
        scores_bd_padded = scores_bd_padded.view(*scores_bd_padded_shape)
        scores_bd = scores_bd_padded[:, :, 1:].view_as(scores_bd)
        scores_bd = scores_bd[:, :, :, : scores_bd.size(-1) // 2 + 1]

        # 6. sum matrices
        # => (batch, head, time1, time2)
        scores = (scores_ac + scores_bd) / math.sqrt(self.head_size)

        return scores


class ConformerEncoderLayer(nn.Module):
    """Conformer block based on https://arxiv.org/abs/2005.08100."""

    def __init__(self, config):
        super().__init__()
        embed_dim = config.hidden_size
        dropout = config.attention_dropout

        # Feed-forward 1
        self.ffn1_layer_norm = nn.LayerNorm(embed_dim)
        self.ffn1 = ConformerFeedForward(config)

        # Self-Attention
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)
        self.self_attn_dropout = nn.Dropout(dropout)
        self.self_attn = ConformerSelfAttention(config)

        # Conformer Convolution
        self.conv_module = ConformerConvolutionModule(config)

        # Feed-forward 2
        self.ffn2_layer_norm = nn.LayerNorm(embed_dim)
        self.ffn2 = ConformerFeedForward(config)
        self.final_layer_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        hidden_states, # [T, B, C]
        attention_mask: Optional[torch.Tensor] = None,
        relative_position_embeddings: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ):
        hidden_states = hidden_states

        # 1. Feed-Forward 1 layer
        residual = hidden_states
        hidden_states = self.ffn1_layer_norm(hidden_states)
        hidden_states = self.ffn1(hidden_states)
        hidden_states = hidden_states * 0.5 + residual
        residual = hidden_states

        # 2. Self-Attention layer
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            relative_position_embeddings=relative_position_embeddings,
            output_attentions=output_attentions,
        )
        hidden_states = self.self_attn_dropout(hidden_states)
        hidden_states = hidden_states + residual

        # 3. Convolutional Layer
        residual = hidden_states
        hidden_states = hidden_states.transpose(0, 1) # [T,B,C] to [B,T,C]
        hidden_states = self.conv_module(hidden_states)
        hidden_states = hidden_states.transpose(0, 1) # [B,T,C] to [T,B,C]
        hidden_states = residual + hidden_states

        # 4. Feed-Forward 2 Layer
        residual = hidden_states
        hidden_states = self.ffn2_layer_norm(hidden_states)
        hidden_states = self.ffn2(hidden_states)
        hidden_states = hidden_states * 0.5 + residual
        hidden_states = self.final_layer_norm(hidden_states)

        return hidden_states, attn_weights


class ConformerEncoder(nn.Module):
    """
    The Conformer encoder module. This module is composed of a stack of Conformer layers and is responsible for
    encoding the input features.

    Args:
        config ([`MeralionBestRqConformerConfig`]):
            The configuration object for the model.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_scale = math.sqrt(config.hidden_size)
        if config.no_scale_embedding:
            self.embed_scale = 1.0

        if config.position_embeddings_type == "relative":
            self.embed_positions = ConformerRelPositionalEmbedding(config)
        elif config.position_embeddings_type == "rotary":
            self.embed_positions = ConformerRotaryPositionalEmbedding(config)
        else:
            self.embed_positions = None

        self.input_projection = ConformerInputFeatureProjection(config) # [T,B,C]

        self.layers = nn.ModuleList([ConformerEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

        self.self_condition_layers = self.config.self_condition_layers
        self.conditioning_layer = None
        self.conditioning_softmax = None
        self.out_projection = None

        if self.self_condition_layers:
            # If self-conditioning is enabled, we need these layers
            if self.config.vocab_size is None:
                raise ValueError("output_size (i.e., vocab_size) must be provided for self-conditioning.")
            self.conditioning_layer = nn.Linear(self.config.vocab_size, self.config.hidden_size)
            self.conditioning_softmax = nn.Softmax(dim=-1)

    def forward(
        self,
        hidden_states, # conv_out
        attention_mask=None, # encoder_padding_mask
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        ctc_decoder=None,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        ctc_outputs = () if self.self_condition_layers else None

        hidden_states = self.embed_scale * hidden_states

        if self.embed_positions is not None:
            relative_position_embeddings = self.embed_positions(hidden_states) # [T,B,C]
        else:
            relative_position_embeddings = None

        hidden_states = self.input_projection(hidden_states) # [T,B,C]

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states.transpose(0, 1),)

        for i, layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = torch.rand([])

            skip_the_layer = True if self.training and (dropout_probability < self.config.layerdrop) else False
            if not skip_the_layer:
                layer_outputs = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    relative_position_embeddings=relative_position_embeddings,
                    output_attentions=output_attentions,
                )
                hidden_states = layer_outputs[0]

            if skip_the_layer:
                layer_outputs = (None, None)

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states.transpose(0, 1),) # [T,B,C] -> [B,T,C]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

            if i in self.self_condition_layers:
                assert isinstance(ctc_decoder, nn.Module), "A CTC decoder must be passed in for self conditioning"
                # Apply CTC decoder to the output of this layer
                ctc_logits = ctc_decoder(hidden_states.transpose(0, 1))
                ctc_outputs = ctc_outputs + (ctc_logits,)

                ctc_probs = self.conditioning_softmax(ctc_logits)
                conditioning_embedding = self.conditioning_layer(ctc_probs).transpose(0, 1) #[T, B, C]

                hidden_states = hidden_states + conditioning_embedding # Additive conditioning residual

        hidden_states = hidden_states.transpose(0, 1) # [B,T,C]

        if self.self_condition_layers:
            all_hidden_states = all_hidden_states + ctc_outputs

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class MeralionBestRqModel(PreTrainedModel):
    """
    The core BEST-RQ Conformer model. This model is a `PreTrainedModel` that takes the raw mel-spectrogram features
    and outputs the final encoder hidden states.

    This model inherits from [`PreTrainedModel`]. For the available methods and functionalities, see the
    documentation in [`PreTrainedModel`].

    Args:
        config ([`MeralionBestRqConformerConfig`]):
            Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
    """
    config_class = MeralionBestRqConformerConfig
    base_model_prefix = "bestrq_encoder"

    def __init__(self, config: MeralionBestRqConformerConfig):
        super().__init__(config)
        self.config = config
        self.conv_subsample = Conv2dSubsampling(config)

        self.encoder = ConformerEncoder(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_values: Optional[torch.Tensor], # [B,C,T]
        attention_mask: Optional[torch.Tensor],
        mask_time_indices: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        ctc_decoder: Optional[nn.Module] = None,
    ) -> Union[Tuple, Wav2Vec2BaseModelOutput]:
        r"""
        Performs the forward pass of the BEST-RQ Conformer model.

        Args:
            input_values (`torch.FloatTensor` of shape `(batch_size, num_features, sequence_length)`):
                Float values of mel features extracted from the raw speech signal.
            attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Mask to avoid performing attention on padding token indices. Mask values are in `[0, 1]`, where 1 for
                tokens that are not masked and 0 for tokens that are masked.
            mask_time_indices (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Currently unused.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
            ctc_decoder (`nn.Module`, *optional*):
                A CTC decoder module that can be used for self-conditioning. If provided, the model will apply this
                decoder at intermediate layers and use the output to condition the subsequent layers.

        Returns:
            [`Wav2Vec2BaseModelOutput`] or `tuple`:
            A [`Wav2Vec2BaseModelOutput`] (if `return_dict=True`) or a tuple of tensors (if `return_dict=False`)
            comprising the following elements:
            - **last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
              Sequence of hidden-states at the output of the last layer of the model.
            - **extract_features** (`torch.FloatTensor` of shape `(batch_size, sequence_length, conv_dim)`):
              Sequence of robustly extracted features from the CNN feature extractor.
            - **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True`):
              Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
              of shape `(batch_size, sequence_length, hidden_size)`.
            - **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True`):
              Tuple of `torch.FloatTensor` (one for each layer) of shape
              `(batch_size, num_heads, sequence_length, sequence_length)`.
            - **output_lengths** (`torch.LongTensor` of shape `(batch_size,)`):
              The length of each sequence after the convolutional subsampling.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        input_lengths = attention_mask.sum(dim=-1)

        input_values = input_values.transpose(2, 1) # [B,C,T] -> [B,T,C]
        conv_outputs, output_lengths = self.conv_subsample(input_values, input_lengths) # returns [B,T,C]
        x = conv_outputs.transpose(0, 1) # [T,B,C]

        encoder_padding_mask = make_pad_mask(output_lengths, max_len=x.shape[0])

        encoder_outputs = self.encoder(
            x,
            attention_mask=encoder_padding_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            ctc_decoder=ctc_decoder,
        )

        hidden_states = encoder_outputs[0]

        if not return_dict:
            return (hidden_states, conv_outputs) + encoder_outputs[1:]

        output = Wav2Vec2BaseModelOutput(
                                last_hidden_state=hidden_states,
                                extract_features=conv_outputs,
                                hidden_states=encoder_outputs.hidden_states,
                                attentions=encoder_outputs.attentions,
                            )
        output["output_lengths"] = output_lengths
        return output


class MeralionBestRqModelForCTC(PreTrainedModel):
    """
    BEST-RQ Conformer model with a CTC head on top for Connectionist Temporal Classification. This model can
    also use a weighted sum of the encoder's hidden states.

    This model inherits from [`PreTrainedModel`]. For the available methods and functionalities, see the
    documentation in [`PreTrainedModel`].

    Args:
        config ([`MeralionBestRqConformerConfig`]):
            Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
    """
    # Copied from transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2ForCTC.__init__ with Wav2Vec2->Wav2Vec2Conformer,wav2vec2->wav2vec2_conformer
    config_class = MeralionBestRqConformerConfig
    base_model_prefix = "bestrq_encoder"

    def __init__(self, config, target_lang: Optional[str] = None, **kwargs):
        super().__init__(config)

        self.bestrq_encoder = MeralionBestRqModel(config)
        self.dropout = nn.Dropout(config.final_dropout)

        self.target_lang = target_lang

        if self.config.use_weighted_sum:
            self.weights = nn.Parameter(torch.zeros(self.config.num_hidden_layers))
            self.softmax = nn.Softmax(dim=-1)

        if config.vocab_size is None:
            raise ValueError(
                f"You are trying to instantiate {self.__class__} with a configuration that "
                "does not define the vocabulary size of the language model head. Please "
                "instantiate the model as follows: `MeralionBestRqModelForCTC.from_pretrained(..., vocab_size=vocab_size)`. "
                "or define `vocab_size` of your model's configuration."
            )
        output_hidden_size = (
            config.output_hidden_size if hasattr(config, "add_adapter") and config.add_adapter else config.hidden_size
        )
        self.lm_head = nn.Linear(output_hidden_size, config.vocab_size)

        # Initialize weights and apply final processing
        self.post_init()

    # Copied from transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2ForCTC.forward with Wav2Vec2->Wav2Vec2Conformer,wav2vec2->wav2vec2_conformer
    def forward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, CausalLMOutput]:
        r"""
        Performs the forward pass of the BEST-RQ Conformer model with a CTC head.

        Args:
            input_values (`torch.FloatTensor` of shape `(batch_size, num_features, sequence_length)`):
                Float values of mel features extracted from the raw speech signal.
            attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Mask to avoid performing attention on padding token indices. Mask values are in `[0, 1]`, where 1 for
                tokens that are not masked and 0 for tokens that are masked.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
            labels (`torch.LongTensor` of shape `(batch_size, target_length)`, *optional*):
                Labels for connectionist temporal classification. Note that `target_length` has to be smaller or equal
                to the sequence length of the output logits. Indices are selected in `[-100, 0, ..., config.vocab_size - 1]`.
                All labels set to `-100` are ignored (masked), the loss is only computed for labels in `[0, ...,
                config.vocab_size - 1]`.

        Returns:
            [`CausalLMOutput`] or `tuple`:
            A [`CausalLMOutput`] (if `return_dict=True`) or a tuple of tensors (if `return_dict=False`)
            comprising the following elements:
            - **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
              CTC loss.
            - **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
              Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
            - **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True`):
              Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
              of shape `(batch_size, sequence_length, hidden_size)`.
            - **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True`):
              Tuple of `torch.FloatTensor` (one for each layer) of shape
              `(batch_size, num_heads, sequence_length, sequence_length)`.
        """
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        if labels is not None and labels.max() >= self.config.vocab_size:
            raise ValueError(f"Label values must be <= vocab_size: {self.config.vocab_size}")

        outputs = self.bestrq_encoder(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if self.config.use_weighted_sum:
            assert output_hidden_states is True, "output_hidden_states must be True when using use_weighted_sum"
            # Skip the first hidden state as that is collected before the first encoder layer forward
            hidden_states = outputs.hidden_states[1:self.config.num_hidden_layers+1]
            hidden_states = self._weighted_sum(hidden_states)
        else:
            hidden_states = outputs.last_hidden_state

        hidden_states = self.dropout(hidden_states)

        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # assuming that padded tokens are filled with -100
            # when not being attended to
            labels_mask = labels >= 0
            target_lengths = labels_mask.sum(-1)
            flattened_targets = labels.masked_select(labels_mask)

            # ctc_loss doesn't support fp16
            log_probs = nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0, 1)

            with torch.backends.cudnn.flags(enabled=False):
                loss = nn.functional.ctc_loss(
                    log_probs,
                    flattened_targets,
                    outputs.output_lengths, #lengths after initial CNN downsampling
                    target_lengths,
                    blank=self.config.pad_token_id,
                    reduction=self.config.ctc_loss_reduction,
                    zero_infinity=self.config.ctc_zero_infinity,
                )

        if not return_dict:
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutput(
            loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )

class LSTMCTCHead(nn.Module):
    """
    A CTC head that includes LSTM layers before the final projection. This is used for the
    `MeralionBestRqModelForLSTMCTC` model.
    """
    def __init__(self, config):
        super().__init__()
        output_hidden_size = (
            config.output_hidden_size if hasattr(config, "add_adapter") and config.add_adapter else config.hidden_size
        )
        self.lstm = nn.LSTM(
            output_hidden_size,
            config.lstm_dim,
            num_layers=config.lstm_num_layers,
            dropout=config.lstm_dropout_prob,
            batch_first=True,
            bidirectional=True,
        )
        self.lm_head = nn.Linear(config.lstm_dim * 2, config.vocab_size)

    def forward(self, hidden_states):
        hidden_states, _ = self.lstm(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits

    def _weighted_sum(self, embeddings, normalize=False):
        assert isinstance(embeddings, list) or isinstance(embeddings, tuple)
        assert len(embeddings) == self.config.num_hidden_layers, f"Number of embeddings: {len(embeddings)} does not match number of layers: {self.config.num_hidden_layers}"
        stacked_hs = torch.stack(embeddings, dim=0)

        if normalize:
            stacked_hs = nn.functional.layer_norm(stacked_hs, (stacked_hs.shape[-1],))

        _, *origin_shape = stacked_hs.shape
        stacked_hs = stacked_hs.view(self.config.num_hidden_layers, -1)
        norm_weights = self.softmax(self.weights)
        weighted_hs = (norm_weights.unsqueeze(-1) * stacked_hs).sum(dim=0)
        weighted_hs = weighted_hs.view(*origin_shape)

        return weighted_hs


class MeralionBestRqModelForLSTMCTC(PreTrainedModel):
    """
    BEST-RQ Conformer model with an LSTM-CTC head on top for Connectionist Temporal Classification. This model can
    also use a weighted sum of the encoder's hidden states.

    This model inherits from [`PreTrainedModel`]. For the available methods and functionalities, see the
    documentation in [`PreTrainedModel`].

    Args:
        config ([`MeralionBestRqConformerConfig`]):
            Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
    """
    # Copied from transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2ForCTC.__init__ with Wav2Vec2->Wav2Vec2Conformer,wav2vec2->wav2vec2_conformer
    config_class = MeralionBestRqConformerConfig
    base_model_prefix = "bestrq_encoder"
    main_input_name = "input_values"

    def __init__(self, config, target_lang: Optional[str] = None, **kwargs):
        super().__init__(config)
        self.config = config

        self.bestrq_encoder = MeralionBestRqModel(config)
        self.dropout = nn.Dropout(config.final_dropout)

        self.target_lang = target_lang

        if config.vocab_size is None:
            raise ValueError(
                f"You are trying to instantiate {self.__class__} with a configuration that "
                "does not define the vocabulary size of the language model head. Please "
                "instantiate the model as follows: `MeralionBestRqModelForLSTMCTC.from_pretrained(..., vocab_size=vocab_size)`. "
                "or define `vocab_size` of your model's configuration."
            )

        if self.config.use_weighted_sum:
            self.weights = nn.Parameter(torch.zeros(self.config.num_hidden_layers))
            self.softmax = nn.Softmax(dim=-1)

        self.lstm_ctc_decoder = LSTMCTCHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    # Copied from transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2ForCTC.forward with Wav2Vec2->Wav2Vec2Conformer,wav2vec2->wav2vec2_conformer
    def forward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = True,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, CausalLMOutput]:
        r"""
        Performs the forward pass of the BEST-RQ Conformer model with an LSTM-CTC head.

        Args:
            input_values (`torch.FloatTensor` of shape `(batch_size, num_features, sequence_length)`):
                Float values of mel features extracted from the raw speech signal.
            attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Mask to avoid performing attention on padding token indices. Mask values are in `[0, 1]`, where 1 for
                tokens that are not masked and 0 for tokens that are masked.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*, defaults to `True`):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
            labels (`torch.LongTensor` of shape `(batch_size, target_length)`, *optional*):
                Labels for connectionist temporal classification. Note that `target_length` has to be smaller or equal
                to the sequence length of the output logits. Indices are selected in `[-100, 0, ..., config.vocab_size - 1]`.
                All labels set to `-100` are ignored (masked), the loss is only computed for labels in `[0, ...,
                config.vocab_size - 1]`.

        Returns:
            [`CausalLMOutput`] or `tuple`:
            A [`CausalLMOutput`] (if `return_dict=True`) or a tuple of tensors (if `return_dict=False`)
            comprising the following elements:
            - **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
              CTC loss.
            - **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
              Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
            - **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True`):
              Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
              of shape `(batch_size, sequence_length, hidden_size)`.
            - **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True`):
              Tuple of `torch.FloatTensor` (one for each layer) of shape
              `(batch_size, num_heads, sequence_length, sequence_length)`.
        """
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        if labels is not None and labels.max() >= self.config.vocab_size:
            raise ValueError(f"Label values must be <= vocab_size: {self.config.vocab_size}")

        outputs = self.bestrq_encoder(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            ctc_decoder=self.lstm_ctc_decoder,
        )

        if self.config.use_weighted_sum:
            assert output_hidden_states is True, "output_hidden_states must be True when using use_weighted_sum"
            # Skip the first hidden state as that is collected before the first encoder layer forward
            hidden_states = outputs.hidden_states[1:self.config.num_hidden_layers+1]
            hidden_states = self._weighted_sum(hidden_states)
        else:
            hidden_states = outputs.last_hidden_state

        hidden_states = self.dropout(hidden_states)
        logits = self.lstm_ctc_decoder(hidden_states)

        loss = None
        if labels is not None:
            # assuming that padded tokens are filled with -100
            # when not being attended to
            labels_mask = labels >= 0
            target_lengths = labels_mask.sum(-1)
            flattened_targets = labels.masked_select(labels_mask)

            # ctc_loss doesn't support fp16
            log_probs = nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0, 1)

            with torch.backends.cudnn.flags(enabled=False):
                loss = nn.functional.ctc_loss(
                    log_probs,
                    flattened_targets,
                    outputs.output_lengths, #lengths after initial CNN downsampling
                    target_lengths,
                    reduction=self.config.ctc_loss_reduction,
                    zero_infinity=self.config.ctc_zero_infinity,
                )

        if not return_dict:
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutput(
            loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )

    def _weighted_sum(self, embeddings, normalize=False):
        assert isinstance(embeddings, list) or isinstance(embeddings, tuple)
        assert len(embeddings) == self.config.num_hidden_layers, f"Number of embeddings: {len(embeddings)} does not match number of layers: {self.config.num_hidden_layers}"
        stacked_hs = torch.stack(embeddings, dim=0)

        if normalize:
            stacked_hs = nn.functional.layer_norm(stacked_hs, (stacked_hs.shape[-1],))

        _, *origin_shape = stacked_hs.shape
        stacked_hs = stacked_hs.view(self.config.num_hidden_layers, -1)
        norm_weights = self.softmax(self.weights)
        weighted_hs = (norm_weights.unsqueeze(-1) * stacked_hs).sum(dim=0)
        weighted_hs = weighted_hs.view(*origin_shape)

        return weighted_hs
