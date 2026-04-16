import torch
from datasets import load_dataset
from transformers import AutoModel, AutoFeatureExtractor, AutoConfig

# to import torch codec 
#import os 
#os.add_dll_directory(r"D:\Programs\Python\Python310\Library\bin")

repo_id = 'MERaLiON/MERaLiON-SpeechEncoder-2'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# load model and feature extractor
cfg = AutoConfig.from_pretrained(repo_id, trust_remote_code=True)
#cfg.return_dict = cfg.use_return_dict 
model = AutoModel.from_pretrained(
    repo_id,
    trust_remote_code=True,
    dtype='auto'
)
model = model.to(device)

feature_extractor = AutoFeatureExtractor.from_pretrained(
    repo_id,
    trust_remote_code=True, 
    dtype='auto'
)

# prepare data
data = load_dataset("distil-whisper/librispeech_long", "clean",
                split="validation")

def batch_collater(data):
    tensors = []
    for idx, sample in enumerate(data):
        tensors.append(sample['audio']['array'])
    return tensors

audio_array = batch_collater(data)
inputs = feature_extractor(audio_array, sampling_rate=16_000,
                        return_attention_mask=True,
                        return_tensors='pt', do_normalize=False)
inputs = inputs.to(device)

# model inference to obtain features
with torch.no_grad():
    model.eval()
    output = model(input_values=inputs['input_values'],
                attention_mask=inputs['attention_mask'],
                output_hidden_states=True)

# output is a Wav2Vec2BaseModelOutput or tuple containing:
# last_hidden_state: torch.FloatTensor containing hidden states of the last layer of the model
# extract_features: torch.FloatTensor containing extracted features from the convolution downsampling layers
# hidden_states: tuple(torch.FloatTensor) containing hidden states of each layer of the model
# attentions: tuple(torch.FloatTensor) containing attention states of each layer of the model
