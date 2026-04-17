from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from new_inference_server.graph_barlow_twins import GraphAnomalyConfig, GraphBarlowAnomalyDetector
from new_inference_server.graph_barlow_twins.graph_types import GraphSample
from new_inference_server.graph_barlow_twins.model import barlow_twins_loss
from new_inference_server.voxpopuli.graph_loader import (
    DEFAULT_SVD_PREPROCESSOR_PATH,
    load_speaker_graphs,
)
from new_inference_server.voxpopuli.loader import list_speakers


MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = MODULE_DIR / "new_inference_server" / "voxpopuli" / "data"
DEFAULT_RESULTS_PATH = MODULE_DIR / "personalized_anomaly_results.json"


@dataclass
class SpeakerResult:
    speaker_id: str
    num_graphs: int
    num_adaptation_graphs: int
    threshold: float
    num_anomalies: int
    anomaly_ratio: float
    mean_score: float
    mean_adaptation_score: float
    labeled_count: int
    tp: Optional[int] = None
    fp: Optional[int] = None
    tn: Optional[int] = None
    fn: Optional[int] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1: Optional[float] = None


def _to_device_graph(graph: GraphSample, device: torch.device) -> GraphSample:
    return GraphSample(
        node_features=graph.node_features.to(device),
        adjacency=graph.adjacency.to(device),
        graph_id=graph.graph_id,
        label=graph.label,
    )


def _train_self_supervised(
    model: GraphBarlowAnomalyDetector,
    graphs: Iterable[GraphSample],
    *,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    lambda_offdiag: float,
    device: torch.device,
) -> list[float]:
    graph_list = list(graphs)
    if not graph_list:
        return []

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    losses: list[float] = []

    model.train()
    for _ in range(epochs):
        epoch_loss = 0.0
        for graph in graph_list:
            graph_device = _to_device_graph(graph, device)
            aug1 = model.augment(graph_device)
            aug2 = model.augment(graph_device)
            _, z1, _ = model(aug1)
            _, z2, _ = model(aug2)
            loss = barlow_twins_loss(z1, z2, lambda_offdiag)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())

        losses.append(epoch_loss / max(1, len(graph_list)))

    return losses


def _split_for_adaptation(graphs: list[GraphSample], adaptation_ratio: float) -> tuple[list[GraphSample], list[GraphSample]]:
    if len(graphs) < 2:
        return graphs, []

    adaptation_count = int(round(len(graphs) * adaptation_ratio))
    adaptation_count = max(1, min(len(graphs) - 1, adaptation_count))
    adaptation_graphs = graphs[:adaptation_count]
    evaluation_graphs = graphs[adaptation_count:]
    return adaptation_graphs, evaluation_graphs


def _confusion_from_scores_and_labels(
    scores: list[float],
    labels: list[Optional[int]],
    threshold: float,
) -> tuple[Optional[int], Optional[int], Optional[int], Optional[int], Optional[float], Optional[float], Optional[float], int]:
    paired = [(score, label) for score, label in zip(scores, labels) if label in (0, 1)]
    if not paired:
        return None, None, None, None, None, None, None, 0

    tp = fp = tn = fn = 0
    for score, label in paired:
        pred = 1 if score > threshold else 0
        if pred == 1 and label == 1:
            tp += 1
        elif pred == 1 and label == 0:
            fp += 1
        elif pred == 0 and label == 0:
            tn += 1
        else:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return tp, fp, tn, fn, precision, recall, f1, len(paired)


def run_personalized_anomaly_detection(
    *,
    data_root: str | Path,
    svd_preprocessor_path: str | Path = DEFAULT_SVD_PREPROCESSOR_PATH,
    max_speakers: Optional[int] = None,
    min_graphs_per_speaker: int = 3,
    adaptation_ratio: float = 0.7,
    threshold_quantile: float = 0.95,
    global_config: Optional[GraphAnomalyConfig] = None,
    global_epochs: int = 20,
    speaker_epochs: int = 8,
    speaker_learning_rate: float = 3e-4,
    device: Optional[torch.device] = None,
) -> dict:
    data_root = Path(data_root)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    speaker_ids = list_speakers(data_root)
    if max_speakers is not None:
        speaker_ids = speaker_ids[:max_speakers]

    if not speaker_ids:
        raise ValueError(f"No speakers found in {data_root}")

    per_speaker_graphs: dict[str, list[GraphSample]] = {}
    global_graphs: list[GraphSample] = []

    for speaker_id in speaker_ids:
        graphs = load_speaker_graphs(
            data_root=data_root,
            speaker_id=speaker_id,
            use_svd_preprocessor=True,
            svd_preprocessor_path=svd_preprocessor_path,
        )
        if len(graphs) < min_graphs_per_speaker:
            continue
        per_speaker_graphs[speaker_id] = graphs
        global_graphs.extend(graphs)

    if not global_graphs:
        raise ValueError("No graphs available after speaker filtering.")

    config = global_config or GraphAnomalyConfig()
    in_dim = int(global_graphs[0].node_features.shape[1])

    global_model = GraphBarlowAnomalyDetector(in_dim=in_dim, config=config).to(device)
    global_losses = _train_self_supervised(
        global_model,
        global_graphs,
        epochs=global_epochs,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        lambda_offdiag=config.lambda_offdiag,
        device=device,
    )
    global_model.fit_center(global_graphs, device=device)

    speaker_summaries: list[SpeakerResult] = []
    speaker_details: dict[str, dict] = {}

    for speaker_id, graphs in per_speaker_graphs.items():
        adaptation_graphs, _evaluation_graphs = _split_for_adaptation(graphs, adaptation_ratio)

        personalized_model = GraphBarlowAnomalyDetector(in_dim=in_dim, config=config).to(device)
        personalized_model.load_state_dict(copy.deepcopy(global_model.state_dict()))

        _train_self_supervised(
            personalized_model,
            adaptation_graphs,
            epochs=speaker_epochs,
            learning_rate=speaker_learning_rate,
            weight_decay=config.weight_decay,
            lambda_offdiag=config.lambda_offdiag,
            device=device,
        )
        personalized_model.fit_center(adaptation_graphs, device=device)

        scores: list[float] = []
        labels: list[Optional[int]] = []
        per_graph_records: list[dict] = []

        for graph in graphs:
            score = personalized_model.score(_to_device_graph(graph, device), device=device)
            scores.append(float(score))
            labels.append(graph.label)

        adaptation_scores = scores[: len(adaptation_graphs)] if adaptation_graphs else scores
        threshold = float(np.quantile(adaptation_scores, threshold_quantile))

        for graph, score in zip(graphs, scores):
            per_graph_records.append(
                {
                    "graph_id": graph.graph_id,
                    "score": float(score),
                    "is_anomaly": bool(score > threshold),
                    "label": graph.label,
                }
            )

        num_anomalies = sum(1 for score in scores if score > threshold)

        tp, fp, tn, fn, precision, recall, f1, labeled_count = _confusion_from_scores_and_labels(
            scores=scores,
            labels=labels,
            threshold=threshold,
        )

        summary = SpeakerResult(
            speaker_id=speaker_id,
            num_graphs=len(graphs),
            num_adaptation_graphs=len(adaptation_graphs),
            threshold=threshold,
            num_anomalies=num_anomalies,
            anomaly_ratio=(num_anomalies / len(graphs)) if graphs else 0.0,
            mean_score=float(np.mean(scores)) if scores else 0.0,
            mean_adaptation_score=float(np.mean(adaptation_scores)) if adaptation_scores else 0.0,
            labeled_count=labeled_count,
            tp=tp,
            fp=fp,
            tn=tn,
            fn=fn,
            precision=precision,
            recall=recall,
            f1=f1,
        )
        speaker_summaries.append(summary)
        speaker_details[speaker_id] = {
            "summary": asdict(summary),
            "per_graph": per_graph_records,
        }

    return {
        "settings": {
            "data_root": str(data_root),
            "svd_preprocessor_path": str(svd_preprocessor_path),
            "global_epochs": global_epochs,
            "speaker_epochs": speaker_epochs,
            "speaker_learning_rate": speaker_learning_rate,
            "adaptation_ratio": adaptation_ratio,
            "threshold_quantile": threshold_quantile,
            "min_graphs_per_speaker": min_graphs_per_speaker,
            "max_speakers": max_speakers,
            "device": str(device),
            "global_config": asdict(config),
        },
        "global": {
            "num_speakers_used": len(per_speaker_graphs),
            "num_graphs_used": len(global_graphs),
            "losses": global_losses,
            "final_loss": global_losses[-1] if global_losses else None,
        },
        "speakers": [asdict(item) for item in speaker_summaries],
        "speaker_details": speaker_details,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Personalized graph anomaly detection on VoxPopuli embeddings (global pretraining + speaker adaptation)."
    )
    parser.add_argument("--data_root", type=str, default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--svd_preprocessor_path", type=str, default=str(DEFAULT_SVD_PREPROCESSOR_PATH))
    parser.add_argument("--results_path", type=str, default=str(DEFAULT_RESULTS_PATH))
    parser.add_argument("--max_speakers", type=int, default=None)
    parser.add_argument("--min_graphs_per_speaker", type=int, default=3)
    parser.add_argument("--adaptation_ratio", type=float, default=0.7)
    parser.add_argument("--threshold_quantile", type=float, default=0.95)
    parser.add_argument("--global_epochs", type=int, default=20)
    parser.add_argument("--speaker_epochs", type=int, default=8)
    parser.add_argument("--speaker_learning_rate", type=float, default=3e-4)

    args = parser.parse_args()

    result = run_personalized_anomaly_detection(
        data_root=args.data_root,
        svd_preprocessor_path=args.svd_preprocessor_path,
        max_speakers=args.max_speakers,
        min_graphs_per_speaker=args.min_graphs_per_speaker,
        adaptation_ratio=args.adaptation_ratio,
        threshold_quantile=args.threshold_quantile,
        global_epochs=args.global_epochs,
        speaker_epochs=args.speaker_epochs,
        speaker_learning_rate=args.speaker_learning_rate,
    )

    results_path = Path(args.results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved personalized anomaly results to {results_path}")
    print(
        f"Used {result['global']['num_speakers_used']} speakers and {result['global']['num_graphs_used']} graphs for global pretraining"
    )


if __name__ == "__main__":
    main()
