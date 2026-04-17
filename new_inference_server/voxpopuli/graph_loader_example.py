from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from new_inference_server.voxpopuli.graph_loader import load_speaker_graphs


if __name__ == "__main__":
    #graphs = build_utterance_graphs("new_inference_server/voxpopuli/data")
    graphs = load_speaker_graphs("new_inference_server/voxpopuli/data", "1185", use_svd_preprocessor=True) 
    print(f"Loaded {len(graphs)} utterance graphs")
    print(graphs[0].node_features.shape if graphs else "No graphs found")
    