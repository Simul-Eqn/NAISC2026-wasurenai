PCA_SIZE = 128 
USE_2D_PCA = False # TODO 

from pathlib import Path
from typing import Dict, List
import os 
import pandas as pd 
import numpy as np
from sklearn.decomposition import SparsePCA 

in_csv = Path(r"inference_server\meralionv2_extraction\inference_server\meralionv2_extraction\outputs\ascend_graph_events.csv",)
out_csv = Path("inference_server/meralionv2_extraction/outputs/pca_{:04d}".format(PCA_SIZE)+("_2D" if USE_2D_PCA else "")+".csv")
os.makedirs(out_csv.parent, exist_ok=True)

# keeps the whole csv same except sensor raw values (of size (233, 19456) in the original csv) are reduced to (PCA_SIZE) using PCA 
df = pd.read_csv(in_csv, index_col=0) 

arr = df['sensor_raw_values'].apply(lambda x: np.array(eval(x))).values 

sparsePCA_model = SparsePCA(n_components=PCA_SIZE, random_state=42)
sparsePCA_model.fit(arr)
transformed_arr = sparsePCA_model.transform(arr)
df['sensor_raw_values'] = transformed_arr 
df.to_csv(out_csv)
