OUT_SIZE = 100 

from pathlib import Path
from typing import Dict, List
import os 
import pandas as pd 
import numpy as np
#from sklearn.decomposition import SparsePCA 
from sklearn.decomposition import TruncatedSVD
from scipy import sparse 

in_csv = Path(r"inference_server\meralionv2_extraction\inference_server\meralionv2_extraction\outputs\ascend_graph_events.csv",)
out_csv = Path("inference_server/meralionv2_extraction/outputs/pca_{:04d}".format(OUT_SIZE) +".csv")
os.makedirs(out_csv.parent, exist_ok=True)

# keeps the whole csv same except sensor raw values (of size (233, 19456) in the original csv) are reduced to (PCA_SIZE) using PCA 
df = pd.read_csv(in_csv, index_col=0) 
print(len(df)) 

#arr = [sparse.csr_matrix(np.array(eval(x))[0]) for x in df['sensor_raw_values']] # ravel() to make it 1d 
arr = [sparse.csr_matrix( i )  for x in df['sensor_raw_values']  for i in np.array(eval(x)) ] # make it 1d  
print(len(arr)) 
first_size = arr[0].shape 
for i, a in enumerate(arr): 
    if a.shape != first_size: 
        print(f"Row {i} has shape {a.shape} which is different from the first row shape {first_size}")
arr = sparse.vstack(arr) 

dp_sizes = [len(np.array(eval(x))) for x in df['sensor_raw_values']]
df = df.drop(columns=['sensor_raw_values']) # drop to save memory 
print(arr.shape)

model = TruncatedSVD(n_components=OUT_SIZE, random_state=42)
transformed_arr = model.fit_transform(arr)
print(transformed_arr.shape)
# turn each data point at timestamp t to m different data points time 1/60s apart, where m is the number of rows in the original sensor raw values at timestamp t (which is 233 in the original csv)
rows_new = []
idx = 0
for row_idx, (_, row) in enumerate(df.iterrows()): 
    sensor_values = transformed_arr[idx:idx+dp_sizes[row_idx], :] 
    idx += dp_sizes[row_idx] 
    for j in range(sensor_values.shape[0]): 
        new_row = row.to_dict()
        new_row['sensor_raw_values'] = sensor_values[j, :].tolist() 
        rows_new.append(new_row)
df_new = pd.DataFrame(rows_new)
#df['sensor_raw_values'] = list(transformed_arr)

#print(df) 
df_new.to_csv(out_csv, index=False)
print(df_new)
