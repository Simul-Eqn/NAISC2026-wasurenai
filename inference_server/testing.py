
import io 

s = io.BytesIO() 

with open(r"inference_server\meralionv2_extraction\inference_server\meralionv2_extraction\outputs\ascend_graph_events.csv", 'rb') as f: 
    for i in range(5): 
        line = f.readline() 
        s.write(line)

s.seek(0)
import pandas as pd 
import numpy as np 


#df = pd.read_csv(s, index_col=0)
#print(df.head())
#print(np.array(eval(df['sensor_raw_values'].values[0])).shape)
#print(np.array(eval(df['sensor_raw_values'].values[1])).shape)








PCA_SIZE = 100 
USE_2D_PCA = False # TODO 

from pathlib import Path
from typing import Dict, List
import os 
import pandas as pd 
import numpy as np
#from sklearn.decomposition import SparsePCA 
from sklearn.decomposition import TruncatedSVD
from scipy import sparse 

# keeps the whole csv same except sensor raw values (of size (233, 19456) in the original csv) are reduced to (PCA_SIZE) using PCA 
df = pd.read_csv(s, index_col=0) 

arr = sparse.vstack([sparse.csr_matrix(np.array(eval(x)).ravel()) for x in df['sensor_raw_values']]) # ravel() to make it 1d 
df = df.drop(columns=['sensor_raw_values']) # drop to save memory 
print(arr.shape)

model = TruncatedSVD(n_components=PCA_SIZE, random_state=42)
transformed_arr = model.fit_transform(arr)
print(transformed_arr.shape)
df['sensor_raw_values'] = list(transformed_arr)

print(df) 



