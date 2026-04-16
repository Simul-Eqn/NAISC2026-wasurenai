
import io 

s = io.BytesIO() 

with open(r"inference_server\meralionv2_extraction\inference_server\meralionv2_extraction\outputs\ascend_graph_events.csv", 'rb') as f: 
    for i in range(10): 
        line = f.readline() 
        s.write(line)

s.seek(0)
import pandas as pd 
import numpy as np 
df = pd.read_csv(s, index_col=0)
print(df.head())
print(np.array(eval(df['sensor_raw_values'].values[0])).shape)
print(np.array(eval(df['sensor_raw_values'].values[1])).shape)
