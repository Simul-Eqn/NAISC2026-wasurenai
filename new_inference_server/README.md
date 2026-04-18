## PROCESS 

### Data preparation 
```conda activate ...``` (environment from inference_server/meralionv2_extraction)
```cd new_inference_server```
```cd voxpopuli``` 
```python run_meralion.py``` 
```deactivate``` 
```python svd_preprocessor.py``` 

Testing: 
```python graph_loader_example.py``` 

### Training 

```cd new_inference_server```
```cd voxpopuli``` 
```python personalized_anomaly_detction.py``` 

Saved personalized anomaly results to ...\new_inference_server\voxpopuli\personalized_anomaly_results.json
Used 50 speakers and 846 graphs for global pretraining

A debug I did: 
```python personalized_anomaly_detection.py --max_speakers 3 --global_epochs 2 --speaker_epochs 1 --results_path ./test_pad_results.json``` 
