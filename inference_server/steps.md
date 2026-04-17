## Steps 

1. run meralionv2_extraction's run.py (with the suitable conda environment), in the directory inference_server/meralionv2_extraction 
2. run dimensionality_reduction.py from the directory of this repo 
3. cd dementia_graph_barlow and 
```bash 
python -m src.cli \
  --events_csv ../meralionv2_extraction/outputs/pca_0100.csv \
  --config ../meralionv2_extraction/meralion_config.yaml \
  --output_dir ../meralionv2_extraction/outputs/extract_0100
``` 
