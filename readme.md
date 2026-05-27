# Synthetic ECG Data Evaluation with FPCA & Isomap

## Code Structure
```text
time_series_fpca/
|--experiments/
|   # Invidual component experiment trials
|   |==basis_smoothing_test.py      
|   |==fpca_test.py                  
|   |==isomap_test.py
|--methods/
|   # Transformation & evaluation methods
|   |==cfpca.py
|   |==fidelity_evaluation.py
|   |==fpca.py
|   |==isomap.py
|   |==preprocess.py
|   |==privacy_domias.py
|   |==utils.py
|--pipelines/
|   # Evaluation pipelines
|   |==fidelity.py
|   |==privacy.py
|--.gitignore
|--readme.md
|--requirement.txt
```

## Execution
### Fidelity Evaluation
```
# cd time_series_fpca
python -m pipelines.fidelity
```
### Privacy Evaluation
```
# cd time_series_fpca
python -m pipelines.privacy
```