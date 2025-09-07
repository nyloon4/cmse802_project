# Predicting Online Shopper Conversions

This project predicts whether an online shopper will convert based on session-level behavior.  
I trained Logistic Regression and XGBoost with class weighting, prioritizing **recall** to handle class imbalance.

**Tech stack:** Python · scikit-learn · XGBoost · NumPy · Pandas · Jupyter

## Dataset
- Source: [UCI Online Shoppers Purchasing Intention Dataset](https://archive.ics.uci.edu/ml/datasets/Online+Shoppers+Purchasing+Intention+Dataset)  
- Not included in this repo; place the file in `data/` before running

## Quickstart
```bash
pip install -r requirements.txt
jupyter notebook "notebooks/buyer final project code.ipynb"

Open the Jupyter notebook:
jupyter notebook notebooks/buyer final project code.ipynb

In Jupyter, select Run All to reproduce training and evaluation.

## Results (best model)

| Model                 | Accuracy | Precision | Recall | F1   | ROC-AUC |
|-----------------------|---------:|----------:|-------:|-----:|--------:|
| Logistic Regression   | 0.883    | 0.764     | 0.356  | 0.486 | 0.668  |
| XGBoost               | 0.889    | 0.670     | 0.563  | 0.612 | 0.756  |
| XGBoost (Weighted)    | 0.878    | 0.595     | 0.675  | 0.632 | 0.795  |

> **XGBoost (Weighted)** achieved the highest recall (0.675) and best balance across F1 and ROC-AUC, making it the preferred model given class imbalance.



## Repo Structure  
- **`data/`** → dataset (not committed)
- **`scripts/`** → individual .py files 
- **`notebooks/`** → jupyter notebook for modeling + analysis  
- **`plots/`** → visual outputs used in presentation  
