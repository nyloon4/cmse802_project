# Predicting Online Shopper Conversions

This project predicts whether an online shopper will convert based on session-level behavior.  
I trained Logistic Regression and XGBoost with class weighting, prioritizing **recall** to handle class imbalance.

**Tech stack:** Python · scikit-learn · XGBoost · NumPy · Pandas · Jupyter

## Dataset
- Source: [UCI Online Shoppers Purchasing Intention Dataset](https://archive.ics.uci.edu/ml/datasets/Online+Shoppers+Purchasing+Intention+Dataset)  
- Not included in this repo; place the file in `data/` before running

## Quickstart
To set up and run this project locally:

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Open the Jupyter notebook:
   ```bash
   jupyter notebook "notebooks/buyer final project code.ipynb"
   ```

3. In Jupyter, select **Run All** to reproduce training and evaluation.

## Results (best model)

| Model               | Accuracy | Precision | Recall | F1   | ROC-AUC |
|---------------------|---------:|----------:|-------:|-----:|--------:|
| Logistic Regression | 0.883    | 0.764     | 0.356  | 0.486 | 0.668  |
| XGBoost             | 0.889    | 0.670     | 0.563  | 0.612 | 0.756  |
| XGBoost (Weighted)  | 0.878    | 0.595     | 0.675  | 0.632 | 0.795  |

> **XGBoost (Weighted)** had the highest recall (0.675) and best F1 + ROC-AUC, making it the preferred model given class imbalance.

## Key methods
- Class weighting for imbalanced data  
- Evaluation with Recall, F1, Accuracy, and ROC-AUC  
- Confusion matrices in the notebook

## Repo structure
- `notebooks/` – modeling & evaluation  
- `scripts/` – reusable code (preprocessing, training)  
- `plots/` – saved figures (optional)  
- `data/` – (ignored, add your own data)  
- `requirements.txt`
