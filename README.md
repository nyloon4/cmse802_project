# Predicting Online Shopper Conversions

This project predicts whether an online shopper will convert based on session-level behavior.  
I trained Logistic Regression and XGBoost with class weighting, prioritizing **recall** to handle class imbalance.

**Tech stack:** Python · scikit-learn · XGBoost · NumPy · Pandas · Jupyter

## Dataset
- Source: [UCI Online Shoppers Purchasing Intention Dataset](https://archive.ics.uci.edu/ml/datasets/Online+Shoppers+Purchasing+Intention+Dataset)  
- Not included in this repo; place dataset file(s) in `data/` before running

## Quickstart
To set up and run this project locally:

1. Clone the repo and install dependencies:
   ```bash
   pip install -r requirements.txt
Open the Jupyter notebook:

## folder structure  
- **`data/`** → dataset (not committed)
- **`scipts/`** → individual .py files 
- **`notebooks/`** → jupyter notebook for modeling + analysis  
- **`plots/`** → visual outputs used in presentation  
- **`docs/`** → report writeup or slide materials  
- **`models/`** → trained models (optional)
