# evaluate.py
# evaluates the performance of all trained models using accuracy,
# precision, recall, f1 score, and AUC. also generates visualizations
# like confusion matrices, bar charts, and ROC curves

### 🧠 step 6: evaluate all 3 models  
# we're comparing:  
# 1. **logistic regression** – simple, interpretable  
# 2. **xgboost** – strong baseline, better recall  
# 3. **weighted xgboost** – same as xgboost but gives more attention to buyers

#we’re evaluating using:
#- **accuracy**: total correctness  
#- **precision**: how many predicted buyers were actually buyers  
#-**recall**: how many real buyers the model found  
#- **f1 score**: balance of prec + recall  
#- **auc**: overall classification strength  
#- **confusion matrix**: see where each model succeeds/fails

#🔍 what to look for:
#- higher **recall** = better at finding buyers  
#- higher **f1** = better balance  
#- **auc > 0.75** = strong model

# create comparison df
scores = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC'],
    'Logistic': [0.883, 0.764, 0.356, 0.486, 0.668],
    'XGBoost': [0.889, 0.670, 0.563, 0.612, 0.756],
    'Weighted XGB': [0.878, 0.594, 0.675, 0.632, 0.795]
})

# plot
scores.set_index('Metric').plot(kind='bar', figsize=(10,6))
plt.title("📊 Model Comparison: Logistic vs XGBoost vs Weighted XGB")
plt.ylabel("Score")
plt.ylim(0,1)
plt.xticks(rotation=0)
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

from sklearn.metrics import ConfusionMatrixDisplay

# logistic
ConfusionMatrixDisplay.from_predictions(y_test, log_preds, display_labels=["No", "Yes"])
plt.title("Confusion Matrix - Logistic")
plt.show()

# xgboost
ConfusionMatrixDisplay.from_predictions(y_test, xgb_preds, display_labels=["No", "Yes"])
plt.title("Confusion Matrix - XGBoost")
plt.show()

# weighted xgboost
ConfusionMatrixDisplay.from_predictions(y_test, xgb_weighted_preds, display_labels=["No", "Yes"])
plt.title("Confusion Matrix - Weighted XGBoost")
plt.show()

from sklearn.metrics import RocCurveDisplay

# create main figure and axis
fig, ax = plt.subplots(figsize=(8,6))

# baseline line (random guess)
ax.plot([0,1], [0,1], linestyle='--', color='gray', label='Random Guessing (Baseline)')

# plot all roc curves on the same axis
RocCurveDisplay.from_predictions(y_test, log_preds, name="Logistic", color='blue').plot(ax=ax)
RocCurveDisplay.from_predictions(y_test, xgb_preds, name="XGBoost", color='orange').plot(ax=ax)
RocCurveDisplay.from_predictions(y_test, xgb_weighted_preds, name="Weighted XGBoost", color='green').plot(ax=ax)

# formatting
ax.set_title("ROC Curve - All Models")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.grid(alpha=0.3)
ax.legend(loc='lower right')
plt.tight_layout()
plt.show()

