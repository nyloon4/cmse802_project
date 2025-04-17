# utils.py
# contains reusable helper functions such as the metric scoring function
# used to evaluate all models

# define score function to print metrics
def get_scores(y_true, preds, name):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
    print(f"\n{name} results:")
    print("acc:", accuracy_score(y_true, preds))
    print("prec:", precision_score(y_true, preds))
    print("recall:", recall_score(y_true, preds))
    print("f1:", f1_score(y_true, preds))
    print("roc auc:", roc_auc_score(y_true, preds))
    print("conf matrix:\n", confusion_matrix(y_true, preds))
