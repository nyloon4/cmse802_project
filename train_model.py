### 🧠 step 4: Logistic regression  
# train a baseline model using logistic regression
# since data was scaled earlier, this should now run without any convergence warnings.

#🔍 why: gives a starting point to compare future models  
#🧠 interpretability: it’s easier to explain than other black-box models

# logistic model (baseline)
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)
log_preds = logreg.predict(X_test)

### 🧠 step 5: Xgboost  
# now train a more advanced model —> XGBoost.  
# this is a boosting algorithm that usually improves performance over simpler models.

#🔍 why: want to see if a more complex model improves accuracy, recall, or AUC  
#🔑 look for: better scores than logistic regression

# xgboost model
xgb_model = xgb.XGBClassifier(eval_metric='logloss')
xgb_model.fit(X_train, y_train)
xgb_preds = xgb_model.predict(X_test)

### 🧠 step 5.5: Weighted xgboost  
# now im testing xgboost again but im adding weight so it focuses more on the ppl who actually buy  
# since the dataset has wayyy more non-buyers, this should help the model catch more real buyers

#🔍 why: our data has way more non-buyers than buyers  
#🧠 goal: increase **recall** by helping the model find more actual buyers  
#✅ look for: higher recall, possibly higher f1 or auc too

# calc class weight (majority / minority)
scale = (y == 0).sum() / (y == 1).sum()

# re-train xgb with weighting
xgb_weighted = xgb.XGBClassifier(eval_metric='logloss', scale_pos_weight=scale)
xgb_weighted.fit(X_train, y_train)
xgb_weighted_preds = xgb_weighted.predict(X_test)

# eval
get_scores(y_test, xgb_weighted_preds, "xgboost (weighted)")

# train weighted xgboost
scale = (y == 0).sum() / (y == 1).sum()
xgb_weighted.fit(X_train, y_train)
xgb_weighted_preds = xgb_weighted.predict(X_test)