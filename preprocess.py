## 🎯 PROJECT: Predicting Buyer Intent from Online Shopper Behavio
# the goal of this project is to build a classification model that **predicts whether an online shopper will complete a purchase based on their session behavior.**
#this helps businesses target potential buyers more effectively and avoid wasting ad spend on users who are unlikely to convert.

# import libraries
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

### 🧠 step 2: load + prep data    
# - convert Revenue and Weekend from true/false to 0/1  
# - use label encoding to turn text data (Month, VisitorType) into numbers

#🔍 why: models can’t work with text or boolean — it needs all numeric input.

# load csv
df = pd.read_csv("/Users/Nylab/Downloads/online_shoppers_intention.csv")

# make a copy so og stays safe
data = df.copy()

# turn true/false to 0/1
data['Weekend'] = data['Weekend'].astype(int)
data['Revenue'] = data['Revenue'].astype(int)

# label encode strings
le_month = LabelEncoder()
le_visitor = LabelEncoder()
data['Month'] = le_month.fit_transform(data['Month'])
data['VisitorType'] = le_visitor.fit_transform(data['VisitorType'])

# preview the data
df.head()

### 🧠 step 2.5: normalize features  
# normalize the data so all features are on the same scale.  
# this helps logistic regression (and other models) converge better, especially when some columns (like durations) are way bigger than others (like bounce rate).

#🔍 why: models like logistic regression assume features are scaled similarly  
#🔑 look for: warning messages disappearing + better performance

from sklearn.preprocessing import StandardScaler

# split feat + target
X = data.drop(columns=['Revenue'])
y = data['Revenue']

# scale feat
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

### 🧠 step 3: split into train/test  
# split our dataset into two parts:  
# - training (80%) → to teach the model  
# - testing (20%) → to check how well it learned

#🔍 why: we need a clean way to test performance on "new" data  
#🔑 look for: balance (so we get a fair mix of buyers and non-buyers)

# split train + test (80/20)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, stratify=y, random_state=42, test_size=0.2)

