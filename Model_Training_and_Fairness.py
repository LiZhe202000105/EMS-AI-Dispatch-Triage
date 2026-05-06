import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, roc_auc_score, precision_score, recall_score
from scipy.sparse import hstack
import jieba
import warnings
warnings.filterwarnings('ignore')

# 1. Data Loading and Sanitization
df = pd.read_csv("EMS_Trauma_NPJ_Ready.csv")
df['Chief_Complaint'] = df['Chief_Complaint'].fillna("")
numeric_cols = ['Age', 'Sex_Male']
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

train_df = df[df['Dataset_Split'] == 'Train'].copy()
test_df = df[df['Dataset_Split'] == 'Test'].copy()

# 2. NLP Pipeline
def jieba_tokenize(text):
    return " ".join(jieba.lcut(text))
vectorizer_word = TfidfVectorizer(ngram_range=(1, 3), max_features=500)
X_text_train = vectorizer_word.fit_transform(train_df['Chief_Complaint'].apply(jieba_tokenize))
X_text_test = vectorizer_word.transform(test_df['Chief_Complaint'].apply(jieba_tokenize))

X_train = hstack([X_text_train, train_df[numeric_cols].values])
X_test = hstack([X_text_test, test_df[numeric_cols].values])
y_train = train_df['Target_PCE'].values
y_test = test_df['Target_PCE'].values

# 3. Model Training with Class Weighting
weight_ratio = float(len(y_train[y_train == 0])) / sum(y_train == 1)
model = xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, 
                          scale_pos_weight=weight_ratio, random_state=42)
model.fit(X_train, y_train)
test_df['AI_Prob'] = model.predict_proba(X_test)[:, 1]

# 4. Calibration and Subgroup Fairness Evaluation
subgroups = {
    'Age >= 65': test_df['Age'] >= 65,
    'Age < 65': test_df['Age'] < 65,
    'Male': test_df['Sex_Male'] == 1,
    'Female': test_df['Sex_Male'] == 0
}

for name, mask in subgroups.items():
    y_true_sub = y_test[mask]
    y_prob_sub = test_df.loc[mask, 'AI_Prob']
    brier = brier_score_loss(y_true_sub, y_prob_sub)
    print(f"Subgroup {name}: Brier Score = {brier:.4f}")