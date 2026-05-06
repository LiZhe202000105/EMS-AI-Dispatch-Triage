import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.sparse import hstack
import xgboost as xgb
import shap
import warnings
warnings.filterwarnings('ignore')

# 1. Load External Chest Pain Cohort
df_c = pd.read_csv("EMS_ChestPain_External_Val.csv")
df_c['Chief_Complaint'] = df_c['Chief_Complaint'].fillna("")
numeric_cols = ['Age', 'Sex_Male']

# 2. Zero-Shot Inference (Assuming vectorizer and model are pre-trained)
# X_chest_text = vectorizer.transform(df_c['Chief_Complaint'].apply(jieba_tokenize))
# X_chest = hstack([X_chest_text, df_c[numeric_cols].values])
# df_c['AI_Prob'] = model.predict_proba(X_chest)[:, 1]

# 3. Calculate Cross-Domain Metrics
# auc_c = roc_auc_score(df_c['Target_PCE'], df_c['AI_Prob'])
# print(f"External Validation AUROC: {auc_c}")

# 4. SHAP Interpretability Analysis (Semantic Inversion)
# X_test_dense = X_test.tocsr()[:1000].toarray()
# explainer = shap.TreeExplainer(model)
# shap_values = explainer.shap_values(X_test_dense)