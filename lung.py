import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import lightgbm as lgb  # LightGBM (Faster than XGBoost)

# Load Data
df = pd.read_csv('dataset_med.csv')

# Convert date columns
df['diagnosis_date'] = pd.to_datetime(df['diagnosis_date'], errors='coerce')
df['end_treatment_date'] = pd.to_datetime(df['end_treatment_date'], errors='coerce')

# Compute treatment duration
df['treatment_duration'] = (df['end_treatment_date'] - df['diagnosis_date']).dt.days
df['treatment_duration'].fillna(df['treatment_duration'].median(), inplace=True)

# Handle missing values
df = df.ffill()
df = df.dropna(subset=['survived'])

# Encode categorical variables
label_encoders = {}
categorical_cols = ['gender', 'country', 'cancer_stage', 'family_history', 'smoking_status',
                    'hypertension', 'asthma', 'cirrhosis', 'other_cancer', 'treatment_type']

for col in categorical_cols:
    df[col] = df[col].astype(str)
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Normalize numerical features
scaler = StandardScaler()
numerical_cols = ['age', 'bmi', 'cholesterol_level', 'treatment_duration']
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Prepare Data
X = df.drop(['id', 'diagnosis_date', 'end_treatment_date', 'survived'], axis=1)
y = df['survived'].astype(int)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Train LightGBM (FASTEST MODEL)
lgb_model = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1, max_depth=7, random_state=42)
lgb_model.fit(X_train, y_train)

# Evaluate Model
y_pred_lgb = lgb_model.predict(X_test)

print("\n===== LightGBM Performance =====")
print("Accuracy:", accuracy_score(y_test, y_pred_lgb))
print("Precision:", precision_score(y_test, y_pred_lgb, zero_division=1))
print("Recall:", recall_score(y_test, y_pred_lgb))
print("F1 Score:", f1_score(y_test, y_pred_lgb))
print("ROC-AUC:", roc_auc_score(y_test, lgb_model.predict_proba(X_test)[:, 1]))

# Save Model
joblib.dump({'model': lgb_model, 'scaler': scaler, 'encoders': label_encoders}, 'lung_cancer_model.pkl')
print("\n Model saved as 'lung_cancer_model.pkl' ")
