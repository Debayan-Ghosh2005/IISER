import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# === Step 1: Load dataset ===
df = pd.read_csv("data.csv")

# === Step 2: One-shot labeled dataset (1 sample per class) ===
one_shot_labeled = df.groupby('label').apply(lambda x: x.sample(1, random_state=42)).reset_index(drop=True)
unlabeled_data = df.drop(one_shot_labeled.index)

# === Step 3: Preprocessing ===
features = ['age', 'physical_activity_days', 'processed_food_meals', 
            'sleep_hours', 'smoking_status', 'alcohol_consumption']

scaler = StandardScaler()
X_labeled = scaler.fit_transform(one_shot_labeled[features])
y_labeled = one_shot_labeled['label']

X_unlabeled = scaler.transform(unlabeled_data[features])
y_unlabeled_true = unlabeled_data['label']

# === Step 4: Train and Predict ===
model = RandomForestClassifier(random_state=42)
model.fit(X_labeled, y_labeled)
y_pred = model.predict(X_unlabeled)

# === Step 5: Evaluate ===
f1 = f1_score(y_unlabeled_true, y_pred)
print(f"One-Shot F1 Score: {f1:.4f}")
