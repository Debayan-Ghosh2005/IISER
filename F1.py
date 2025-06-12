import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# === Step 1: Load dataset ===
df = pd.read_csv("data.csv")

# Simulate 'label_known' column (assume only 10% are true known labels)
np.random.seed(42)
df['label_known'] = 0
known_indices = np.random.choice(df.index, size=int(0.1 * len(df)), replace=False)
df.loc[known_indices, 'label_known'] = 1

# === Step 2: Split labeled and unlabeled ===
labeled_data = df[df['label_known'] == 1]
unlabeled_data = df[df['label_known'] == 0]

# === Step 3: Preprocess features ===
features = ['age', 'physical_activity_days', 'processed_food_meals', 
            'sleep_hours', 'smoking_status', 'alcohol_consumption']
scaler = StandardScaler()

X_labeled = scaler.fit_transform(labeled_data[features])
y_labeled = labeled_data['label']

X_unlabeled = scaler.transform(unlabeled_data[features])
y_unlabeled_true = unlabeled_data['label']  # We assume we secretly have the true value to evaluate

# === Step 4: Train model on few-shot data ===
model = RandomForestClassifier(random_state=42)
model.fit(X_labeled, y_labeled)

# === Step 5: Predict and Evaluate ===
y_pred = model.predict(X_unlabeled)
f1 = f1_score(y_unlabeled_true, y_pred)

print(f"F1 Score on Unlabeled Data: {f1:.4f}")