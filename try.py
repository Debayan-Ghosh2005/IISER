import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

# === 1. Load dataset ===
df = pd.read_csv("data.csv")

# === 2. Encode categorical ===
le = LabelEncoder()
df['continent'] = le.fit_transform(df['continent'])

# === 3. Few-Shot Labeling (10%) ===
np.random.seed(42)
df['label_known'] = 0
known_indices = np.random.choice(df.index, size=int(0.1 * len(df)), replace=False)
df.loc[known_indices, 'label_known'] = 1

# === 4. Feature Engineering ===
df['activity_per_sleep'] = df['physical_activity_days'] / (df['sleep_hours'] + 1)

# === 5. Re-split after feature engineering ===
labeled_data = df[df['label_known'] == 1]
unlabeled_data = df[df['label_known'] == 0]

# === 6. Feature Selection ===
features = ['age', 'continent', 'physical_activity_days', 'processed_food_meals',
            'sleep_hours', 'smoking_status', 'alcohol_consumption', 'activity_per_sleep']

# === 7. Scaling ===
scaler = StandardScaler()
X_labeled = scaler.fit_transform(labeled_data[features])
y_labeled = labeled_data['label']

X_unlabeled = scaler.transform(unlabeled_data[features])
y_unlabeled_true = unlabeled_data['label']

# === 8. Original RF ===
original_rf = RandomForestClassifier(random_state=42)
original_rf.fit(X_labeled, y_labeled)
y_pred_original = original_rf.predict(X_unlabeled)
f1_original = f1_score(y_unlabeled_true, y_pred_original)
print(f"\nOriginal Model F1 Score: {f1_original:.4f}")

# === 9. Tuned RF with GridSearch ===
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    scoring='f1',
    cv=3,
    n_jobs=-1,
    verbose=2
)

print("\nStarting Grid Search...")
grid_search.fit(X_labeled, y_labeled)
print(f"\nBest Parameters: {grid_search.best_params_}")

best_rf = grid_search.best_estimator_
y_pred_tuned = best_rf.predict(X_unlabeled)
f1_tuned = f1_score(y_unlabeled_true, y_pred_tuned)
print(f"Tuned Model F1 Score: {f1_tuned:.4f}")

# === 10. One-shot Pseudo Labeling ===
y_proba = best_rf.predict_proba(X_unlabeled)
confidence_threshold = 0.95  # You can tune this
high_conf_indices = np.where(np.max(y_proba, axis=1) >= confidence_threshold)[0]

pseudo_X = X_unlabeled[high_conf_indices]
pseudo_y = np.argmax(y_proba[high_conf_indices], axis=1)
