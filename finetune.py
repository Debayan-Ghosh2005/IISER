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

labeled_data = df[df['label_known'] == 1]
unlabeled_data = df[df['label_known'] == 0]

# === 4. Features Selection ===
features = ['age', 'continent', 'physical_activity_days', 'processed_food_meals',
            'sleep_hours', 'smoking_status', 'alcohol_consumption']

# === 5. Scaling ===
scaler = StandardScaler()
X_labeled = scaler.fit_transform(labeled_data[features])
y_labeled = labeled_data['label']

X_unlabeled = scaler.transform(unlabeled_data[features])
y_unlabeled_true = unlabeled_data['label']

# === 6. Original RF ===
original_rf = RandomForestClassifier(random_state=42)
original_rf.fit(X_labeled, y_labeled)
y_pred_original = original_rf.predict(X_unlabeled)
f1_original = f1_score(y_unlabeled_true, y_pred_original)
print(f"\nOriginal Model F1 Score: {f1_original:.4f}")

# === 7. Tuned RF ===
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

# === 8. Save Predictions to df ===
df.loc[unlabeled_data.index, 'pred_original'] = y_pred_original
df.loc[unlabeled_data.index, 'pred_tuned'] = y_pred_tuned

# === 9. Confusion Matrices ===
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

cm_original = confusion_matrix(y_unlabeled_true, y_pred_original)
cm_tuned = confusion_matrix(y_unlabeled_true, y_pred_tuned)

ConfusionMatrixDisplay(cm_original).plot(ax=axes[0])
axes[0].set_title(f"Original RF\nF1: {f1_original:.4f}")

ConfusionMatrixDisplay(cm_tuned).plot(ax=axes[1])
axes[1].set_title(f"Tuned RF\nF1: {f1_tuned:.4f}")

plt.tight_layout()
plt.show()

# === 10. Save CSV ===
df.to_csv("comparison_predictions.csv", index=False)
print("\nPredictions saved to 'comparison_predictions.csv'")
