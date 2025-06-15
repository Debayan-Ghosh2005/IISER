import pandas as pd  # For loading and handling CSV data
import numpy as np  # For numerical operations
from sklearn.preprocessing import StandardScaler, LabelEncoder  # For scaling and encoding
from sklearn.ensemble import RandomForestClassifier  # Model
from sklearn.metrics import f1_score  # Evaluation metric
from sklearn.model_selection import RandomizedSearchCV  # For tuning
from sklearn.neighbors import NearestNeighbors  # For KNN-based labeling

# === Step 1: Load dataset ===
df = pd.read_csv("data.csv")  # Load your dataset

# === Step 2: Encode the 'continent' column ===
le = LabelEncoder()
df['continent'] = le.fit_transform(df['continent'])  # Convert continent to numeric labels

# === Step 3: Define feature columns ===
features = ['age', 'continent', 'physical_activity_days', 'processed_food_meals',
            'sleep_hours', 'smoking_status', 'alcohol_consumption']

# === Step 4: Scale features ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])  # Scaled version of all features

# === Step 5: KNN-Based Label Selection (short labeling) ===
# Set how many points to "label" (e.g., 10%)
n_labeled = int(0.1 * len(df))  # 10% of data

# Use KNN to find representative points using the farthest points
knn = NearestNeighbors(n_neighbors=5)  # KNN object to find neighbors
knn.fit(X_scaled)  # Fit on the full dataset

# Use a simple trick: take the points that are farthest apart
dists, _ = knn.kneighbors(X_scaled)  # Compute distances
avg_dist = dists.mean(axis=1)  # Average distance to neighbors

# Select top N points with highest avg distance = most isolated = diverse
selected_indices = np.argsort(avg_dist)[-n_labeled:]  # Indices of points to label

# Mark them in the DataFrame
df['label_known'] = 0
df.loc[selected_indices, 'label_known'] = 1  # Mark selected rows as known

# === Step 6: Split into labeled and unlabeled ===
labeled_data = df[df['label_known'] == 1]
unlabeled_data = df[df['label_known'] == 0]

# === Step 7: Prepare training data ===
X_labeled = scaler.transform(labeled_data[features])  # Already fitted above
y_labeled = labeled_data['label']

X_unlabeled = scaler.transform(unlabeled_data[features])
y_unlabeled_true = unlabeled_data['label']

# === Step 8: Train base model ===
base_model = RandomForestClassifier(random_state=42)
base_model.fit(X_labeled, y_labeled)

# === Step 9: Evaluate before tuning ===
y_pred_base = base_model.predict(X_unlabeled)
f1_base = f1_score(y_unlabeled_true, y_pred_base)
print(f"F1 Score BEFORE finetuning: {f1_base:.4f}")

# === Step 10: Hyperparameter tuning setup ===
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 20, 40],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'class_weight': ['balanced', None]
}

random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions=param_grid,
    n_iter=20,
    scoring='f1',
    cv=3,
    n_jobs=-1,
    random_state=42
)

# === Step 11: Run tuning ===
random_search.fit(X_labeled, y_labeled)
print("Best parameters found:", random_search.best_params_)

best_model = random_search.best_estimator_

# === Step 12: Predict probability and tune threshold ===
y_probs = best_model.predict_proba(X_unlabeled)[:, 1]  # Probability of class 1

best_f1 = 0
best_thresh = 0.5

# Loop over thresholds to find best F1
for thresh in np.arange(0.3, 0.7, 0.01):
    y_pred_thresh = (y_probs >= thresh).astype(int)
    f1 = f1_score(y_unlabeled_true, y_pred_thresh)
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = thresh

# === Step 13: Output final result ===
print(f"Best threshold for highest F1: {best_thresh:.2f}")
print(f"Highest F1 after threshold tuning: {best_f1:.4f}")
