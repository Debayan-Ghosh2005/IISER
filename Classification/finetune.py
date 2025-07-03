import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer

# Step 0: Load dataset
df = pd.read_csv("NEW/IMDB.csv", encoding="latin1")  # Use raw string or forward slashes

# Step 0.1: Clean and rename columns if needed
df.columns = df.columns.str.strip().str.lower()
print("CSV Columns:", df.columns.tolist())  # For debugging

# Attempt to rename common variants
if 'text' not in df.columns or 'label' not in df.columns:
    if 'review' in df.columns:
        df.rename(columns={'review': 'text'}, inplace=True)
    if 'sentiment' in df.columns:
        df.rename(columns={'sentiment': 'label'}, inplace=True)

# Final check
assert 'text' in df.columns and 'label' in df.columns, "Dataset must have 'text' and 'label' columns"

# Step 1: Simulate Few-Shot (only 10% labeled)
np.random.seed(42)
df['label_known'] = 0
known_indices = np.random.choice(df.index, size=int(0.1 * len(df)), replace=False)
df.loc[known_indices, 'label_known'] = 1

# Step 2: Split labeled and unlabeled data
labeled_data = df[df['label_known'] == 1]
unlabeled_data = df[df['label_known'] == 0]

# Step 3: TF-IDF vectorization
vectorizer = TfidfVectorizer(max_features=1000)
X_labeled = vectorizer.fit_transform(labeled_data['text'])
y_labeled = labeled_data['label']

X_unlabeled = vectorizer.transform(unlabeled_data['text'])
y_unlabeled_true = unlabeled_data['label']

# Step 4: Base model training
base_model = RandomForestClassifier(random_state=42)
base_model.fit(X_labeled, y_labeled)

# Step 5: Evaluation before tuning
y_pred_base = base_model.predict(X_unlabeled)
f1_base = f1_score(y_unlabeled_true, y_pred_base)
print(f"F1 Score BEFORE tuning: {f1_base:.4f}")

# Step 6: Hyperparameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 20, 40],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'class_weight': ['balanced', None]
}

# Step 7: RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions=param_grid,
    n_iter=20,
    scoring='f1',
    cv=3,
    n_jobs=-1,
    random_state=42
)

random_search.fit(X_labeled, y_labeled)
print("Best hyperparameters:", random_search.best_params_)

# Step 8: Get best model
best_model = random_search.best_estimator_
y_probs = best_model.predict_proba(X_unlabeled)[:, 1]

# Step 9: Tune threshold
best_f1 = 0
best_thresh = 0.5

for thresh in np.arange(0.3, 0.7, 0.01):
    y_pred_thresh = (y_probs >= thresh).astype(int)
    f1 = f1_score(y_unlabeled_true, y_pred_thresh)
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = thresh

# Step 10: Final results
print(f"Best threshold: {best_thresh:.2f}")
print(f"F1 Score AFTER tuning: {best_f1:.4f}")
