import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import TfidfVectorizer

# Load and clean the dataset
df = pd.read_csv("NEW/IMDB.csv", encoding="latin1")

# Normalize column names
df.columns = df.columns.str.strip().str.lower()
print("Available columns:", df.columns.tolist())  # Debug info

# Try to rename columns if needed
if 'text' not in df.columns or 'label' not in df.columns:
    if 'review' in df.columns:
        df.rename(columns={'review': 'text'}, inplace=True)
    if 'sentiment' in df.columns:
        df.rename(columns={'sentiment': 'label'}, inplace=True)

# Final check
assert 'text' in df.columns and 'label' in df.columns, "Dataset must have 'text' and 'label' columns"

# Step 1: Simulate Few-Shot (10% labeled data)
np.random.seed(42)
df['label_known'] = 0
known_indices = np.random.choice(df.index, size=int(0.1 * len(df)), replace=False)
df.loc[known_indices, 'label_known'] = 1

# Step 2: Split labeled and unlabeled
labeled_data = df[df['label_known'] == 1]
unlabeled_data = df[df['label_known'] == 0]

# Step 3: TF-IDF vectorization
vectorizer = TfidfVectorizer(max_features=1000)
X_labeled = vectorizer.fit_transform(labeled_data['text'])
y_labeled = labeled_data['label']

X_unlabeled = vectorizer.transform(unlabeled_data['text'])
y_unlabeled_true = unlabeled_data['label']

# Step 4: Train RandomForest
model = RandomForestClassifier(random_state=42)
model.fit(X_labeled, y_labeled)

# Step 5: Evaluate
y_pred = model.predict(X_unlabeled)
f1 = f1_score(y_unlabeled_true, y_pred)

print(f"F1 Score on Unlabeled Data: {f1:.4f}")
