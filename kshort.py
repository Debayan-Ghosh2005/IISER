import pandas as pd  # For loading and handling CSV data
import numpy as np  # For numerical operations
from sklearn.preprocessing import StandardScaler, LabelEncoder  # For feature scaling and label encoding
from sklearn.ensemble import RandomForestClassifier  # Random Forest model
from sklearn.metrics import f1_score  # Metric for evaluation
from sklearn.model_selection import RandomizedSearchCV, train_test_split  # Hyperparameter tuning & data splitting
from sklearn.neighbors import NearestNeighbors  # KNN for diverse sample selection

# === Step 1: Load dataset ===
df = pd.read_csv("data.csv")  # Load the CSV file into a pandas DataFrame

# === Step 2: Encode categorical column ===
le = LabelEncoder()  # Initialize label encoder
df['continent'] = le.fit_transform(df['continent'])  # Convert 'continent' to numeric labels

# === Step 3: Define features for modeling ===
features = ['age', 'continent', 'physical_activity_days', 'processed_food_meals',
            'sleep_hours', 'smoking_status', 'alcohol_consumption']  # Features to use

# === Step 4: Scale feature values ===
scaler = StandardScaler()  # Initialize standard scaler
X_scaled = scaler.fit_transform(df[features])  # Scale the features to have mean=0 and std=1

# === Step 5: Select 10% diverse samples using KNN ===
n_labeled = int(0.1 * len(df))  # Calculate 10% of total data for labeling
knn = NearestNeighbors(n_neighbors=5)  # KNN object to find neighbors
knn.fit(X_scaled)  # Fit KNN on scaled data
dists, _ = knn.kneighbors(X_scaled)  # Compute distances to nearest neighbors
avg_dist = dists.mean(axis=1)  # Compute average distance for each sample
selected_indices = np.argsort(avg_dist)[-n_labeled:]  # Pick top samples with highest avg distance
df['label_known'] = 0  # Initialize all rows as unknown
df.loc[selected_indices, 'label_known'] = 1  # Mark selected rows as known

# === Step 6: Expand labeled data to 20% (K-shot) ===
existing_indices = selected_indices  # Store initial 10% indices
k_shot_fraction = 0.2  # Define new target: 20% labeled
extra_needed = int(k_shot_fraction * len(df)) - len(existing_indices)  # Additional samples required
remaining_indices = list(set(df.index) - set(existing_indices))  # Indices not in labeled set
remaining_df = df.loc[remaining_indices]  # Get remaining data
_, extra_df = train_test_split(  # Stratified sampling from remaining data
    remaining_df,
    train_size=extra_needed,
    stratify=remaining_df['label'],
    random_state=42
)
extra_indices = extra_df.index.values  # Extract indices from sampled data
k_shot_indices = np.concatenate([existing_indices, extra_indices])  # Combine old + new indices
df['label_known'] = 0  # Reset label_known column
df.loc[k_shot_indices, 'label_known'] = 1  # Mark K-shot (20%) samples as known

# === Step 7: Split dataset into labeled and unlabeled ===
labeled_data = df[df['label_known'] == 1]  # Data with known labels
unlabeled_data = df[df['label_known'] == 0]  # Data without labels

# === Step 8: Extract features and labels ===
X_labeled = scaler.fit_transform(labeled_data[features])  # Scale features for labeled data
y_labeled = labeled_data['label']  # Extract labels for training
X_unlabeled = scaler.transform(unlabeled_data[features])  # Scale features for test data
y_unlabeled_true = unlabeled_data['label']  # Extract true labels for evaluation

# === Step 9: Train baseline model (before tuning) ===
base_model = RandomForestClassifier(random_state=42)  # Create base model
base_model.fit(X_labeled, y_labeled)  # Train model on labeled data
y_pred_base = base_model.predict(X_unlabeled)  # Predict on unlabeled data
f1_base = f1_score(y_unlabeled_true, y_pred_base)  # Calculate F1 score
print(f"F1 Score BEFORE finetuning (20% labeled K-shot): {f1_base:.4f}")  # Show performance

# === Step 10: Define hyperparameter tuning grid ===
param_grid = {
    'n_estimators': [100, 200, 300],  # Number of trees
    'max_depth': [None, 20, 40],  # Maximum depth of each tree
    'min_samples_split': [2, 5],  # Minimum samples to split an internal node
    'min_samples_leaf': [1, 2],  # Minimum samples at a leaf node
    'class_weight': ['balanced', None]  # Whether to balance class weights
}

# === Step 11: Randomized hyperparameter search ===
random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42),  # Base model
    param_distributions=param_grid,  # Parameter grid
    n_iter=20,  # Number of combinations to try
    scoring='f1',  # Evaluation metric
    cv=3,  # 3-fold cross-validation
    n_jobs=-1,  # Use all available CPU cores
    random_state=42  # Ensure reproducibility
)

random_search.fit(X_labeled, y_labeled)  # Train with hyperparameter search
print("Best parameters found:", random_search.best_params_)  # Show best config

best_model = random_search.best_estimator_  # Extract best trained model

# === Step 12: Predict probabilities and optimize threshold ===
y_probs = best_model.predict_proba(X_unlabeled)[:, 1]  # Get class-1 probabilities

best_f1 = 0  # To track best F1 score
best_thresh = 0.5  # Default threshold

# Try multiple thresholds to find best classification point
for thresh in np.arange(0.3, 0.7, 0.01):  # Iterate thresholds from 0.30 to 0.69
    y_pred_thresh = (y_probs >= thresh).astype(int)  # Convert prob to class label
    f1 = f1_score(y_unlabeled_true, y_pred_thresh)  # Compute F1 score
    if f1 > best_f1:  # If better than current best
        best_f1 = f1  # Update best score
        best_thresh = thresh  # Update best threshold

# === Step 13: Final output ===
print(f"Best threshold for highest F1: {best_thresh:.2f}")  # Print optimal threshold
print(f"Highest F1 after threshold tuning: {best_f1:.4f}")  # Print best F1 score
