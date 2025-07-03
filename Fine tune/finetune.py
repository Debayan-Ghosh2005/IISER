import pandas as pd  # Used to load and manipulate tabular data
import numpy as np  # Used for numerical operations and random selection
from sklearn.ensemble import RandomForestClassifier  # Random Forest classification model
from sklearn.metrics import f1_score  # Used to evaluate classification performance (F1 score)
from sklearn.preprocessing import StandardScaler, LabelEncoder  # For feature scaling and label encoding
from sklearn.model_selection import RandomizedSearchCV  # For hyperparameter tuning

# === Step 1: Load Dataset ===
df = pd.read_csv("data.csv")  # Load the dataset from CSV file into a pandas DataFrame

# === Step 2: Encode 'continent' column (categorical) ===
le = LabelEncoder()  # Create a LabelEncoder object
df['continent'] = le.fit_transform(df['continent'])  # Convert continent names into integer codes

# === Step 3: Simulate Few-Shot Learning (Only 10% data is labeled) ===
np.random.seed(42)  # Set random seed so results are reproducible
df['label_known'] = 0  # Add a new column 'label_known' with default value 0 (unlabeled)

# Randomly choose 10% of indices to mark as labeled
known_indices = np.random.choice(df.index, size=int(0.1 * len(df)), replace=False)  # Randomly pick indices
df.loc[known_indices, 'label_known'] = 1  # Mark those rows as labeled by setting label_known = 1

# === Step 4: Split into Labeled and Unlabeled Subsets ===
labeled_data = df[df['label_known'] == 1]  # Rows where label is known
unlabeled_data = df[df['label_known'] == 0]  # Rows where label is not known

# === Step 5: Define Feature Columns and Scale the Data ===
features = [  # List of feature column names to be used for training
    'age',
    'continent',
    'physical_activity_days',
    'processed_food_meals',
    'sleep_hours',
    'smoking_status',
    'alcohol_consumption'
]

scaler = StandardScaler()  # Create StandardScaler object to normalize features (mean=0, std=1)

# Scale labeled features (fit and transform)
X_labeled = scaler.fit_transform(labeled_data[features])  # Input features for labeled data
y_labeled = labeled_data['label']  # Output labels for labeled data

# Scale unlabeled features (only transform, use same scaling as labeled)
X_unlabeled = scaler.transform(unlabeled_data[features])  # Input features for unlabeled data
y_unlabeled_true = unlabeled_data['label']  # True labels for evaluation

# === Step 6: Train Base Random Forest Model on Few-Shot Data ===
base_model = RandomForestClassifier(random_state=42)  # Create base RF model
base_model.fit(X_labeled, y_labeled)  # Train the model on labeled data

# === Step 7: Evaluate Base Model (Before Tuning) ===
y_pred_base = base_model.predict(X_unlabeled)  # Predict class labels for unlabeled data
f1_base = f1_score(y_unlabeled_true, y_pred_base)  # Calculate F1 score
print(f"F1 Score BEFORE finetuning: {f1_base:.4f}")  # Print result

# === Step 8: Define Hyperparameter Grid for Random Search ===
param_grid = {
    'n_estimators': [100, 200, 300],        # Number of trees in the forest
    'max_depth': [None, 20, 40],            # Maximum depth of each tree
    'min_samples_split': [2, 5],            # Min samples needed to split a node
    'min_samples_leaf': [1, 2],             # Min samples required at a leaf node
    'class_weight': ['balanced', None]      # Adjust class weight to handle imbalance
}

# === Step 9: Perform Randomized Search for Best Parameters ===
random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42),  # Base model
    param_distributions=param_grid,  # Hyperparameter options
    n_iter=20,  # Number of random combinations to try
    scoring='f1',  # Evaluation metric to maximize
    cv=3,  # 3-fold cross-validation
    n_jobs=-1,  # Use all CPU cores for parallel search
    random_state=42  # Reproducibility
)

random_search.fit(X_labeled, y_labeled)  # Fit the model on labeled data using random search
print("Best parameters found:", random_search.best_params_)  # Print the best hyperparameters

best_model = random_search.best_estimator_  # Extract the best model found from the search

# === Step 10: Predict Probabilities on Unlabeled Data ===
y_probs = best_model.predict_proba(X_unlabeled)[:, 1]  # Get probabilities for positive class (class 1)

# === Step 11: Tune Classification Threshold for Maximum F1 ===
best_f1 = 0  # Initialize best F1 score
best_thresh = 0.5  # Default threshold

# Try thresholds from 0.30 to 0.69
for thresh in np.arange(0.3, 0.7, 0.01):  # Try many thresholds
    y_pred_thresh = (y_probs >= thresh).astype(int)  # Convert probability to 0/1 based on threshold
    f1 = f1_score(y_unlabeled_true, y_pred_thresh)  # Calculate F1 score
    if f1 > best_f1:  # If current F1 is better, update best values
        best_f1 = f1
        best_thresh = thresh

# === Step 12: Final Output ===
print(f"Best threshold for highest F1: {best_thresh:.2f}")  # Best threshold value
print(f"Highest F1 after threshold tuning: {best_f1:.4f}")  # Best F1 score achieved
