import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

# === 1. Load dataset ===
df = pd.read_csv("data.csv")  # Load the dataset into a DataFrame

# === 2. Encode categorical ===
le = LabelEncoder()  # Initialize label encoder
df['continent'] = le.fit_transform(df['continent'])  # Encode 'continent' column to numeric values

# === 3. Few-Shot Labeling (10%) ===
np.random.seed(42)  # Set seed for reproducibility
df['label_known'] = 0  # Initialize a new column for marking labeled data
known_indices = np.random.choice(df.index, size=int(0.1 * len(df)), replace=False)  # Randomly select 10% of rows
df.loc[known_indices, 'label_known'] = 1  # Mark selected rows as labeled

# Separate labeled and unlabeled data
labeled_data = df[df['label_known'] == 1]
unlabeled_data = df[df['label_known'] == 0]

# === 4. Features Selection ===
# Define the input features used for training
features = ['age', 'continent', 'physical_activity_days', 'processed_food_meals',
            'sleep_hours', 'smoking_status', 'alcohol_consumption']

# === 5. Scaling ===
scaler = StandardScaler()  # Initialize scaler
X_labeled = scaler.fit_transform(labeled_data[features])  # Fit scaler on labeled data and transform it
y_labeled = labeled_data['label']  # Extract labels for labeled data

X_unlabeled = scaler.transform(unlabeled_data[features])  # Transform unlabeled data using same scaler
y_unlabeled_true = unlabeled_data['label']  # Extract true labels for evaluation

# === 6. Original RF ===
original_rf = RandomForestClassifier(random_state=42)  # Initialize Random Forest model
original_rf.fit(X_labeled, y_labeled)  # Train model on labeled data
y_pred_original = original_rf.predict(X_unlabeled)  # Predict labels on unlabeled data
f1_original = f1_score(y_unlabeled_true, y_pred_original)  # Calculate F1 Score
print(f"\nOriginal Model F1 Score: {f1_original:.4f}")

# === 7. Tuned RF ===
# Define hyperparameter grid
param_grid = {
    'n_estimators': [100, 200],           # Number of trees
    'max_depth': [None, 10, 20],          # Max depth of tree
    'min_samples_split': [2, 5],          # Minimum samples to split an internal node
    'min_samples_leaf': [1, 2]            # Minimum samples required at leaf node
}

rf = RandomForestClassifier(random_state=42)  # Base model
grid_search = GridSearchCV(
    estimator=rf,                          # Model to tune
    param_grid=param_grid,                # Parameter grid
    scoring='f1',                          # Optimize for F1 score
    cv=3,                                  # 3-fold cross validation
    n_jobs=-1,                             # Use all CPU cores
    verbose=2                              # Print progress
)

print("\nStarting Grid Search...")
grid_search.fit(X_labeled, y_labeled)  # Run grid search on labeled data
print(f"\nBest Parameters: {grid_search.best_params_}")  # Print best parameters

best_rf = grid_search.best_estimator_  # Get best model from grid search
y_pred_tuned = best_rf.predict(X_unlabeled)  # Predict on unlabeled data using best model
f1_tuned = f1_score(y_unlabeled_true, y_pred_tuned)  # Calculate F1 Score
print(f"Tuned Model F1 Score: {f1_tuned:.4f}")

# === 8. Save Predictions to df ===
# Save all predictions to the main DataFrame
df.loc[unlabeled_data.index, 'pred_original'] = y_pred_original
df.loc[unlabeled_data.index, 'pred_tuned'] = y_pred_tuned

# === 9. Confusion Matrices ===
fig, axes = plt.subplots(1, 2, figsize=(12, 5))  # Create subplots (1 row, 2 plots)

# Confusion Matrix for original model
cm_original = confusion_matrix(y_unlabeled_true, y_pred_original)
ConfusionMatrixDisplay(cm_original).plot(ax=axes[0])
axes[0].set_title(f"Original RF\nF1: {f1_original:.4f}")

# Confusion Matrix for tuned model
cm_tuned = confusion_matrix(y_unlabeled_true, y_pred_tuned)
ConfusionMatrixDisplay(cm_tuned).plot(ax=axes[1])
axes[1].set_title(f"Tuned RF\nF1: {f1_tuned:.4f}")

plt.tight_layout()  # Adjust layout to avoid overlap
plt.show()  # Display the plots

# === 10. Save CSV ===
df.to_csv("comparison_predictions.csv", index=False)  # Save results to CSV file
print("\nPredictions saved to 'comparison_predictions.csv'")
