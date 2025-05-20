import pandas as pd
import numpy as np
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import os

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Load training data
print("Loading training data...")
training_data = pd.read_csv('data/Training.csv')

# Prepare features and target
X = training_data.iloc[:, :-1]  # All columns except the last one
y = training_data.iloc[:, -1]   # Last column (prognosis)

# Split data for validation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train KNN model
print("Training KNN model...")
knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(X_train, y_train)

# Train Decision Tree model
print("Training Decision Tree model...")
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train, y_train)

# Evaluate models
knn_accuracy = knn_classifier.score(X_test, y_test)
dt_accuracy = dt_classifier.score(X_test, y_test)

print(f"KNN Accuracy: {knn_accuracy:.4f}")
print(f"Decision Tree Accuracy: {dt_accuracy:.4f}")

# Save models to disk
print("Saving models...")
with open('models/disease_knn.pkl', 'wb') as f:
    pickle.dump(knn_classifier, f)

with open('models/disease_dt.pkl', 'wb') as f:
    pickle.dump(dt_classifier, f)

print("Models trained and saved successfully!")