import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Simulated Dataset Creation ---
# We'll simulate a simple dataset where each "food item" is represented by two features:
# Feature 1: A value related to color (e.g., a normalized red channel intensity).
# Feature 2: A value related to shape or texture (e.g., roundness, smoothness).
# We also have a dictionary for calorie information for our labels.

def create_food_dataset(num_samples_per_class=100):
    """
    Generates a simulated dataset for a few food items.
    
    Args:
        num_samples_per_class (int): The number of samples to generate for each food item.
    
    Returns:
        tuple: A tuple containing the feature matrix (X) and label vector (y).
    """
    # Define our food items and their conceptual features
    food_info = {
        'Apple': {'features': [1.0, 5.0], 'calories': 95},   # Red color, round shape
        'Banana': {'features': [4.0, 3.0], 'calories': 105}, # Yellow color, oblong shape
        'Carrot': {'features': [2.0, 8.0], 'calories': 25},  # Orange color, long shape
        'Broccoli': {'features': [0.5, 0.5], 'calories': 55} # Green color, bumpy texture
    }

    features = []
    labels = []
    
    # Generate noisy data around the conceptual features for each food item
    for food_name, info in food_info.items():
        base_features = np.array(info['features'])
        
        # Add some random noise to simulate real-world data variation
        samples = base_features + np.random.randn(num_samples_per_class, 2) * 0.7
        features.extend(samples)
        labels.extend([food_name] * num_samples_per_class)
        
    X = np.array(features)
    y = np.array(labels)
    
    return X, y, food_info

# Generate the simulated data and food info
X, y, food_info = create_food_dataset(num_samples_per_class=150)

# --- Visualize the Dataset ---
# This plot helps to see how our different food "clusters" are separated.
plt.figure(figsize=(10, 8))
sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, style=y, palette='Set2', s=100)
plt.title('Simulated Food Recognition Data')
plt.xlabel('Feature 1 (Conceptual Color)')
plt.ylabel('Feature 2 (Conceptual Shape/Texture)')
plt.legend(title='Food Item')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# --- 2. Model Training ---
# Split the dataset for training and testing.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)
print("-" * 40)

# We'll use a K-Nearest Neighbors (K-NN) classifier. This model is intuitive:
# it classifies a new point based on the majority class of its nearest neighbors.
model = KNeighborsClassifier(n_neighbors=5)

print("Starting model training...")
model.fit(X_train, y_train)
print("Model training complete.")
print("-" * 40)

# --- 3. Prediction and Evaluation ---
# Make predictions on the test set to evaluate performance.
print("Making predictions on the test data...")
y_pred = model.predict(X_test)

# Display the classification report and confusion matrix to see how well the model performed.
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred, labels=np.unique(y_test))
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
plt.title('Confusion Matrix for Food Recognition')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# --- 4. Calorie Estimation (Using the Trained Model) ---
# Now, let's simulate a real-world use case: taking a new "image" (feature vector)
# and getting its calorie count.
def estimate_calories(feature_vector):
    """
    Classifies a food item from its features and returns the estimated calories.
    
    Args:
        feature_vector (np.array): A 2-element NumPy array representing the food's features.
    
    Returns:
        tuple: The predicted food name and its calorie count.
    """
    # The model predicts the food name from the features
    predicted_food_name = model.predict([feature_vector])[0]
    
    # We look up the calorie information from our predefined dictionary
    calories = food_info.get(predicted_food_name, {}).get('calories', 'N/A')
    
    return predicted_food_name, calories

# Example usage:
print("\n--- Example Calorie Estimation ---")
new_food_item_features_1 = np.array([0.9, 5.1]) # Features similar to Apple
new_food_item_features_2 = np.array([4.1, 3.2]) # Features similar to Banana

predicted_food_1, calories_1 = estimate_calories(new_food_item_features_1)
print(f"Features: {new_food_item_features_1} -> Predicted Food: {predicted_food_1}, Estimated Calories: {calories_1}")

predicted_food_2, calories_2 = estimate_calories(new_food_item_features_2)
print(f"Features: {new_food_item_features_2} -> Predicted Food: {predicted_food_2}, Estimated Calories: {calories_2}")
