import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the dataset
data = pd.read_csv('juice.csv')

data = data.dropna(how='all')
# Drop columns if they exist
columns_to_drop = ['Sample ID', 'Unnamed: 22']
data_cleaned = data.drop(columns=[col for col in columns_to_drop if col in data.columns])

# Prepare the dataset by separating features (X) and target ('Quality')
X = data_cleaned.drop(columns=['Quality'])
y = data_cleaned['Quality']

# Convert categorical variables into dummy/indicator variables for machine learning
X_encoded = pd.get_dummies(X)

# Handle missing values
# Fill missing values in numerical columns with their mean
numerical_cols = X_encoded.select_dtypes(include=['float64', 'int64']).columns
X_encoded[numerical_cols] = X_encoded[numerical_cols].fillna(X_encoded[numerical_cols].mean())

# Fill missing values in the target 'Quality' column with the most frequent value
y = y.fillna(y.mode()[0])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Train a RandomForestClassifier
classifier = RandomForestClassifier(random_state=42)
classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = classifier.predict(X_test)

# Evaluate the performance of the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Create a DataFrame to display test input with actual and predicted quality
test_results = X_test.copy()
test_results['Actual Quality'] = y_test.values
test_results['Predicted Quality'] = y_pred



# Display results
print(f"Accuracy: {accuracy}")
print(f"Confusion Matrix:\n{conf_matrix}")
print(f"Classification Report:\n{class_report}")
