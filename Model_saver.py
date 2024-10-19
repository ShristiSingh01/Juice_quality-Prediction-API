import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load the dataset
data = pd.read_csv('juice.csv')
data = data.dropna(how='all')
# Drop columns if they exist
columns_to_drop = ['Sample ID', 'Unnamed: 22']
data_cleaned = data.drop(columns=[col for col in columns_to_drop if col in data.columns])
print(data_cleaned.shape)
# Separate features and target
X = data_cleaned.drop(columns=['Quality'])
y = data_cleaned['Quality']

# Encode categorical variables
X_encoded = pd.get_dummies(X)

# Fill missing values for numeric columns if any
numerical_cols = X_encoded.select_dtypes(include=['float64', 'int64']).columns
X_encoded[numerical_cols] = X_encoded[numerical_cols].fillna(X_encoded[numerical_cols].mean())

# Fill missing values in target variable
y = y.fillna(y.mode()[0])

# Train the model
classifier = RandomForestClassifier(random_state=42)
classifier.fit(X_encoded, y)

# Save the trained model
with open('trained_classifier.pkl', 'wb') as file:
    pickle.dump(classifier, file)
    
# Save the columns used in the training set
with open('model_columns.pkl', 'wb') as file:
    pickle.dump(X_encoded.columns.tolist(), file)
