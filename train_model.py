import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# Load the dataset
data_path = 'online_shopping_pref.csv'  # Update with the correct path to your dataset
data = pd.read_csv(data_path)

# Rename columns for easier access (optional)
data.columns = ['Timestamp', 'Shopping_Frequency', 'Age_Group', 'Electronics_Platform',
                'Fashion_Platform', 'Beauty_Platform', 'Grocery_Platform', 
                'Important_Factor', 'Trust_Reviews', 'Best_Return_Policy']

# Drop the 'Timestamp' column as it's not relevant for training
data = data.drop(columns=['Timestamp'])

# Drop rows with missing values (if any)
data = data.dropna()

# Define the feature columns (excluding the target column)
X = data.drop(columns=['Important_Factor'])

# Encode categorical variables using one-hot encoding
X = pd.get_dummies(X, drop_first=True)

# Define the target variable (what we want to predict)
y = data['Important_Factor']

# Encode the target variable (converting text labels into numbers)
y = pd.factorize(y)[0]

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest Classifier
rf_model = RandomForestClassifier(random_state=42)

# Train the Random Forest model
rf_model.fit(X_train, y_train)

# Make predictions on the test set
rf_y_pred = rf_model.predict(X_test)

# Initialize the Logistic Regression model
lr_model = LogisticRegression(max_iter=1000, random_state=42)

# Train the Logistic Regression model
lr_model.fit(X_train, y_train)

# Make predictions on the test set
lr_y_pred = lr_model.predict(X_test)

# Evaluate both models
print("Random Forest Results:")
print(confusion_matrix(y_test, rf_y_pred))
print(classification_report(y_test, rf_y_pred))
print(f"Random Forest Accuracy: {accuracy_score(y_test, rf_y_pred):.4f}\n")

print("Logistic Regression Results:")
print(confusion_matrix(y_test, lr_y_pred))
print(classification_report(y_test, lr_y_pred))
print(f"Logistic Regression Accuracy: {accuracy_score(y_test, lr_y_pred):.4f}\n")

# Save the trained models to files (optional)
joblib.dump(rf_model, 'random_forest_model.pkl')
joblib.dump(lr_model, 'logistic_regression_model.pkl')
