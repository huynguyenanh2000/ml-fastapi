# Script to train machine learning model.

from sklearn.model_selection import train_test_split


# Add the necessary imports for the starter code.
import pandas as pd
from ml.data import process_data
from ml.model import train_model, compute_model_metrics
import joblib


# Add code to load in the data.
data = pd.read_csv("./data/census_cleaned.csv")

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native_country",
]
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="income", training=True
)

# Save the label encoder.
joblib.dump(encoder, './model/encoder.pkl')
joblib.dump(lb, './model/lb.pkl')

# Proces the test data with the process_data function.
X_test, y_test, encoder, lb = process_data(test, categorical_features=cat_features, label="income", training=False, encoder=encoder, lb=lb)

# Train a model.
lr = train_model(X_train, y_train)

# Predict on the test data.
preds = lr.predict(X_test)

# Compute evaluation metrics.
precision, recall, fbeta = compute_model_metrics(y_test, preds)
print(f"precision: {precision}")
print(f"recall: {recall}")
print(f"fbeta: {fbeta}")

# Save the model.
joblib.dump(lr, './model/logistic-regression.sav')



