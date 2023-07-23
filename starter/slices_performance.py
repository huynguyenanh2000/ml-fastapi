""" Write a function that outputs the performance of the model on slices of the  test data."""

from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import train_model, compute_model_metrics
import joblib
import pandas as pd

def slices_performance(data, categorical_features, label):
    """
    This function will output the performance of the model on slices of the test data.
    """

    # Load the encoder and lb files.
    encoder = joblib.load('./model/encoder.pkl')
    lb = joblib.load('./model/lb.pkl')
    

    
    # Load the model.
    model = joblib.load('./model/logistic-regression.sav')

    dict = {}
    
    # Predict on the slices of the test data.
    for feature in categorical_features:
        for category in data[feature].unique():

            # Slice the data.
            dat = data[data[str(feature)] == str(category)]

            # Process the test data with the process_data function.
            X_test, y_test, encoder, lb = process_data(dat, categorical_features=categorical_features, label=label, training=False, encoder=encoder, lb=lb)

            preds = model.predict(X_test)
            precision, recall, fbeta = compute_model_metrics(y_test, preds)
            dict[str(feature) + '_' + str(category)] = [precision, recall, fbeta]
    
    return dict


data = pd.read_csv("./data/census_cleaned.csv")
_, test = train_test_split(data, test_size=0.20)
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
res = slices_performance(test, cat_features, 'income')
print(res)

    



