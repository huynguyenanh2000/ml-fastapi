"""
This is the Python Test for the train_model.py file

This file will be used to test
    1. train_model
    2. compute_model_metrics
    3. inference

Author: Nguyen Huy Anh
Date: 16th July 2023
"""

import os
import logging
import sys
import pandas as pd
import pytest

# Adjust the sys.path to include the path to the 'ml-fastapi' directory
current_dir = os.path.dirname(os.path.abspath(__file__))
ml_fastapi_dir = os.path.join(current_dir, "..")
sys.path.insert(0, ml_fastapi_dir)

import starter.ml.model as md
import starter.ml.data as dt
from sklearn.model_selection import train_test_split

logging.basicConfig(filename='./tests/test.log', 
                    level=logging.INFO,filemode='w', 
                    format='%(name)s - %(levelname)s - %(message)s')

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


@pytest.fixture(name='df_for_testing')
def data():
    """
    Fixture will be used by the unit tests.
    """
    yield pd.read_csv("./data/census_cleaned.csv")

def test_load_data(df_for_testing):
    """
    This function will test the load_data function in the train_model.py file
    """
    
    try:
        assert df_for_testing.shape[0] > 0
        assert df_for_testing.shape[1] > 0
    except AssertionError as err:
        logging.error("Testing loading data: The dataframe is empty")
        raise err
    
def test_train_test_split(df_for_testing):
    """
    This function will test the train_test_split function in the train_model.py file
    """
    try:
        train, test = train_test_split(df_for_testing, test_size=0.20)
        logging.info("Data split successfully")
    except AssertionError as err:
        assert len(train) > 0
        assert len(test) > 0
        assert len(train) + len(test) == len(df_for_testing)
        assert len(train.columns) == len(test.columns)
        logging.error("Testing train_test_split: The dataframe is empty")
        raise err

def test_process_data(df_for_testing):
    """
    This function will test the process_data function in the train_model.py file
    """
    try:
        train, test = train_test_split(df_for_testing, test_size=0.20)
        X_train, y_train, encoder, lb = dt.process_data(
            train, categorical_features=cat_features, label="income", training=True
        )
        logging.info("Data processed successfully")
    except AssertionError as err:
        assert len(X_train) > 0
        assert len(y_train) > 0
        assert len(X_train) == len(y_train)
        logging.error("Testing test_process_data: The dataframe is empty")
        raise err  
    
    try:
        assert os.path.exists('./model/encoder.pkl')
        assert os.path.exists('./model/lb.pkl')
    except AssertionError as err:
        logging.error("Testing test_process_data: The encoder and lb files are not saved")
        raise err
    
    try: 
        X_test, y_test, encoder, lb = dt.process_data(test, categorical_features=cat_features, label="income", training=False, encoder=encoder, lb=lb)
        logging.info("Data processed successfully")
    except AssertionError as err:
        assert len(X_test) > 0
        assert len(y_test) > 0
        assert len(X_test) == len(y_test)
        logging.error("Testing test_process_data: The dataframe is empty")
        raise err
    
def test_train_model(df_for_testing):
    """
    This function will test the train_model function in the train_model.py file
    """
    try:
        train, test = train_test_split(df_for_testing, test_size=0.20)
        X_train, y_train, encoder, lb = dt.process_data(
            train, categorical_features=cat_features, label="income", training=True
        )
        model = md.train_model(X_train, y_train)
        logging.info("Model trained successfully")
    except AssertionError as err:
        assert os.path.exists('./model/logistic-regression.pkl')
        logging.error("Testing test_train_model: The model is not trained")
        raise err
    



