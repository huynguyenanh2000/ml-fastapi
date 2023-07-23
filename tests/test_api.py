"""This file contains the tests for the API."""
from fastapi.testclient import TestClient
import sys
import os

# Adjust the sys.path to include the path to the 'ml-fastapi' directory
current_dir = os.path.dirname(os.path.abspath(__file__))
ml_fastapi_dir = os.path.join(current_dir, "..")
sys.path.insert(0, ml_fastapi_dir)

# Import the app from the main.py file
from main import app

# Instantiate the test client
client = TestClient(app)

def test_get_root():
    """Test the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello World"}

def test_post_predict1():
    """Test the predict endpoint with label >50k."""
    response = client.post("/predict", json={
        "age": 52,
        "workclass": "Private",
        "fnlgt": 129177,
        "education": "Bachelor's",
        "education_num": 13,
        "marital_status": "Widowed",
        "occupation": "Other-service",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Female",
        "capital_gain": 0,
        "capital_loss": 2824,
        "hours_per_week": 20,
        "native_country": "United-States",
    })
    assert response.status_code == 200
    assert response.json() == {"prediction": ">50K"}

def test_post_predict2():
    """Test the predict endpoint with label <=50k."""
    response = client.post("/predict", json={
        "age": 28,
        "workclass": "Private",
        "fnlgt": 183175,
        "education": "Some-college",
        "education_num": 10,
        "marital_status": "Divorced",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Female",
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "United-States",
    })
    assert response.status_code == 200
    assert response.json() == {"prediction": "<=50K"}
