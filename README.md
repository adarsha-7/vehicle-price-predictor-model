# Used Vehicle Price Prediction API

### This project is a microservice API created using FastAPI for serving a model that predicts the price of a used car given its attributes.

## Features:

#### - Users can input different attributes of a used car, and the model can predict an appropriate price for it, based on more than 300,000 listings.

## Tools/Libraries/Frameworks Used

### Developmnet Environment:

#### - Jupyter Notebook

### Data Analysis (EDA):

#### - Numpy

#### - Pandas

#### - Matplotlib

### Pipelines/Model training:

#### - Scikit-learn

### API/Microservice:

#### - FastAPI

## How to test the model

#### Send a POST request at https://vehicle-price-predictor-model-render.onrender.com/predict using Postman or similar tool
#### Example of body(JSON):

{   
    "year": 2021,
    "odometer": 100000,
    "cylinders": "4 cylinders",
    "manufacturer": "ford",
    "condition": "excellent",
    "fuel": "gas",
    "transmission": "automatic",
    "drive": "rwd",
    "type": "van"
}
