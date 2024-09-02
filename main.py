from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from tensorflow.keras.models import load_model

# Initialize FastAPI app
app = FastAPI()

# Load the model once when the API starts
model = load_model('kirtan_model_last.h5')

# Define the input data structure using Pydantic
class InputData(BaseModel):
    features: list  # A list of lists, each representing a set of features for prediction

# Define the prediction route

@app.get('/')
def show():
    print('Hello World')

@app.post("/predict")
async def predict(input_data: InputData):
    # Convert input data to numpy array
    data = np.array(input_data.features)
    
    # Make a prediction using the loaded model
    prediction = model.predict(data)
    
    # Convert the prediction result to a Python list
    prediction_list = prediction.tolist()
    
    # Return the prediction as a JSON response
    return {"prediction": prediction_list}

# Run the API with: uvicorn main:app --reload

