# Python Import:
import os
import joblib
import numpy as np
from flask import Flask
from flask import jsonify
# Pytorch Import:
import torch
from forex-demo.machine-learn import forex

# Define Flask app:
app = Flask(__name__)
# Define and load pre-trained models:
fx_encoder = forex.Encoder(source_size=4, hidden_size=256, num_layers=1, dropout=0.0, bidirectional=False)
fx_decoder = forex.Decoder(target_size=4, hidden_size=256, num_layers=1, dropout=0.0)
prepath = "../../machine-learn/checkpoint/"
fx_encoder.load_state_dict(torch.load(prepath + "fx_encoder"), map_location="cpu")
fx_decoder.load_state_dict(torch.load(prepath + "fx_decoder"), map_location="cpu") 
scaler = joblib.load(prepath + "scaler.save")
# Toggle evaluation mode:
fx_encoder.eval()
fx_decoder.eval()

@app.route("/predict")
def predict(x):
    # Convert input to python object:
    
    # Scale input with scaler:
    x = scaler.transform(x)
    # Convert python object to pytorch tensor:
    x = torch.FloatTensor(x).unsqueeze(0)
    # Predict:
    with torch.no_grad():
        _, h = fx_encoder(x)
        o, _ = fx_decoder(h, 32, None, False)
    # Convert pytorch prediction back to python object:
    o = o.squeeze().numpy()
    o = scaler.inverse_transform(o)
    o = {
        "HIGH": list(o[:, 0]), 
        "LOW": list(o[:, 1]), 
        "OPEN": list(o[:, 2]),
        "CLOSE": list(o[:, 3])
    }
    return jsonify(o)

@app.route("/")
def main():
    return "Welcome!"

@app.route('/how are you')
def hello():
    return 'I am good, how about you?'

@app.route('/forex')
def forex():
    data = { 'datasets': [
            {
              'data': [86, 114, 106, 106, 107, 111, 133, 221, 783, 2478],
              'label': "Africa",
              'borderColor': "#3e95cd",
              'fill': 'false',
            },
            {
              'data': [282, 350, 411, 502, 635, 809, 947, 1402, 3700, 5267],
              'label': "Asia",
              'borderColor': "#8e5ea2",
              'fill': 'false',
            },
            {
              'data': [168, 170, 178, 190, 203, 276, 408, 547, 675, 734],
              'label': "Europe",
              'borderColor': "#3cba9f",
              'fill': 'false',
            },
            {
              'data': [40, 20, 10, 16, 24, 38, 74, 167, 508, 784],
              'label': "Latin America",
              'borderColor': "#e8c3b9",
              'fill': 'false',
            },
            {
              'data': [6, 3, 2, 2, 7, 26, 82, 172, 312, 433],
              'label': "North America",
              'borderColor': "#c45850",
              'fill': 'false',
            },
          ]}
    response = jsonify(data)
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)