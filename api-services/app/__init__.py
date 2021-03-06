# Python Import:
import os
import joblib
import numpy as np
from flask import Flask
from flask import jsonify, make_response
import torch
from machineLearnPackage import forex, checkpoint
import pathlib

# Define Flask app:
app = Flask(__name__)
# Define and load pre-trained models:
fx_encoder = forex.Encoder(source_size=4, hidden_size=256, num_layers=1, dropout=0.0, bidirectional=False)
fx_decoder = forex.Decoder(target_size=4, hidden_size=256, num_layers=1, dropout=0.0)

encoder_path = pathlib.Path('./machineLearnPackage/checkpoint/fx_encoder.pth')
decoder_path = pathlib.Path('./machineLearnPackage/checkpoint/fx_decoder.pth')
scaler_path = pathlib.Path('./machineLearnPackage/checkpoint/scaler.save')

fx_encoder.load_state_dict(torch.load(encoder_path, map_location="cpu"))
fx_decoder.load_state_dict(torch.load(decoder_path, map_location="cpu"))
scaler = joblib.load(scaler_path)
# Toggle evaluation mode:
fx_encoder.eval()
fx_decoder.eval()


def predict(x):
    # Convert input to numpy array:
    x = np.array([x["HIGH"], x["LOW"], x["OPEN"], x["CLOSE"]]).T
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
        "HIGH": [float(i) for i in list(o[:, 0])],
        "LOW": [float(i) for i in list(o[:, 1])],
        "OPEN": [float(i) for i in list(o[:, 2])],
        "CLOSE": [float(i) for i in list(o[:, 3])]
    }
    return o


def castDataToFe(data,dataWithDateTime):
    dataModel = {
        'labels': dataWithDateTime['DATE_TIME'],
        'datasets': [
        {
            'data': data['HIGH'],
            
            'label': "HIGH",
            'borderColor': "#3e95cd",
            'fill': 'false',
        },
        {
            'data': data['LOW'],
            
            'label': "LOW",
            'borderColor': "#8e5ea2",
            'fill': 'false',
        },
        {
            'data': data['OPEN'],
            
            'label': "OPEN",
            'borderColor': "#3cba9f",
            'fill': 'false',
        },
        {
            'data': data['CLOSE'],
            
            'label': "CLOSE",
            'borderColor': "#e8c3b9",
            'fill': 'false',
        },
    ]}
    print('dataModel',dataModel)
    return dataModel


@app.route("/")
def main():
    return "Welcome!"


@app.route('/how are you')
def hello():
    return 'I am good, how about you?'


@app.route('/forex')
def forex():
    data = {
            "DATE_TIME": ["2000.01.03 00:00:00", "2000.01.03 00:15:00", "2000.01.03 00:30:00", "2000.01.03 00:45:00", "2000.01.03 01:00:00", "2000.01.03 01:15:00", "2000.01.03 01:30:00", "2000.01.03 01:45:00", "2000.01.03 02:00:00", "2000.01.03 02:15:00", "2000.01.03 02:30:00", "2000.01.03 02:45:00", ],
            "HIGH": [1.008, 1.0087, 1.0089, 1.0132, 1.0133, 1.0125, 1.0137, 1.0141, 1.0145, 1.0142, 1.0147, 1.0173, ],
            "LOW": [1.0073, 1.0076, 1.0079, 1.0078, 1.012, 1.012, 1.0129, 1.0133, 1.0134, 1.0135, 1.0137, 1.0142, ],
            "OPEN": [1.0073, 1.0078, 1.0087, 1.0078, 1.0129, 1.0123, 1.0132, 1.0135, 1.014, 1.0135, 1.0142, 1.0142, ],
            "CLOSE": [1.0077, 1.0086, 1.0079, 1.0128, 1.0122, 1.0124, 1.0133, 1.0137, 1.0138, 1.0141, 1.0145, 1.0171, ]}

    # Query data from DB return back the set of Step with &step is argument
    # stepData = Db.query();

    # Invoke the prediction data with argument from step Data
    afterPredictData = predict(data)
    print(afterPredictData)

    # Note: for Khanh make repsonse is a wrapper
    resp = make_response(jsonify(castDataToFe(afterPredictData,data)), 200)
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp


if __name__ == "__main__":
    # Run Flask app:
    app.run(host="0.0.0.0", port=8080)
