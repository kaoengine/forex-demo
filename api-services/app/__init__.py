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
        "HIGH": list(o[:, 0]),
        "LOW": list(o[:, 1]),
        "OPEN": list(o[:, 2]),
        "CLOSE": list(o[:, 3])
    }
    return o


@app.route("/")
def main():
    return "Welcome!"


@app.route('/how are you')
def hello():
    return 'I am good, how about you?'


@app.route('/forex')
def forex():
    dataModel = {'datasets': [
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

    data = {"DATE_TIME":["2000.01.03 00:00:00","2000.01.03 00:15:00","2000.01.03 00:30:00","2000.01.03 00:45:00","2000.01.03 01:00:00","2000.01.03 01:15:00","2000.01.03 01:30:00","2000.01.03 01:45:00","2000.01.03 02:00:00","2000.01.03 02:15:00","2000.01.03 02:30:00","2000.01.03 02:45:00",],
            "HIGH":[1.008,1.0087,1.0089,1.0132,1.0133,1.0125,1.0137,1.0141,1.0145,1.0142,1.0147,1.0173,],
            "LOW":[1.0073,1.0076,1.0079,1.0078,1.012,1.012,1.0129,1.0133,1.0134,1.0135,1.0137,1.0142,],
            "OPEN":[1.0073,1.0078,1.0087,1.0078,1.0129,1.0123,1.0132,1.0135,1.014,1.0135,1.0142,1.0142,],
            "CLOSE":[1.0077,1.0086,1.0079,1.0128,1.0122,1.0124,1.0133,1.0137,1.0138,1.0141,1.0145,1.0171,]}

    # Query data from DB return back the set of Step with &step is argument
    # stepData = Db.query();

    # Invoke the prediction data with argument from step Data
    afterPredictData = predict(data)

    # Note: for Khanh make repsonse is a wrapper
    resp = make_response(jsonify(data), 200)
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp


if __name__ == "__main__":
    # Run Flask app:
    app.run(host="0.0.0.0", port=8080)


