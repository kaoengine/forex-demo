import os
from flask import Flask
from flask import jsonify


app = Flask(__name__)

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
              'data': [86, 114,],
              'label': "Africa",
              'borderColors': ["#3e95cd", "#8e5ea2", ],
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