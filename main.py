import pandas as pd
import pickle
from flask import Flask, jsonify, request
from flask_cors import CORS
import json

app = Flask(__name__)
CORS(app)
filename = "randomforest.pkl"
model = pickle.load(open(filename, 'rb'))


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            response  = request.data
            jsonResponse = json.loads(response.decode('utf-8'))
            processedData = pd.DataFrame.from_dict(jsonResponse, orient='index').transpose()
            result = round(model.predict(processedData)[0])
            return jsonify({'data': result})
        except:
            return jsonify({'error': "Error while predicting"}) 


@app.route('/health', methods=['GET'])
def ping():
    return jsonify({'heath': "healthy"})
