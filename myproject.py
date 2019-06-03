from flask import Flask, request, jsonify, Response
import pandas as pd
import json
import sys
import os
import shutil
import time
import traceback
from flask_cors import CORS, cross_origin

app = Flask(__name__)
cros = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route("/")
def hello():
    return "<h1 style='color:red'>Hello There!</h1>"

@app.route('/predict', methods=['POST'])
def predict():
        try:
            json_ = request.json
            datavk = pd.DataFrame(json_)

            answer123 = datavk[0][0] % 3
            return jsonify(answer123)            

        except Exception as e:
            return jsonify({'error': str(e), 'trace': traceback.format_exc()})

if __name__ == "__main__":
    app.run(host='0.0.0.0')
