from flask import Flask, request, jsonify, Response
import pandas as pd
import json
import sys
import os
import shutil
import time
import traceback

app = Flask(__name__)

@app.route("/")
def hello():
    return "<h1 style='color:red'>Hello There!</h1>"

@app.route('/predict', methods=['POST'])
def predict():
        try:
            json_ = request.json
            datavk = pd.DataFrame(json_)

            answer123 = datavk[0][0] % 3
            json_encode = json.JSONEncoder().encode
            resp = Response(json_encode(answer123),
                    mimetype=("application/json"))
            resp.headers['Access-Control-Allow-Origin'] = '*'
            resp.headers['Access-Control-Allow-Headers'] = "Origin, X-Requested-With, Content-Type, Accept"
            return resp

        except Exception as e:
            return jsonify({'error': str(e), 'trace': traceback.format_exc()})

if __name__ == "__main__":
    app.run(host='0.0.0.0')
