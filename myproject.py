from flask import Flask, request, jsonify, Response
import pandas as pd
import pickle
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


onehot_encoder_filename = 'one_hot_encoder_urfu.pkl'
onehot_encoder = pickle.load(open(onehot_encoder_filename, 'rb'))

# загрузка pickle-файлов
clf_filename = 'classifier_LR.pkl'
clf = pickle.load(open(clf_filename, 'rb'))

pca_filename = 'PCA.pkl'
pca = pickle.load(open(pca_filename, 'rb'))

#предсказание и получение данных из ВК
def user_predict(datavk):
    me=pd.DataFrame()
    for s_id in datavk[1].tolist():
            try:
                user_id = datavk[0][0]
                name = None
                group_id=s_id
                me = me.append({'user_id': user_id, 'name': name, 'group_id':group_id}, ignore_index=True)
            except:
                pass
    me = me.astype({"user_id": int, "group_id": int})
    me_data = onehot_encoder.transform(me.group_id.values.reshape(-1,1))
    meOneHot = pd.DataFrame(me_data, columns = ["group"+str(int(i)) for i in range(me_data.shape[1])])
    me = pd.concat([me, meOneHot], axis=1)
    del me['group_id']
    del me['name']
    test_api_data = me.groupby('user_id').max()
    test_api_data_reduced = pca.transform(test_api_data)
    me_pred = clf.predict(test_api_data_reduced)
    return me_pred

@app.route("/")
def hello():
    return "<h1 style='color:red'>server ok!</h1>"

#веб-часть
@app.route("/predict", methods=['POST'])
def predict():
        try:
            json_ = request.json
            datavk = pd.DataFrame(json_)
            '''
            result = user_predict(datavk)[0]
            if result=='INFO':
                output=0
            elif result=='IGUP':
                output=1
            elif result=='GSEM':
                output=2
            return jsonify(output)            
            '''
            return jsonify(datavk) 
        except Exception as e:
            return jsonify({'error': str(e), 'trace': traceback.format_exc()})

if __name__ == "__main__":
    app.run(host='0.0.0.0')


