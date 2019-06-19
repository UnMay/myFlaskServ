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
import vk
#test

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

#авторизация в ВК
session = vk.AuthSession(app_id='6729196', user_login='jeka87@e1.ru', user_password='P@ssword1987')
api = vk.API(session)

#предсказание и получение данных из ВК
def user_predict(user_ids):
    me=pd.DataFrame()
    user_info = api.users.get(user_ids=user_ids, extended='1', v='5.87')
    user_id = user_info[0]['id']
    subscriptions = api.users.getSubscriptions(user_id=user_id, extended='1', v='5.87')
    subscriptions = subscriptions['items']
    for subscription in subscriptions:
            try:
                user_id = user_id
                name = subscription['name']
                group_id=subscription['id']
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

#веб-часть
@app.route('/')
def predict():
        try:
            user_id = request.args.get('user_ids')
            result = user_predict(user_id)[0]
            if result=='INFO':
                output=0
            elif result=='IGUP':
                output=1
            elif result=='GSEM':
                output=2
            return jsonify(output)            

        except Exception as e:
            return jsonify({'error': str(e), 'trace': traceback.format_exc()})

if __name__ == "__main__":
    app.run()
    #app.run(host='0.0.0.0')


