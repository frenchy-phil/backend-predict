from flask import Flask, jsonify, request
import joblib
import numpy as np
import shap
import pickle
import pandas as pd

import json
from json import JSONEncoder
from sklearn.externals import joblib
from utils import tokenize

app = Flask(__name__)


model = pickle.load(open('model.pkl','rb'))
data=pd.read_csv('df_test_sample.csv')
#valid_x=pd.read_csv('valid_x2.csv')
listid=data['SK_ID_CURR'].tolist()





@app.route('/predict', methods=['POST'])
    print('bonjour')
'''def predict():
    iddic=request.get_json(force=True)
    idval = iddic.values()
    id = int(list(idval)[0])
    x=data.loc[data['SK_ID_CURR'] == id]
    y=model.predict_proba(x, num_iteration=model.best_iteration_)[:, 1]
    return jsonify(y.tolist())
'''
if __name__ == '__main__':
    app.run()

