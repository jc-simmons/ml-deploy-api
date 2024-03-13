from flask import Flask, request
import json
import pandas as pd
import numpy as np
import pickle


app = Flask(__name__)

model = pickle.load(open('clfmodel.pkl','rb'))


@app.route('/', methods=['POST', 'GET'])
def handle_request():
  
    payload=request.get_json()
    data = np.array(list(payload.values())).reshape(1,-1)
    output=model.predict(data)
    #text = str(request.args.get('input'))
    #characters = len(text)
    #out_data = {'chars': characters}

    response =  {'prediction': output[0].item()}

    return json.dumps(response)
  


if __name__ == "__main__":
    app.run()

    #df = pd.read_csv('diabetes.csv')
    #pickled_model.predict(df[['Pregnancies','PlasmaGlucose','DiastolicBloodPressure','TricepsThickness',
    #'SerumInsulin','BMI','DiabetesPedigree','Age']].iloc[0:60].to_numpy())


