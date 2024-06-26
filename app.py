from flask import Flask, request
import json
import pandas as pd
import numpy as np
import pickle


app = Flask(__name__)

model = pickle.load(open('log/clfmodel.pkl','rb'))


@app.route('/', methods=['POST', 'GET'])
def handle_request():
  
    payload=request.get_json()
    #data = np.array(list(payload.values())).reshape(1,-1)
    data = pd.DataFrame.from_dict(payload)
    output=model.predict(data)

    response =  {'prediction': output[0].item()}

    return json.dumps(response)
  


if __name__ == "__main__":
    app.run()



