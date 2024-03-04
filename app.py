from flask import Flask, request
import json
import requests

app = Flask(__name__)

@app.route('/', methods=['POST', 'GET'])
def handle_request():
  
    text = str(request.args.get('input'))
    characters = len(text)
    out_data = {'chars': characters}
  
    return json.dumps(out_data)
  

if __name__ == "__main__":
    app.run()


