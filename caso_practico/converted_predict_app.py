import pickle

from flask import Flask, jsonify, request
from converted_predict_service import predict_single

app = Flask('converted-predict')

with open('/home/usuario/caso-practico/models/converted-model.pck', 'rb') as f:
    dv, model = pickle.load(f)


@app.route('/predict', methods=['POST'])
def predict():
    cliente = request.get_json()
    converted, prediction = predict_single(cliente, dv, model)

    result = {
        'converted': bool(converted),
        'converted_probability': float(prediction),
    }

    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, port=8000)