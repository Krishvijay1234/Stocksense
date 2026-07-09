from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import joblib
import json
import pandas as pd

import house_price_prediction as hpp

app = Flask(__name__)
CORS(app)

MODEL_PATH = 'model.pkl'
COLS_PATH = 'columns.json'


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})


@app.route('/train', methods=['POST'])
def train():
    data = request.get_json(silent=True) or {}
    train_path = data.get('train_path', 'train.csv')
    test_path = data.get('test_path', 'test.csv')

    if not os.path.exists(train_path):
        return jsonify({'error': f'{train_path} not found'}), 400

    model_path, cols_path = hpp.train_and_save(train_path, test_path, MODEL_PATH, COLS_PATH)
    return jsonify({'message': 'training completed', 'model_path': model_path, 'cols_path': cols_path})


@app.route('/predict', methods=['POST'])
def predict():
    # Accept JSON with 'rows': [ {col: val, ...}, ... ]
    payload = request.get_json(silent=True)
    if not payload:
        return jsonify({'error': 'Invalid or missing JSON body'}), 400

    rows = payload.get('rows')
    if rows is None:
        return jsonify({'error': "Missing 'rows' in JSON body"}), 400

    if not os.path.exists(MODEL_PATH) or not os.path.exists(COLS_PATH):
        return jsonify({'error': 'Model artifacts not found. Run /train first.'}), 400

    model = joblib.load(MODEL_PATH)
    with open(COLS_PATH, 'r', encoding='utf-8') as f:
        cols = json.load(f)

    df = pd.DataFrame(rows)
    # Align to training columns
    df = df.reindex(columns=cols, fill_value=0)

    preds = model.predict(df)
    results = preds.tolist()
    return jsonify({'predictions': results})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
