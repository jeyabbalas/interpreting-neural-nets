import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import model_from_json

import numpy as np
import pandas as pd


from flask import request
from flask import jsonify
from flask import Flask

app = Flask(__name__)


def get_model():
    global model
    model = model_from_json(open('./model/model_architecture.json').read())
    model.load_weights('./model/model_weights.h5')
    model.compile(loss='binary_crossentropy',
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])
    print(" * Keras model loaded")


print(" * Loading Keras model")
get_model()


@app.route("/predict", methods=['POST'])
def predict():
    message = request.get_json(force=True)
    sepal_length = float(message['sepal_length'])
    sepal_width = float(message['sepal_width'])
    petal_length = float(message['petal_length'])
    petal_width = float(message['petal_width'])

    query = np.array([sepal_length, sepal_width,
                      petal_length, petal_width]).reshape(1, -1)
    y_pred = model.predict(query).tolist()
    print(y_pred)

    response = {
        'prediction': y_pred[0][0]
    }

    return jsonify(response)
