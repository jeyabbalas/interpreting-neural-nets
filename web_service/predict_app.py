import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import model_from_json

import shap

import numpy as np
import pandas as pd


from flask import request
from flask import jsonify
from flask import Flask

import warnings
warnings.filterwarnings('ignore')


app = Flask(__name__)


def get_model():
    global model
    model = model_from_json(open('./model/model_architecture.json').read())
    model.load_weights('./model/model_weights.h5')
    model.compile(loss='binary_crossentropy',
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])
    print(" * Keras model loaded.")


def get_data():
    global data
    data = pd.read_csv("./data/iris_train.csv")
    print(" * Training data loaded.")


def compute_shap():
    global explainer
    explainer = shap.KernelExplainer(model.predict_proba, data)
    print(" * SHAP values computed.")


def abs_adimensional_jacobian_1output(x, model):
    x_tensor = tf.convert_to_tensor(x.reshape(1, -1), dtype=tf.float32)
    with tf.GradientTape() as g:
        g.watch(x_tensor)
        y_tensor = model(x_tensor)
    jacobian = g.jacobian(y_tensor, x_tensor)
    jacobian = jacobian.numpy()[0][0][0]
    input_by_loss = x / y_tensor.numpy()[0][0]
    adim_jacobian = np.multiply(jacobian, input_by_loss)
    return np.absolute(adim_jacobian)


def weigh_instance_by_sensitivity(x, model):
    sens_wts = abs_adimensional_jacobian_1output(x, model)
    weighted_x = np.multiply(sens_wts, x)
    return weighted_x


def weigh_matrix_by_sensitivity(reference, model):
    return np.apply_along_axis(weigh_instance_by_sensitivity,
                               1,
                               reference, model=model)


def min_weighted_euclidean_dist(query, reference, model):
    weighted_query = weigh_instance_by_sensitivity(query, model)
    weighted_ref = weigh_matrix_by_sensitivity(reference, model)
    dists = np.apply_along_axis(np.linalg.norm,
                                1,
                                (weighted_ref - weighted_query))
    min_dist = min(dists)
    return min_dist


def in_forbidden_zone(query):
    return np.any(query < 0)


def compute_novelty_index(query, nov_deno, reference, model):
    if(in_forbidden_zone(query)):
        return 10000.0
    nov_nume = min_weighted_euclidean_dist(query, reference, model)
    return nov_nume / nov_deno


print(" * Loading Keras model...")
get_model()
print(" * Loading Training data...")
get_data()
print(" * Computing SHAP values...")
compute_shap()


@app.route("/predict", methods=['POST'])
def predict():
    message = request.get_json(force=True)
    sepal_length = float(message['sepal_length'])
    sepal_width = float(message['sepal_width'])
    petal_length = float(message['petal_length'])
    petal_width = float(message['petal_width'])

    query = np.array([sepal_length, sepal_width,
                      petal_length, petal_width]).reshape(1, -1)

    novelty = compute_novelty_index(query, 1.7835123384693827,
                                    data.to_numpy(), model)

    y_pred = model.predict(query).tolist()

    query_pd = pd.DataFrame(query, columns=data.columns)
    shap_values = explainer.shap_values(query_pd, nsamples=100)

    response = {
        'novelty': novelty,
        'prediction': y_pred[0][0],
        'baseValue': explainer.expected_value[0],
        'featureEffects': {
            '0': shap_values[0][0, :][0],
            '1': shap_values[0][0, :][1],
            '2': shap_values[0][0, :][2],
            '3': shap_values[0][0, :][3]
        }
    }

    return jsonify(response)
