import tensorflow as tf
from tensorflow import keras

def create_model(model_id):
    if model_id == "singleinput_singleoutput":
        input   = keras.layers.Input(shape=(8,), name="deep_input")
        hidden1 = keras.layers.Dense(30, activation="relu")(input)
        hidden2 = keras.layers.Dense(30, activation="relu")(hidden1)
        output = keras.layers.Dense(1, name="output")(hidden2)
        model = keras.Model(inputs=[input], outputs=[output])
    elif model_id == "multiinput_singleoutput":
        input_A = keras.layers.Input(shape=(5,), name="wide_input")
        input_B = keras.layers.Input(shape=(6,), name="deep_input")
        hidden1 = keras.layers.Dense(30, activation="relu")(input_B)
        hidden2 = keras.layers.Dense(30, activation="relu")(hidden1)
        concat = keras.layers.concatenate([input_A, hidden2])
        output = keras.layers.Dense(1, name="output")(concat)
        model = keras.Model(inputs=[input_A, input_B], outputs=[output]) 
    elif model_id == "multiinput_multioutput":
        input_A = keras.layers.Input(shape=(5,), name="wide_input")
        input_B = keras.layers.Input(shape=(6,), name="deep_input")
        hidden1 = keras.layers.Dense(30, activation="relu")(input_B)
        hidden2 = keras.layers.Dense(30, activation="relu")(hidden1)
        concat = keras.layers.concatenate([input_A, hidden2])
        output = keras.layers.Dense(1, name="output")(concat)
        aux_output = keras.layers.Dense(1, name="aux_output")(hidden2)
        model = keras.Model(inputs=[input_A, input_B], outputs=[output, aux_output])

    return model