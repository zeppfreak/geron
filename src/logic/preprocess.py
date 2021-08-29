import tensorflow as tf

def create_preprocess(model_id):
    if model_id == "singleinput_singleoutput":
        return _preprocess__singleinput__singleoutput
    elif model_id == "multiinput_singleoutput":
        return _preprocess_multiinput__singleoutput
    elif model_id == "multiinput_multioutput":
        return _preprocess_multiinput__multioutput

@tf.function
def _preprocess__singleinput__singleoutput(line):
    n_inputs = 8 
    
    defs = [0.] * n_inputs + [tf.constant([], dtype=tf.float32)]
    fields = tf.io.decode_csv(line, record_defaults=defs)
    x = tf.stack(fields[:-1])
    y = tf.stack(fields[-1:])
    return x,y

@tf.function
def _preprocess_multiinput__singleoutput(line):
    n_inputs = 8
    
    defs = [0.] * n_inputs + [tf.constant([], dtype=tf.float32)]
    fields = tf.io.decode_csv(line, record_defaults=defs)
    x_A = tf.stack(fields[:5])
    x_B = tf.stack(fields[2:-1])
    y = tf.stack(fields[-1:])
    return (x_A, x_B), y

@tf.function
def _preprocess_multiinput__multioutput(line):
    n_inputs = 8
    
    defs = [0.] * n_inputs + [tf.constant([], dtype=tf.float32)]
    fields = tf.io.decode_csv(line, record_defaults=defs)
    x_A = tf.stack(fields[:5])
    x_B = tf.stack(fields[2:-1])
    y = tf.stack(fields[-1:])
    return (x_A, x_B), (y, y)