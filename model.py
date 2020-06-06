from tensorflow import keras
from tensorflow.keras import backend as K

def Deep_DenseOnly(input_dim,
                   wide,
                   activate):
    ###########################
    # Building model by keras #
    ###########################

    layer_num = len(wide)
    assert len(wide)==len(activate), "Sizes of wide and activate is not equal."

    model = keras.Sequential()
    model.add(keras.layers.Dense(wide[0],
                                 input_shape=(input_dim,),
                                 activation=activate[0]))

    for i in range(1, layer_num):
        model.add(keras.layers.Dense(wide[i],
                                     activation=activate[i]))
    model.add(keras.layers.Dense(1,
                                 activation='sigmoid'))
    model.summary()
    return model

def metric(y_true, y_pred):
    #y_pred = K.map_fn(lambda x: 0 if x<0 else 1, y_pred)
    return keras.metrics.binary_accuracy(y_true, y_pred, 0.5)
            
  

        

