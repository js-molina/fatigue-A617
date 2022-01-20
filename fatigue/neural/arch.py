from keras.models import Model
import tensorflow as tf
from keras import layers, Input, optimizers, losses, metrics, regularizers
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer, MaxAbsScaler, PowerTransformer, Normalizer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold, LeaveOneOut, train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from scipy.signal import savgol_filter
from keras.wrappers.scikit_learn import KerasRegressor

def load_known_lstm_model(time_input_shape, const_input_shape):
    
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    # Create separate inputs for time series and constants
    time_input = Input(shape=time_input_shape, name='time_input')
    const_input = Input(shape=const_input_shape, name='const_input')

    # Feed time_input through Masking and LSTM layers
    time_mask = layers.Masking(mask_value=-999)(time_input)
    time_feats = layers.LSTM(10, return_sequences=False)(time_mask)

    # Concatenate the LSTM output with the constant input
    concat_vector = layers.concatenate([time_feats, const_input])

    # Feed through Dense layers
    dnn = layers.Dense(5, activation='relu')(concat_vector)
    life_pred = layers.Dense(1)(dnn)

    # Instantiate model
    model = Model(inputs=[time_input, const_input], outputs=[life_pred])

    # Compile
    model.compile(loss='huber_loss', optimizer=opt, metrics=[tf.keras.metrics.RootMeanSquaredError()])

    return model

def hyperx1_lstm_model(time_input_shape, const_input_shape):
    
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    # Create separate inputs for time series and constants
    time_input = Input(shape=time_input_shape, name='time_input')
    const_input = Input(shape=const_input_shape, name='const_input')

    # Feed time_input through Masking and LSTM layers
    time_mask = layers.Masking(mask_value=-999)(time_input)
    time_feats = layers.LSTM(232, return_sequences=False)(time_mask)

    # Concatenate the LSTM output with the constant input
    concat_vector = layers.concatenate([time_feats, const_input])

    # Feed through Dense layers
    dnn = layers.Dense(296, activation='relu')(concat_vector)
    life_pred = layers.Dense(1)(dnn)

    # Instantiate model
    model = Model(inputs=[time_input, const_input], outputs=[life_pred])

    # Compile
    model.compile(loss='huber_loss', optimizer=opt, metrics=[tf.keras.metrics.RootMeanSquaredError()])

    return model

def hyperx2_lstm_model(time_input_shape, const_input_shape):
    
    opt = tf.keras.optimizers.Adam(learning_rate=0.05)
    
    # Create separate inputs for time series and constants
    time_input = Input(shape=time_input_shape)
    const_input = Input(shape=const_input_shape)

    # Feed time_input through Masking and LSTM layers
    time_mask = layers.Masking(mask_value=-999)(time_input)
    time_feats = layers.LSTM(32, kernel_regularizer=regularizers.l2(),
                             recurrent_regularizer=regularizers.l2(),
                             bias_regularizer=regularizers.l2())(time_mask)

    # Concatenate the LSTM output with the constant input
    temp_vector = layers.concatenate([time_feats, const_input])

    # Feed through Dense layers
    for i in range(2):
        temp_vector = layers.Dense(128, kernel_regularizer=regularizers.l2(),
                             bias_regularizer=regularizers.l2(), activation='relu')(temp_vector)

    life_pred = layers.Dense(1)(temp_vector)

    # Instantiate model
    model = Model(inputs=[time_input, const_input], outputs=[life_pred])

    # Compile
    model.compile(loss='huber_loss', optimizer=opt, metrics=[tf.keras.metrics.RootMeanSquaredError()])

    return model

def hyperx3_lstm_model(time_input_shape, const_input_shape):
    
    opt = tf.keras.optimizers.Adam(learning_rate=0.01)
    
    # Create separate inputs for time series and constants
    time_input = Input(shape=time_input_shape, name='time_input')
    const_input = Input(shape=const_input_shape, name='const_input')

    # Feed time_input through Masking and LSTM layers
    time_mask = layers.Masking(mask_value=-999)(time_input)
    time_feats = layers.LSTM(256, kernel_regularizer=regularizers.l2(),
                             recurrent_regularizer=regularizers.l2(),
                             bias_regularizer=regularizers.l2(),
                             return_sequences=False)(time_mask)

    # Concatenate the LSTM output with the constant input
    concat_vector = layers.concatenate([time_feats, const_input])

    # Feed through Dense layers
    hidden_lay = layers.Dense(256, kernel_regularizer=regularizers.l2(),
                             bias_regularizer=regularizers.l2(), activation='relu')(concat_vector)

    drop_lay = layers.Dropout(0.5)(hidden_lay)
    
    dnn = layers.Dense(256, kernel_regularizer=regularizers.l1(),
                        bias_regularizer=regularizers.l2(), activation='relu')(drop_lay)

    drop_dnn = layers.Dropout(0.2)(dnn)

    life_pred = layers.Dense(1)(drop_dnn)

    # Instantiate model
    model = Model(inputs=[time_input, const_input], outputs=[life_pred])

    # Compile
    model.compile(loss='huber_loss', optimizer=opt, metrics=[tf.keras.metrics.RootMeanSquaredError()])

    return model

def m_lstm_shallow(time_input_shape, const_input_shape):
    
    opt = tf.keras.optimizers.Adam(learning_rate=0.01)
    
    # Create separate inputs for time series and constants
    time_input = Input(shape=time_input_shape)
    const_input = Input(shape=const_input_shape)

    # Feed time_input through Masking and LSTM layers
    time_mask = layers.Masking(mask_value=-999)(time_input)
    time_feats = layers.LSTM(16, return_sequences=False)(time_mask)

    # Concatenate the LSTM output with the constant input
    concat_vector = layers.concatenate([const_input, time_feats])

    # Feed through Dense layers
    dnn = layers.Dense(32, activation='relu')(concat_vector)
    life_pred = layers.Dense(1)(dnn)

    # Instantiate model
    model = Model(inputs=[time_input, const_input], outputs=[life_pred])

    # Compile
    model.compile(loss='huber_loss', optimizer=opt, metrics=[tf.keras.metrics.RootMeanSquaredError()])

    return model

def s_lstm_shallow(time_input_shape):
    
    opt = tf.keras.optimizers.Adam(learning_rate=0.01)
    
    # Create separate inputs for time series and constants
    time_input = Input(shape=time_input_shape)

    # Feed time_input through Masking and LSTM layers
    time_mask = layers.Masking(mask_value=-999)(time_input)
    time_feats = layers.LSTM(16, return_sequences=False)(time_mask)

    # Feed through Dense layers
    dnn = layers.Dense(32, activation='relu')(time_feats)
    life_pred = layers.Dense(1)(dnn)

    # Instantiate model
    model = Model(inputs=[time_input], outputs=[life_pred])

    # Compile
    model.compile(loss='huber_loss', optimizer=opt, metrics=[tf.keras.metrics.RootMeanSquaredError()])

    return model