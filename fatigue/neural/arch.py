from keras.models import Model, Sequential
import tensorflow as tf
from keras import layers, Input, optimizers, losses, metrics, regularizers
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer, MaxAbsScaler, PowerTransformer, Normalizer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold, LeaveOneOut, train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from scipy.signal import savgol_filter
from keras.wrappers.scikit_learn import KerasRegressor

metrics = [tf.keras.metrics.RootMeanSquaredError(), 'mean_absolute_percentage_error']

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
    model.compile(loss='huber_loss', optimizer=opt, metrics=metrics)

    return model

def hyperx1(time_input_shape, const_input_shape):
    
    opt = tf.keras.optimizers.Adam(learning_rate=0.05)
    
    # Create separate inputs for time series and constants
    time_input = Input(shape=time_input_shape, name='time_input')
    const_input = Input(shape=const_input_shape, name='const_input')

    # Feed time_input through Masking and LSTM layers
    time_mask = layers.Masking(mask_value=-999)(time_input)
    time_feats = layers.LSTM(8, kernel_regularizer=regularizers.l1_l2(1.703159e-5),
                             recurrent_regularizer=regularizers.l1_l2(3.95848e-10),
                             bias_regularizer=regularizers.l1_l2(0.0917412))(time_mask)

    # Concatenate the LSTM output with the constant input
    concat_vector = layers.concatenate([time_feats, const_input])

    # Feed through Dense layers
    dnn = layers.Dense(54, kernel_regularizer=regularizers.l1_l2(0.00070517),
                       bias_regularizer=regularizers.l1_l2(3.81407e-11), activation='relu')(concat_vector)
    life_pred = layers.Dense(1)(dnn)

    # Instantiate model
    model = Model(inputs=[time_input, const_input], outputs=[life_pred])

    # Compile
    model.compile(loss='mean_absolute_percentage_error', optimizer=opt, metrics=metrics)

    return model

def hyperx2(time_input_shape, const_input_shape):
    
    opt = tf.keras.optimizers.Adam(learning_rate=0.05)
    
    # Create separate inputs for time series and constants
    time_input = Input(shape=time_input_shape)
    const_input = Input(shape=const_input_shape)

    # Feed time_input through Masking and LSTM layers
    time_mask = layers.Masking(mask_value=-999)(time_input)
    time_feats = layers.LSTM(12, kernel_regularizer=regularizers.l2(2.12126544861303e-12),
                             recurrent_regularizer=regularizers.l2(1.974246992291062e-06),
                             bias_regularizer=regularizers.l2(1.9853483677831437e-08))(time_mask)

    # Concatenate the LSTM output with the constant input
    temp_vector = layers.concatenate([time_feats, const_input])

    # Feed through Dense layers
    temp_vector = layers.Dense(58, kernel_regularizer=regularizers.l2(5.147420852142302e-10),
                            bias_regularizer=regularizers.l2(3.8026322361203077e-10), activation='relu')(temp_vector)

    life_pred = layers.Dense(1)(temp_vector)

    # Instantiate model
    model = Model(inputs=[time_input, const_input], outputs=[life_pred])

    # Compile
    model.compile(loss='huber_loss', optimizer=opt, metrics=metrics)

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
    model.compile(loss='huber_loss', optimizer=opt, metrics=metrics)

    return model

def hyperx2_gru_model(time_input_shape, const_input_shape):
    
    opt = tf.keras.optimizers.Adam(learning_rate=0.05)
    
    # Create separate inputs for time series and constants
    time_input = Input(shape=time_input_shape)
    const_input = Input(shape=const_input_shape)

    # Feed time_input through Masking and LSTM layers
    time_mask = layers.Masking(mask_value=-999)(time_input)
    time_feats = layers.GRU(32, kernel_regularizer=regularizers.l2(),
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
    model.compile(loss='huber_loss', optimizer=opt, metrics=metrics)

    return model

def hyperx3(time_input_shape, const_input_shape):
    
    opt = tf.keras.optimizers.Adam(learning_rate=0.05)
    
    # Create separate inputs for time series and constants
    time_input = Input(shape=time_input_shape, name='time_input')
    const_input = Input(shape=const_input_shape, name='const_input')

    # Feed time_input through Masking and LSTM layers
    time_mask = layers.Masking(mask_value=-999)(time_input)
    time_feats = layers.LSTM(40, kernel_regularizer=regularizers.l1_l2(0.013491474174332142, 0.004355987086768707),
                             recurrent_regularizer=regularizers.l1_l2(3.5788076188136305e-06, 0.038251121538016755),
                             bias_regularizer=regularizers.l1_l2(2.358256969621759e-09, 2.7988380055278342e-09),
                             return_sequences=False)(time_mask)

    # Concatenate the LSTM output with the constant input
    concat_vector = layers.concatenate([time_feats, const_input])

    dnn = layers.Dense(34, kernel_regularizer=regularizers.l1_l2(3.6700055846798666e-07, 1.2636301180302531e-10),
                        bias_regularizer=regularizers.l1_l2(4.462083996604814e-05, 1.296603217683237e-10), activation='relu')(concat_vector)

    life_pred = layers.Dense(1)(dnn)

    # Instantiate model
    model = Model(inputs=[time_input, const_input], outputs=[life_pred])

    # Compile
    model.compile(loss='huber_loss', optimizer=opt, metrics=metrics)

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
    model.compile(loss='huber_loss', optimizer=opt, metrics=metrics)

    return model

def m_lstm_deep(time_input_shape, const_input_shape):
    
    opt = tf.keras.optimizers.Adam(learning_rate=0.05)
    
    # Create separate inputs for time series and constants
    time_input = Input(shape=time_input_shape)
    const_input = Input(shape=const_input_shape)

    # Feed time_input through Masking and LSTM layers
    time_mask = layers.Masking(mask_value=-999)(time_input)
    time_feats = layers.LSTM(8)(time_mask)

    # Concatenate the LSTM output with the constant input
    temp_vector = layers.concatenate([time_feats, const_input])
    hidden_units = [11, 122]
    # Feed through Dense layers
    for i in range(2):
        temp_vector = layers.Dense(hidden_units[i], activation='relu')(temp_vector)

    life_pred = layers.Dense(1)(temp_vector)

    # Instantiate model
    model = Model(inputs=[time_input, const_input], outputs=[life_pred])

    # Compile
    model.compile(loss='huber_loss', optimizer=opt, metrics=metrics)

    return model

def m_gru_deep(time_input_shape, const_input_shape):
    
    opt = tf.keras.optimizers.Adam(learning_rate=0.05)
    
    # Create separate inputs for time series and constants
    time_input = Input(shape=time_input_shape)
    const_input = Input(shape=const_input_shape)

    # Feed time_input through Masking and LSTM layers
    time_mask = layers.Masking(mask_value=-999)(time_input)
    time_feats = layers.LSTM(8)(time_mask)

    # Concatenate the LSTM output with the constant input
    temp_vector = layers.concatenate([time_feats, const_input])
    hidden_units = [11, 122]
    # Feed through Dense layers
    for i in range(2):
        temp_vector = layers.Dense(hidden_units[i], activation='relu')(temp_vector)

    life_pred = layers.Dense(1)(temp_vector)

    # Instantiate model
    model = Model(inputs=[time_input, const_input], outputs=[life_pred])

    # Compile
    model.compile(loss='huber_loss', optimizer=opt, metrics=metrics)

    return model

def m_lstmconv_deep(tshape, cshape):
    
    time_input = Input(shape=tshape)
    const_input = Input(shape=cshape)
    
    time_mask = layers.Masking(mask_value=-999)(time_input)
    
    bi1 = layers.Bidirectional(layers.LSTM(32, return_sequences = True)(time_mask))
    bi2 = layers.Bidirectional(layers.LSTM(32, return_sequences = True)(bi1))
    bi3 = layers.Bidirectional(layers.LSTM(32)(bi2))
    
    concat_vector = layers.concatenate([const_input, bi3])
    
    dense = layers.Dense(32, kernel_regularizer=regularizers.l1_l2(),
              bias_regularizer=regularizers.l1_l2(), activation='relu')(concat_vector)
    
    out = layers.Dense(1, activation='relu')(dense)
    
    opt = tf.keras.optimizers.Adam(learning_rate=0.04)
    
    model = Model(inputs=[time_input, const_input], outputs=[out])
    
    model.compile(loss='huber_loss', optimizer=opt, metrics=metrics)

    return model

def m_lstm_deep_r_drop(time_input_shape, const_input_shape):
    
    opt = tf.keras.optimizers.Adam(learning_rate=0.05)
    
    # Create separate inputs for time series and constants
    time_input = Input(shape=time_input_shape)
    const_input = Input(shape=const_input_shape)

    # Feed time_input through Masking and LSTM layers
    time_mask = layers.Masking(mask_value=-999)(time_input)
    time_feats = layers.Dropout(0.1)(layers.LSTM(32)(time_mask))

    # Concatenate the LSTM output with the constant input
    temp_vector = layers.concatenate([time_feats, const_input])

    # Feed through Dense layers
    for i in range(2):
        temp_vector = layers.Dropout(0.5)(layers.Dense(256, activation='relu')(temp_vector))

    life_pred = layers.Dense(1)(temp_vector)

    # Instantiate model
    model = Model(inputs=[time_input, const_input], outputs=[life_pred])

    # Compile
    model.compile(loss='huber_loss', optimizer=opt, metrics=metrics)

    return model

def m_lstm_s(time_input_shape, const_input_shape, l0, l1):
    
    opt = tf.keras.optimizers.Adam(learning_rate=0.05)
    
    # Create separate inputs for time series and constants
    time_input = Input(shape=time_input_shape)
    const_input = Input(shape=const_input_shape)

    # Feed time_input through Masking and LSTM layers
    time_mask = layers.Masking(mask_value=-999)(time_input)
    time_feats = layers.LSTM(l0)(time_mask)

    # Concatenate the LSTM output with the constant input
    temp_vector = layers.concatenate([time_feats, const_input])

    # Feed through Dense layers
    for i in range(1):
        temp_vector = layers.Dense(l1, activation='relu')(temp_vector)

    life_pred = layers.Dense(1)(temp_vector)

    # Instantiate model
    model = Model(inputs=[time_input, const_input], outputs=[life_pred])

    # Compile
    model.compile(loss='huber_loss', optimizer=opt, metrics=metrics)

    return model

def m_lstm_deep_r_l1l2(time_input_shape, const_input_shape):
    
    opt = tf.keras.optimizers.Adam(learning_rate=0.05)
    
    # Create separate inputs for time series and constants
    time_input = Input(shape=time_input_shape)
    const_input = Input(shape=const_input_shape)

    # Feed time_input through Masking and LSTM layers
    time_mask = layers.Masking(mask_value=-999)(time_input)
    time_feats = layers.LSTM(8, kernel_regularizer=regularizers.l1_l2(0.006),
                             recurrent_regularizer=regularizers.l1_l2(0.05),
                             bias_regularizer=regularizers.l1_l2(0.0138))(time_mask)

    # Concatenate the LSTM output with the constant input
    temp_vector = layers.concatenate([time_feats, const_input])
    hidden_units = [11, 122]
    hidden_kr = [0.00034, 0.0028]
    hidden_br = [0.0076, 0.0723]
    # Feed through Dense layers
    for i in range(2):
        temp_vector = layers.Dense(hidden_units[i], kernel_regularizer=regularizers.l1_l2(hidden_kr[i]),
                             bias_regularizer=regularizers.l1_l2(hidden_br[i]), activation='relu')(temp_vector)

    life_pred = layers.Dense(1)(temp_vector)

    # Instantiate model
    model = Model(inputs=[time_input, const_input], outputs=[life_pred])

    # Compile
    model.compile(loss='huber_loss', optimizer=opt, metrics=metrics)

    return model

def m_lstm_r(time_input_shape, const_input_shape):
    
    opt = tf.keras.optimizers.Adam(learning_rate=0.05)
    
    # Create separate inputs for time series and constants
    time_input = Input(shape=time_input_shape)
    const_input = Input(shape=const_input_shape)

    # Feed time_input through Masking and LSTM layers
    time_mask = layers.Masking(mask_value=-999)(time_input)
    time_feats = layers.LSTM(24, kernel_regularizer=regularizers.l1_l2(5.68313e-5),
                             recurrent_regularizer=regularizers.l1_l2(1.504424e-5),
                             bias_regularizer=regularizers.l1_l2(0.00028161))(time_mask)

    # Concatenate the LSTM output with the constant input
    temp_vector = layers.concatenate([time_feats, const_input])
    hidden_units = [20, 13]
    hidden_kr = [5.08641e-5, 0.0155066]
    hidden_br = [0.0035155, 0.00421745]
    # Feed through Dense layers
    for i in range(2):
        temp_vector = layers.Dense(hidden_units[i], kernel_regularizer=regularizers.l1_l2(hidden_kr[i]),
                             bias_regularizer=regularizers.l1_l2(hidden_br[i]), activation='relu')(temp_vector)

    life_pred = layers.Dense(1)(temp_vector)

    # Instantiate model
    model = Model(inputs=[time_input, const_input], outputs=[life_pred])

    # Compile
    model.compile(loss='huber_loss', optimizer=opt, metrics=metrics)

    return model
    
def m_lstm_r2(time_input_shape, const_input_shape):
    
    opt = tf.keras.optimizers.Adam(learning_rate=0.05)
    
    # Create separate inputs for time series and constants
    time_input = Input(shape=time_input_shape)
    const_input = Input(shape=const_input_shape)

    # Feed time_input through Masking and LSTM layers
    time_mask = layers.Masking(mask_value=-999)(time_input)
    time_feats = layers.LSTM(8, kernel_regularizer=regularizers.l1_l2(0.00028462800560779536),
                             recurrent_regularizer=regularizers.l1_l2(1.9268887183442944e-05),
                             bias_regularizer=regularizers.l1_l2(3.110100698824893e-05))(time_mask)

    # Concatenate the LSTM output with the constant input
    temp_vector = layers.concatenate([time_feats, const_input])
    hidden_units = [20, 9]
    hidden_kr = [0.061126517453785154, 1.3260242610312342e-05]
    hidden_br = [0.008262816441779708, 0.004629817692328819]
    # Feed through Dense layers
    for i in range(2):
        temp_vector = layers.Dense(hidden_units[i], kernel_regularizer=regularizers.l1_l2(hidden_kr[i]),
                             bias_regularizer=regularizers.l1_l2(hidden_br[i]), activation='relu')(temp_vector)

    life_pred = layers.Dense(1)(temp_vector)

    # Instantiate model
    model = Model(inputs=[time_input, const_input], outputs=[life_pred])

    # Compile
    model.compile(loss='huber_loss', optimizer=opt, metrics=metrics)

    return model

def m_gru_r(time_input_shape, const_input_shape):
    
    opt = tf.keras.optimizers.Adam(learning_rate=0.05)
    
    # Create separate inputs for time series and constants
    time_input = Input(shape=time_input_shape)
    const_input = Input(shape=const_input_shape)

    # Feed time_input through Masking and LSTM layers
    time_mask = layers.Masking(mask_value=-999)(time_input)
    time_feats = layers.GRU(40, kernel_regularizer=regularizers.l1_l2(0.020024),
                             recurrent_regularizer=regularizers.l1_l2(0.0001285),
                             bias_regularizer=regularizers.l1_l2(0.0001036))(time_mask)

    # Concatenate the LSTM output with the constant input
    temp_vector = layers.concatenate([time_feats, const_input])
    hidden_units = [21, 54]
    hidden_kr = [1.27863e-5, 0.001084]
    hidden_br = [0.0059324, 7.82817e-5]
    # Feed through Dense layers
    for i in range(2):
        temp_vector = layers.Dense(hidden_units[i], kernel_regularizer=regularizers.l1_l2(hidden_kr[i]),
                             bias_regularizer=regularizers.l1_l2(hidden_br[i]), activation='relu')(temp_vector)

    life_pred = layers.Dense(1)(temp_vector)

    # Instantiate model
    model = Model(inputs=[time_input, const_input], outputs=[life_pred])

    # Compile
    model.compile(loss='huber_loss', optimizer=opt, metrics=metrics)

    return model

def m_gru_r2(time_input_shape, const_input_shape):
    
    opt = tf.keras.optimizers.Adam(learning_rate=0.05)
    
    # Create separate inputs for time series and constants
    time_input = Input(shape=time_input_shape)
    const_input = Input(shape=const_input_shape)

    # Feed time_input through Masking and LSTM layers
    time_mask = layers.Masking(mask_value=-999)(time_input)
    time_feats = layers.GRU(8, kernel_regularizer=regularizers.l1_l2(0.0017861698065506438),
                             recurrent_regularizer=regularizers.l1_l2(1.80195014451597e-05),
                             bias_regularizer=regularizers.l1_l2(0.07707716367029467))(time_mask)

    # Concatenate the LSTM output with the constant input
    temp_vector = layers.concatenate([time_feats, const_input])
    hidden_units = [18, 187]
    hidden_kr = [0.00270570561993386, 5.672007235312129e-05]
    hidden_br = [0.0019489296682414366, 0.0005965532891851287]
    # Feed through Dense layers
    for i in range(2):
        temp_vector = layers.Dense(hidden_units[i], kernel_regularizer=regularizers.l1_l2(hidden_kr[i]),
                             bias_regularizer=regularizers.l1_l2(hidden_br[i]), activation='relu')(temp_vector)

    life_pred = layers.Dense(1)(temp_vector)

    # Instantiate model
    model = Model(inputs=[time_input, const_input], outputs=[life_pred])

    # Compile
    model.compile(loss='huber_loss', optimizer=opt, metrics=metrics)

    return model

def m_lstm_best(time_input_shape, const_input_shape):
    
    opt = tf.keras.optimizers.Adam(learning_rate=0.05)
    
    # Create separate inputs for time series and constants
    time_input = Input(shape=time_input_shape)
    const_input = Input(shape=const_input_shape)

    # Feed time_input through Masking and LSTM layers
    time_mask = layers.Masking(mask_value=-999)(time_input)
    time_feats = layers.LSTM(42, kernel_regularizer=regularizers.l1_l2(9.301e-7),
                             recurrent_regularizer=regularizers.l1_l2(0.004248),
                             bias_regularizer=regularizers.l1_l2(8.7213e-12))(time_mask)

    # Concatenate the LSTM output with the constant input
    temp_vector = layers.concatenate([time_feats, const_input])

    temp_vector = layers.Dense(23, kernel_regularizer=regularizers.l1_l2(0.0044339),
                            bias_regularizer=regularizers.l1_l2(0.01275), activation='relu')(temp_vector)

    life_pred = layers.Dense(1)(temp_vector)

    # Instantiate model
    model = Model(inputs=[time_input, const_input], outputs=[life_pred])

    # Compile
    model.compile(loss='huber_loss', optimizer=opt, metrics=metrics)

    return model
    
def m_lstm_best2(time_input_shape, const_input_shape):
    
    opt = tf.keras.optimizers.Adam(learning_rate=0.05)
    
    # Create separate inputs for time series and constants
    time_input = Input(shape=time_input_shape)
    const_input = Input(shape=const_input_shape)

    # Feed time_input through Masking and LSTM layers
    time_mask = layers.Masking(mask_value=-999)(time_input)
    time_feats = layers.LSTM(15, kernel_regularizer=regularizers.l1_l2(0.03845357445881336),
                             recurrent_regularizer=regularizers.l1_l2(0.005064922262886326),
                             bias_regularizer=regularizers.l1_l2(2.249429023993107e-06))(time_mask)

    # Concatenate the LSTM output with the constant input
    temp_vector = layers.concatenate([time_feats, const_input])

    # Feed through Dense layers

    temp_vector = layers.Dense(10, kernel_regularizer=regularizers.l1_l2(0.009229756020324546),
                            bias_regularizer=regularizers.l1_l2(0.009275976478015173), activation='relu')(temp_vector)

    life_pred = layers.Dense(1)(temp_vector)

    # Instantiate model
    model = Model(inputs=[time_input, const_input], outputs=[life_pred])

    # Compile
    model.compile(loss='huber_loss', optimizer=opt, metrics=metrics)

    return model

def m_lstm_best3(time_input_shape, const_input_shape):
    
    opt = tf.keras.optimizers.Adam(learning_rate=0.05)
    
    # Create separate inputs for time series and constants
    time_input = Input(shape=time_input_shape)
    const_input = Input(shape=const_input_shape)

    # Feed time_input through Masking and LSTM layers
    time_mask = layers.Masking(mask_value=-999)(time_input)
    time_feats = layers.LSTM(39, kernel_regularizer=regularizers.l1_l2(7.172086325519386e-05),
                             recurrent_regularizer=regularizers.l1_l2(0.00014104724799327662),
                             bias_regularizer=regularizers.l1_l2(0.025795580427727564))(time_mask)

    # Concatenate the LSTM output with the constant input
    temp_vector = layers.concatenate([time_feats, const_input])

    # Feed through Dense layers

    temp_vector = layers.Dense(62, kernel_regularizer=regularizers.l1_l2(3.5955842989312775e-06),
                            bias_regularizer=regularizers.l1_l2(0.00887381861161938), activation='relu')(temp_vector)

    life_pred = layers.Dense(1)(temp_vector)

    # Instantiate model
    model = Model(inputs=[time_input, const_input], outputs=[life_pred])

    # Compile
    model.compile(loss='huber_loss', optimizer=opt, metrics=metrics)

    return model

def m_gru_r_l1l2(time_input_shape, const_input_shape):
    
    opt = tf.keras.optimizers.Adam(learning_rate=0.05)
    
    # Create separate inputs for time series and constants
    time_input = Input(shape=time_input_shape)
    const_input = Input(shape=const_input_shape)

    # Feed time_input through Masking and LSTM layers
    time_mask = layers.Masking(mask_value=-999)(time_input)
    time_feats = layers.GRU(8, kernel_regularizer=regularizers.l1_l2(0.000263),
                             recurrent_regularizer=regularizers.l1_l2(0.01412),
                             bias_regularizer=regularizers.l1_l2(0.000126))(time_mask)

    # Concatenate the LSTM output with the constant input
    temp_vector = layers.concatenate([time_feats, const_input])
    hidden_units = [10, 468]
    hidden_kr = [0.0038, 0.000214]
    hidden_br = [0.00297, 0.05646]
    # Feed through Dense layers
    for i in range(2):
        temp_vector = layers.Dense(hidden_units[i], kernel_regularizer=regularizers.l1_l2(hidden_kr[i]),
                             bias_regularizer=regularizers.l1_l2(hidden_br[i]), activation='relu')(temp_vector)

    life_pred = layers.Dense(1)(temp_vector)

    # Instantiate model
    model = Model(inputs=[time_input, const_input], outputs=[life_pred])

    # Compile
    model.compile(loss='huber_loss', optimizer=opt, metrics=metrics)

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
    model.compile(loss='huber_loss', optimizer=opt, metrics=metrics)

    return model

def s_lstm_deep_r_drop(time_input_shape):
    
    opt = tf.keras.optimizers.Adam(learning_rate=0.01)
    
    # Create separate inputs for time series and constants
    time_input = Input(shape=time_input_shape)

    # Feed time_input through Masking and LSTM layers
    time_mask = layers.Masking(mask_value=-999)(time_input)
    time_feats = layers.LSTM(56, return_sequences=False)(time_mask)
    d1 = layers.Dropout(0.4)(time_feats)

    # Feed through Dense layers
    d1 = layers.Dropout(0.8)(layers.Dense(232, activation='relu')(d1))
    d1 = layers.Dropout(0.2)(layers.Dense(424, activation='relu')(d1))

    life_pred = layers.Dense(1)(d1)

    # Instantiate model
    model = Model(inputs=[time_input], outputs=[life_pred])

    # Compile
    model.compile(loss='huber_loss', optimizer=opt, metrics=metrics)

    return model

def s_lstmconv_deep(tshape):
    
    model = Sequential()
    
    model.add(layers.Bidirectional(layers.LSTM(32, return_sequences = True, input_shape = tshape)))
    model.add(layers.Bidirectional(layers.LSTM(32, return_sequences = True)))
    model.add(layers.Bidirectional(layers.LSTM(32)))
    model.add(layers.Dense(32, kernel_regularizer=regularizers.l1_l2(),
              bias_regularizer=regularizers.l1_l2(), activation='relu'))
    
    model.add(layers.Dense(1, activation='relu'))
    
    opt = tf.keras.optimizers.Adam(learning_rate=0.04)
    
    model.compile(loss='huber_loss', optimizer=opt, metrics=metrics)

    return model