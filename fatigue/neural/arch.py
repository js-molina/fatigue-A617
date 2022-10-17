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

def full1(time_input_shape, const_input_shape):
    
    opt = tf.keras.optimizers.Adam(learning_rate=0.01)
    
    time_input = Input(shape=time_input_shape)
    const_input = Input(shape=const_input_shape)

    time_mask = layers.Masking(mask_value=-999)(time_input)

    time_feats = layers.LSTM(54, kernel_regularizer=regularizers.l1_l2(5.51590773056887e-09, 8.81297292087968e-12),
                            recurrent_regularizer=regularizers.l1_l2(2.4487689653795996e-09, 0.001484564272686839),
                            bias_regularizer=regularizers.l1_l2(1.4242687029764056e-07, 0.05199236422777176))(time_mask)

    temp_vector = layers.concatenate([time_feats, const_input])
    temp_vector = layers.Dense(58, kernel_regularizer=regularizers.l1_l2(0.021888492628932, 3.5722358404655097e-09),
    bias_regularizer=regularizers.l1_l2(0.0029191209468990564, 2.314221092092339e-06), activation='relu')(temp_vector)
    life_pred = layers.Dense(1)(temp_vector)

    model = Model(inputs=[time_input, const_input], outputs=[life_pred])

    model.compile(loss='huber_loss', optimizer=opt, metrics=metrics)

    return model

def hyperx1(time_input_shape, const_input_shape):
    
    opt = tf.keras.optimizers.Adam(learning_rate=0.01)
    
    time_input = Input(shape=time_input_shape)
    const_input = Input(shape=const_input_shape)

    time_mask = layers.Masking(mask_value=-999)(time_input)

    time_feats = layers.LSTM(39, kernel_regularizer=regularizers.l1_l2(4.7574260503280286e-11, 7.320385408959851e-10),
                            recurrent_regularizer=regularizers.l1_l2(2.1842875952415852e-08, 0.00011142213043058291),
                            bias_regularizer=regularizers.l1_l2(1.7072690483566078e-10, 2.013251254595616e-08))(time_mask)

    temp_vector = layers.concatenate([time_feats, const_input])
    temp_vector = layers.Dense(20, kernel_regularizer=regularizers.l1_l2(0.006484633311629295, 6.028091661391954e-07),
    bias_regularizer=regularizers.l1_l2(2.06541855618525e-07, 4.705910760094412e-05), activation='relu')(temp_vector)

    life_pred = layers.Dense(1)(temp_vector)

    model = Model(inputs=[time_input, const_input], outputs=[life_pred])

    model.compile(loss='huber_loss', optimizer=opt, metrics=metrics)

    return model

def hyperx1_(time_input_shape, const_input_shape):
    
    opt = tf.keras.optimizers.Adam(learning_rate=0.01)
    
    time_input = Input(shape=time_input_shape)
    const_input = Input(shape=const_input_shape)

    time_mask = layers.Masking(mask_value=-999)(time_input)

    time_feats = layers.LSTM(62, kernel_regularizer=regularizers.l1_l2(1.4247136803646754e-08, 1.2262842574273236e-05),
                            recurrent_regularizer=regularizers.l1_l2(1.487018619350522e-09, 2.7363665111579394e-08),
                            bias_regularizer=regularizers.l1_l2(2.87908637902623e-11, 0.004924624226987362))(time_mask)

    temp_vector = layers.concatenate([time_feats, const_input])
    temp_vector = layers.Dense(55, kernel_regularizer=regularizers.l1_l2(1.1948046108045673e-07, 8.909753490549122e-11),
    bias_regularizer=regularizers.l1_l2(2.8103937579904148e-11, 2.713873830523239e-12), activation='relu')(temp_vector)

    life_pred = layers.Dense(1)(temp_vector)

    model = Model(inputs=[time_input, const_input], outputs=[life_pred])

    model.compile(loss='huber_loss', optimizer=opt, metrics=metrics)

    return model

def hyperx2(time_input_shape, const_input_shape):
    
    opt = tf.keras.optimizers.Adam(learning_rate=0.01)
    
    time_input = Input(shape=time_input_shape)
    const_input = Input(shape=const_input_shape)

    time_mask = layers.Masking(mask_value=-999)(time_input)

    time_feats = layers.LSTM(31, kernel_regularizer=regularizers.l1_l2(7.464816091651283e-12, 1.634583093879094e-12),
                            recurrent_regularizer=regularizers.l1_l2(2.692592503128477e-11, 1.7701340393472265e-10),
                            bias_regularizer=regularizers.l1_l2(0.04140672832727432, 0.004530162550508976))(time_mask)

    temp_vector = layers.concatenate([time_feats, const_input])
    temp_vector = layers.Dense(35, kernel_regularizer=regularizers.l1_l2(4.814602405645019e-08, 0.07164408266544342),
    bias_regularizer=regularizers.l1_l2(1.960297035807912e-09, 0.014614270068705082), activation='relu')(temp_vector)

    temp_vector = layers.Dense(63, kernel_regularizer=regularizers.l1_l2(3.396346745510037e-11, 0.056515615433454514),
    bias_regularizer=regularizers.l1_l2(5.976560357723315e-10, 0.0071025812067091465), activation='relu')(temp_vector)
    life_pred = layers.Dense(1)(temp_vector)

    model = Model(inputs=[time_input, const_input], outputs=[life_pred])

    model.compile(loss='huber_loss', optimizer=opt, metrics=metrics)

    return model

def hyperx3(time_input_shape, const_input_shape):
    
    opt = tf.keras.optimizers.Adam(learning_rate=0.01)
    
    time_input = Input(shape=time_input_shape)
    const_input = Input(shape=const_input_shape)

    time_mask = layers.Masking(mask_value=-999)(time_input)

    time_feats = layers.LSTM(45, kernel_regularizer=regularizers.l1_l2(1.733379585699968e-11, 1.6522717487532645e-06),
                            recurrent_regularizer=regularizers.l1_l2(0.0002117684780387208, 0.0011270426912233233),
                            bias_regularizer=regularizers.l1_l2(4.432921738017903e-11, 3.557764109360373e-12))(time_mask)

    temp_vector = layers.concatenate([time_feats, const_input])
    temp_vector = layers.Dense(58, kernel_regularizer=regularizers.l1_l2(3.074768528676941e-07, 0.00011142526636831462),
    bias_regularizer=regularizers.l1_l2(1.9424006847401643e-10, 8.905729452501898e-12), activation='relu')(temp_vector)

    temp_vector = layers.Dense(35, kernel_regularizer=regularizers.l1_l2(4.261859885446029e-06, 2.2821163361830266e-12),
    bias_regularizer=regularizers.l1_l2(8.07635139321583e-11, 0.00013469415716826916), activation='relu')(temp_vector)

    temp_vector = layers.Dense(17, kernel_regularizer=regularizers.l1_l2(0.04279255121946335, 6.580592737392976e-10),
    bias_regularizer=regularizers.l1_l2(2.1102696413227706e-12, 2.1339438549539125e-10), activation='relu')(temp_vector)
    life_pred = layers.Dense(1)(temp_vector)

    model = Model(inputs=[time_input, const_input], outputs=[life_pred])

    model.compile(loss='huber_loss', optimizer=opt, metrics=metrics)

    return model

def hyperx4(time_input_shape, const_input_shape):
    
    opt = tf.keras.optimizers.Adam(learning_rate=0.01)
    
    time_input = Input(shape=time_input_shape)
    const_input = Input(shape=const_input_shape)

    time_mask = layers.Masking(mask_value=-999)(time_input)

    time_feats = layers.LSTM(36, kernel_regularizer=regularizers.l1_l2(1.7547265723782957e-11, 3.2578647335412825e-08),
                            recurrent_regularizer=regularizers.l1_l2(1.0229591680399608e-05, 1.1925305187787671e-10),
                            bias_regularizer=regularizers.l1_l2(5.099541622310966e-11, 3.6223164556759e-07))(time_mask)

    temp_vector = layers.concatenate([time_feats, const_input])
    temp_vector = layers.Dense(25, kernel_regularizer=regularizers.l1_l2(1.556631104904227e-05, 1.0227319080513553e-06),
    bias_regularizer=regularizers.l1_l2(2.4665194331419116e-08, 1.266798818622128e-11), activation='relu')(temp_vector)

    temp_vector = layers.Dense(59, kernel_regularizer=regularizers.l1_l2(0.025503354147076607, 5.681756078956823e-07),
    bias_regularizer=regularizers.l1_l2(0.00012984834029339254, 9.25462445593439e-05), activation='relu')(temp_vector)

    temp_vector = layers.Dense(56, kernel_regularizer=regularizers.l1_l2(5.814993642161426e-07, 0.003231818089261651),
    bias_regularizer=regularizers.l1_l2(2.7001564872897177e-10, 3.178917324930808e-07), activation='relu')(temp_vector)

    temp_vector = layers.Dense(31, kernel_regularizer=regularizers.l1_l2(1.988704134703312e-08, 0.0015827128663659096),
    bias_regularizer=regularizers.l1_l2(8.169342285979653e-11, 1.5344276960149728e-08), activation='relu')(temp_vector)
    life_pred = layers.Dense(1)(temp_vector)

    model = Model(inputs=[time_input, const_input], outputs=[life_pred])

    model.compile(loss='huber_loss', optimizer=opt, metrics=metrics)

    return model

def hyperx5(time_input_shape, const_input_shape):
    
    opt = tf.keras.optimizers.Adam(learning_rate=0.01)
    
    time_input = Input(shape=time_input_shape)
    const_input = Input(shape=const_input_shape)

    time_mask = layers.Masking(mask_value=-999)(time_input)

    time_feats = layers.LSTM(47, kernel_regularizer=regularizers.l1_l2(1.0987118059802015e-07, 1.3356354600091436e-07),
                            recurrent_regularizer=regularizers.l1_l2(5.4749406397380795e-11, 0.003189302049577236),
                            bias_regularizer=regularizers.l1_l2(5.652836879144196e-11, 6.240363290999085e-06))(time_mask)

    temp_vector = layers.concatenate([time_feats, const_input])
	
    temp_vector = layers.Dense(51, kernel_regularizer=regularizers.l1_l2(0.0026039471849799156, 1.1837389402025167e-10),
    bias_regularizer=regularizers.l1_l2(0.00039278302574530244, 1.0068599465284933e-07), activation='relu')(temp_vector)

    temp_vector = layers.Dense(62, kernel_regularizer=regularizers.l1_l2(3.557083230396052e-12, 3.356897815276483e-10),
    bias_regularizer=regularizers.l1_l2(0.002102266764268279, 1.2130651518005298e-11), activation='relu')(temp_vector)

    temp_vector = layers.Dense(18, kernel_regularizer=regularizers.l1_l2(1.2201256671673377e-10, 1.6766227517450716e-08),
    bias_regularizer=regularizers.l1_l2(2.490333717020121e-09, 0.06846927106380463), activation='relu')(temp_vector)

    temp_vector = layers.Dense(31, kernel_regularizer=regularizers.l1_l2(3.7084690873712134e-10, 4.946640075331743e-08),
    bias_regularizer=regularizers.l1_l2(1.668988807068672e-05, 1.3414934301181347e-06), activation='relu')(temp_vector)

    temp_vector = layers.Dense(20, kernel_regularizer=regularizers.l1_l2(0.04130685329437256, 1.3589988509532525e-12),
    bias_regularizer=regularizers.l1_l2(9.511165292852564e-12, 5.6173987104557455e-05), activation='relu')(temp_vector)

    life_pred = layers.Dense(1)(temp_vector)

    model = Model(inputs=[time_input, const_input], outputs=[life_pred])

    model.compile(loss='huber_loss', optimizer=opt, metrics=metrics)

    return model

def hyperx6(time_input_shape, const_input_shape):
    
    opt = tf.keras.optimizers.Adam(learning_rate=0.01)
    
    time_input = Input(shape=time_input_shape)
    const_input = Input(shape=const_input_shape)

    time_mask = layers.Masking(mask_value=-999)(time_input)

    time_feats = layers.LSTM(53, kernel_regularizer=regularizers.l1_l2(1.639660851537883e-08, 6.564826460220274e-10),
                            recurrent_regularizer=regularizers.l1_l2(7.248214917154883e-09, 1.0028735175637848e-08),
                            bias_regularizer=regularizers.l1_l2(7.219664865942832e-08, 5.185681328789826e-10))(time_mask)

    temp_vector = layers.concatenate([time_feats, const_input])
    temp_vector = layers.Dense(43, kernel_regularizer=regularizers.l1_l2(6.492994998552604e-06, 2.4199548533943016e-06),
    bias_regularizer=regularizers.l1_l2(8.011779755179305e-06, 2.3224731599685855e-12), activation='relu')(temp_vector)

    temp_vector = layers.Dense(19, kernel_regularizer=regularizers.l1_l2(3.1554858552496068e-12, 0.0003060656017623842),
    bias_regularizer=regularizers.l1_l2(0.015238162130117416, 0.0004443212819751352), activation='relu')(temp_vector)

    temp_vector = layers.Dense(46, kernel_regularizer=regularizers.l1_l2(6.995002799214944e-09, 1.3810793461743742e-05),
    bias_regularizer=regularizers.l1_l2(0.007231844589114189, 0.023452023044228554), activation='relu')(temp_vector)

    temp_vector = layers.Dense(18, kernel_regularizer=regularizers.l1_l2(0.07418092340230942, 7.462029429916583e-07),
    bias_regularizer=regularizers.l1_l2(0.001984973205253482, 0.00016718055121600628), activation='relu')(temp_vector)

    temp_vector = layers.Dense(57, kernel_regularizer=regularizers.l1_l2(1.6985920048978587e-12, 1.002695214184779e-12),
    bias_regularizer=regularizers.l1_l2(0.006460623815655708, 2.899957326008007e-05), activation='relu')(temp_vector)

    temp_vector = layers.Dense(23, kernel_regularizer=regularizers.l1_l2(0.0011287762317806482, 1.3130986298293124e-09),
    bias_regularizer=regularizers.l1_l2(5.906805600197629e-11, 1.345200689684134e-05), activation='relu')(temp_vector)
    life_pred = layers.Dense(1)(temp_vector)

    model = Model(inputs=[time_input, const_input], outputs=[life_pred])

    model.compile(loss='huber_loss', optimizer=opt, metrics=metrics)

    return model

def hyperx7(time_input_shape, const_input_shape):
    
    opt = tf.keras.optimizers.Adam(learning_rate=0.01)
    
    time_input = Input(shape=time_input_shape)
    const_input = Input(shape=const_input_shape)

    time_mask = layers.Masking(mask_value=-999)(time_input)

    time_feats = layers.LSTM(47, kernel_regularizer=regularizers.l1_l2(2.2205815639608772e-06, 2.0464543393217127e-09),
                            recurrent_regularizer=regularizers.l1_l2(9.459871944272891e-05, 1.0274636176588192e-11),
                            bias_regularizer=regularizers.l1_l2(6.352809123200132e-08, 1.4414563764830746e-08))(time_mask)

    temp_vector = layers.concatenate([time_feats, const_input])
    temp_vector = layers.Dense(34, kernel_regularizer=regularizers.l1_l2(9.68916405441611e-11, 2.4302262424669152e-09),
    bias_regularizer=regularizers.l1_l2(5.103085641167127e-06, 1.5774485291331075e-06), activation='relu')(temp_vector)

    temp_vector = layers.Dense(20, kernel_regularizer=regularizers.l1_l2(1.5152288142417092e-05, 1.6032541613753004e-11),
    bias_regularizer=regularizers.l1_l2(4.939792064978521e-12, 1.1254166487617567e-09), activation='relu')(temp_vector)

    temp_vector = layers.Dense(63, kernel_regularizer=regularizers.l1_l2(0.00010461412603035569, 1.518148495804894e-09),
    bias_regularizer=regularizers.l1_l2(6.441481303909313e-08, 3.189508061041124e-05), activation='relu')(temp_vector)

    temp_vector = layers.Dense(38, kernel_regularizer=regularizers.l1_l2(2.0255408799130237e-06, 0.00012146158405812457),
    bias_regularizer=regularizers.l1_l2(0.00209432328119874, 2.3519590719445205e-09), activation='relu')(temp_vector)

    temp_vector = layers.Dense(36, kernel_regularizer=regularizers.l1_l2(0.0001581552642164752, 2.2142744526831848e-08),
    bias_regularizer=regularizers.l1_l2(4.3074872557724575e-09, 4.965314617799699e-11), activation='relu')(temp_vector)

    temp_vector = layers.Dense(47, kernel_regularizer=regularizers.l1_l2(0.008563496172428131, 0.0087411068379879),
    bias_regularizer=regularizers.l1_l2(5.643043365921585e-09, 1.3280051689434913e-06), activation='relu')(temp_vector)

    temp_vector = layers.Dense(44, kernel_regularizer=regularizers.l1_l2(3.7904328564764e-08, 1.1417525456636213e-05),
    bias_regularizer=regularizers.l1_l2(9.79932295308572e-08, 4.5962065264859575e-09), activation='relu')(temp_vector)
    life_pred = layers.Dense(1)(temp_vector)

    model = Model(inputs=[time_input, const_input], outputs=[life_pred])

    model.compile(loss='huber_loss', optimizer=opt, metrics=metrics)

    return model

def hyperx8(time_input_shape, const_input_shape):
    
    opt = tf.keras.optimizers.Adam(learning_rate=0.01)
    
    time_input = Input(shape=time_input_shape)
    const_input = Input(shape=const_input_shape)

    time_mask = layers.Masking(mask_value=-999)(time_input)

    time_feats = layers.LSTM(33, kernel_regularizer=regularizers.l1_l2(7.516734457091445e-10, 7.981203475893039e-10),
                            recurrent_regularizer=regularizers.l1_l2(6.178939202072797e-06, 8.420851571599997e-08),
                            bias_regularizer=regularizers.l1_l2(0.00016762345330789685, 1.4076730232848167e-12))(time_mask)

    temp_vector = layers.concatenate([time_feats, const_input])
    temp_vector = layers.Dense(50, kernel_regularizer=regularizers.l1_l2(7.267049184633834e-10, 1.5869269787138762e-10),
    bias_regularizer=regularizers.l1_l2(1.3233937383450023e-11, 5.582757780370207e-12), activation='relu')(temp_vector)

    temp_vector = layers.Dense(19, kernel_regularizer=regularizers.l1_l2(1.2825121823920038e-12, 3.4641534512047656e-06),
    bias_regularizer=regularizers.l1_l2(2.0945115863924002e-08, 6.869166099932045e-05), activation='relu')(temp_vector)

    temp_vector = layers.Dense(52, kernel_regularizer=regularizers.l1_l2(8.098719916915798e-08, 1.9067842416120584e-08),
    bias_regularizer=regularizers.l1_l2(2.128665421707865e-08, 1.4616707844083976e-08), activation='relu')(temp_vector)

    temp_vector = layers.Dense(22, kernel_regularizer=regularizers.l1_l2(1.3439505892165471e-05, 0.0022707961034029722),
    bias_regularizer=regularizers.l1_l2(1.344237261946546e-06, 4.49590815965184e-11), activation='relu')(temp_vector)

    temp_vector = layers.Dense(61, kernel_regularizer=regularizers.l1_l2(0.0038339479360729456, 3.250639929319732e-05),
    bias_regularizer=regularizers.l1_l2(5.026569397159619e-06, 9.157481883903529e-08), activation='relu')(temp_vector)

    temp_vector = layers.Dense(26, kernel_regularizer=regularizers.l1_l2(1.0693770313707773e-09, 6.925945967850566e-07),
    bias_regularizer=regularizers.l1_l2(2.9441986626466132e-08, 1.9744565982193762e-09), activation='relu')(temp_vector)

    temp_vector = layers.Dense(52, kernel_regularizer=regularizers.l1_l2(1.8420627523330069e-10, 0.00034649556619115174),
    bias_regularizer=regularizers.l1_l2(2.04879502252453e-10, 0.07205890864133835), activation='relu')(temp_vector)

    temp_vector = layers.Dense(18, kernel_regularizer=regularizers.l1_l2(3.190540155628696e-05, 1.2399593574130563e-09),
    bias_regularizer=regularizers.l1_l2(3.474751320009517e-10, 3.133530117338523e-05), activation='relu')(temp_vector)
    life_pred = layers.Dense(1)(temp_vector)

    model = Model(inputs=[time_input, const_input], outputs=[life_pred])

    model.compile(loss='huber_loss', optimizer=opt, metrics=metrics)

    return model

def hyperx9(time_input_shape, const_input_shape):
    
    opt = tf.keras.optimizers.Adam(learning_rate=0.01)
    
    time_input = Input(shape=time_input_shape)
    const_input = Input(shape=const_input_shape)

    time_mask = layers.Masking(mask_value=-999)(time_input)

    time_feats = layers.LSTM(63, kernel_regularizer=regularizers.l1_l2(6.689730724929177e-08, 1.197145493847529e-08),
                            recurrent_regularizer=regularizers.l1_l2(2.3868587106790073e-09, 3.108501511750461e-10),
                            bias_regularizer=regularizers.l1_l2(4.520706031740929e-09, 1.3847362145824604e-09))(time_mask)

    temp_vector = layers.concatenate([time_feats, const_input])
    temp_vector = layers.Dense(26, kernel_regularizer=regularizers.l1_l2(2.1543689854297554e-06, 7.404589164039521e-10),
    bias_regularizer=regularizers.l1_l2(5.065241975854562e-11, 4.1228086047340184e-05), activation='relu')(temp_vector)

    temp_vector = layers.Dense(48, kernel_regularizer=regularizers.l1_l2(4.61862319223183e-11, 0.0003481749154161662),
    bias_regularizer=regularizers.l1_l2(1.6221539453908917e-06, 7.525304681621492e-05), activation='relu')(temp_vector)

    temp_vector = layers.Dense(17, kernel_regularizer=regularizers.l1_l2(2.502457755326759e-05, 2.754112138347864e-08),
    bias_regularizer=regularizers.l1_l2(0.0001627376623218879, 2.6270166597619493e-10), activation='relu')(temp_vector)

    temp_vector = layers.Dense(35, kernel_regularizer=regularizers.l1_l2(5.166806912870747e-11, 7.132435198009546e-10),
    bias_regularizer=regularizers.l1_l2(2.070432856271509e-06, 3.238491501633689e-07), activation='relu')(temp_vector)

    temp_vector = layers.Dense(18, kernel_regularizer=regularizers.l1_l2(9.531294199405238e-05, 0.004895695019513369),
    bias_regularizer=regularizers.l1_l2(4.5706492812769284e-08, 3.3909152989508584e-05), activation='relu')(temp_vector)

    temp_vector = layers.Dense(55, kernel_regularizer=regularizers.l1_l2(2.141297278379639e-10, 8.220311428885907e-06),
    bias_regularizer=regularizers.l1_l2(3.4076823794748634e-05, 3.212980104194685e-08), activation='relu')(temp_vector)

    temp_vector = layers.Dense(57, kernel_regularizer=regularizers.l1_l2(4.9881131417350844e-05, 1.8672747809783674e-12),
    bias_regularizer=regularizers.l1_l2(7.981689753577825e-10, 3.005044391102274e-07), activation='relu')(temp_vector)

    temp_vector = layers.Dense(29, kernel_regularizer=regularizers.l1_l2(4.4477385330310426e-08, 1.2580151143026796e-08),
    bias_regularizer=regularizers.l1_l2(1.5895026406198554e-09, 0.004968342836946249), activation='relu')(temp_vector)

    temp_vector = layers.Dense(19, kernel_regularizer=regularizers.l1_l2(1.8769953039760034e-12, 2.5475852538647814e-09),
    bias_regularizer=regularizers.l1_l2(7.635407541783934e-07, 3.6560889732362156e-11), activation='relu')(temp_vector)
    life_pred = layers.Dense(1)(temp_vector)

    model = Model(inputs=[time_input, const_input], outputs=[life_pred])

    model.compile(loss='huber_loss', optimizer=opt, metrics=metrics)

    return model

def m_lstm_dev1(time_input_shape, const_input_shape):
    
    opt = tf.keras.optimizers.Adam(learning_rate=0.01)
    
    # Create separate inputs for time series and constants
    time_input = Input(shape=time_input_shape)
    const_input = Input(shape=const_input_shape)

    # Feed time_input through Masking and LSTM layers
    time_mask = layers.Masking(mask_value=-999)(time_input)
    time_feats = layers.LSTM(61, kernel_regularizer=regularizers.l1_l2(0.011444014497101307, 7.300281481548154e-07),
                             recurrent_regularizer=regularizers.l1_l2(0.0002718233736231923, 0.003039814531803131),
                             bias_regularizer=regularizers.l1_l2(2.389202117397682e-12, 3.467680926405592e-06))(time_mask)

    # Concatenate the LSTM output with the constant input
    temp_vector = layers.concatenate([time_feats, const_input])

    temp_vector = layers.Dense(20, kernel_regularizer=regularizers.l1_l2(9.202096407534555e-05, 0.0005918613751418889),
                            bias_regularizer=regularizers.l1_l2(0.036026500165462494, 4.757801070809364e-05), activation='relu')(temp_vector)

    life_pred = layers.Dense(1)(temp_vector)

    # Instantiate model
    model = Model(inputs=[time_input, const_input], outputs=[life_pred])

    # Compile
    model.compile(loss='huber_loss', optimizer=opt, metrics=metrics)

    return model

def m_lstm_dev2(time_input_shape, const_input_shape):
    
    opt = tf.keras.optimizers.Adam(learning_rate=0.01)
    
    # Create separate inputs for time series and constants
    time_input = Input(shape=time_input_shape)
    const_input = Input(shape=const_input_shape)

    # Feed time_input through Masking and LSTM layers
    time_mask = layers.Masking(mask_value=-999)(time_input)
    time_feats = layers.LSTM(57, kernel_regularizer=regularizers.l1_l2(0.0023184821475297213, 0.00045686541125178337),
                             recurrent_regularizer=regularizers.l1_l2(3.342594867561388e-09, 7.84595055591808e-09),
                             bias_regularizer=regularizers.l1_l2(1.0906169300994861e-08, 1.9527479633296707e-10))(time_mask)

    # Concatenate the LSTM output with the constant input
    temp_vector = layers.concatenate([time_feats, const_input])

    temp_vector = layers.Dense(37, kernel_regularizer=regularizers.l1_l2(1.1080036194099918e-11, 0.05474541708827019),
                            bias_regularizer=regularizers.l1_l2(0.0003809732443187386, 1.1690389324983674e-12), activation='relu')(temp_vector)

    temp_vector = layers.Dense(37, kernel_regularizer=regularizers.l1_l2(4.4969530876848296e-10, 4.1496790800010785e-05),
                            bias_regularizer=regularizers.l1_l2(3.500933942746087e-08, 2.290576972541203e-08), activation='relu')(temp_vector)

    life_pred = layers.Dense(1)(temp_vector)

    # Instantiate model
    model = Model(inputs=[time_input, const_input], outputs=[life_pred])

    # Compile
    model.compile(loss='huber_loss', optimizer=opt, metrics=metrics)

    return model

def s_lstm_dev2(time_input_shape, const_input_shape):
    
    opt = tf.keras.optimizers.Adam(learning_rate=0.01)
    
    # Create separate inputs for time series and constants
    time_input = Input(shape=time_input_shape)
    const_input = Input(shape=const_input_shape)

    # Feed time_input through Masking and LSTM layers
    time_mask = layers.Masking(mask_value=-999)(time_input)
    time_feats = layers.LSTM(46, kernel_regularizer=regularizers.l1_l2(3.171006568436496e-08, 1.369990232369389e-10),
                             recurrent_regularizer=regularizers.l1_l2(0.01229761354625225, 6.910993732844872e-08),
                             bias_regularizer=regularizers.l1_l2(2.2401343002798058e-08, 2.5950281853925894e-11))(time_mask)

    # Concatenate the LSTM output with the constant input
    temp_vector = layers.concatenate([time_feats, const_input])

    temp_vector = layers.Dense(38, kernel_regularizer=regularizers.l1_l2(0.024249326437711716, 5.0193196821091135e-11),
                            bias_regularizer=regularizers.l1_l2(3.582382987588062e-06, 9.630478416511323e-06), activation='relu')(temp_vector)

    temp_vector = layers.Dense(20, kernel_regularizer=regularizers.l1_l2(5.417953374831086e-08, 0.0007285480387508869),
                            bias_regularizer=regularizers.l1_l2(1.7092896108933386e-12, 2.514738116587978e-05), activation='relu')(temp_vector)

    life_pred = layers.Dense(1)(temp_vector)

    # Instantiate model
    model = Model(inputs=[time_input, const_input], outputs=[life_pred])

    # Compile
    model.compile(loss='huber_loss', optimizer=opt, metrics=metrics)

    return model

def m_lstm_dev3(time_input_shape, const_input_shape):
    
    opt = tf.keras.optimizers.Adam(learning_rate=0.05)
    
    # Create separate inputs for time series and constants
    time_input = Input(shape=time_input_shape)
    const_input = Input(shape=const_input_shape)

    # Feed time_input through Masking and LSTM layers
    time_mask = layers.Masking(mask_value=-999)(time_input)
    time_feats = layers.LSTM(24, kernel_regularizer=regularizers.l1_l2(0.0022543142549693584, 0.001635525724850595),
                             recurrent_regularizer=regularizers.l1_l2(1.868051685560701e-10, 5.00042611484286e-11),
                             bias_regularizer=regularizers.l1_l2(2.2938243318670892e-11, 2.597095871692545e-09))(time_mask)

    # Concatenate the LSTM output with the constant input
    temp_vector = layers.concatenate([time_feats, const_input])

    temp_vector = layers.Dense(47, kernel_regularizer=regularizers.l1_l2(5.3940404226571204e-11, 1.418772460626272e-10),
                            bias_regularizer=regularizers.l1_l2(2.5730504966264833e-12, 4.376530853278382e-07), activation='relu')(temp_vector)

    life_pred = layers.Dense(1)(temp_vector)

    # Instantiate model
    model = Model(inputs=[time_input, const_input], outputs=[life_pred])

    # Compile
    model.compile(loss='huber_loss', optimizer=opt, metrics=metrics)

    return model

def s_lstm_dev1(time_input_shape, const_input_shape):
    
    opt = tf.keras.optimizers.Adam(learning_rate=0.01)
    
    # Create separate inputs for time series and constants
    time_input = Input(shape=time_input_shape)
    const_input = Input(shape=const_input_shape)

    # Feed time_input through Masking and LSTM layers
    time_mask = layers.Masking(mask_value=-999)(time_input)
    time_feats = layers.LSTM(57, kernel_regularizer=regularizers.l1_l2(0.00010112956078955904, 6.242420624857914e-08),
                             recurrent_regularizer=regularizers.l1_l2(0.009874645620584488, 4.4626954909254835e-11),
                             bias_regularizer=regularizers.l1_l2(9.781341535342047e-12, 0.0049304114654660225))(time_mask)

    # Concatenate the LSTM output with the constant input
    temp_vector = layers.concatenate([time_feats, const_input])

    temp_vector = layers.Dense(54, kernel_regularizer=regularizers.l1_l2(0.010746859014034271, 0.00151325692422688),
                            bias_regularizer=regularizers.l1_l2(1.722696833894588e-05, 2.137088943310328e-11), activation='relu')(temp_vector)

    life_pred = layers.Dense(1)(temp_vector)

    # Instantiate model
    model = Model(inputs=[time_input, const_input], outputs=[life_pred])

    # Compile
    model.compile(loss='huber_loss', optimizer=opt, metrics=metrics)

    return model


def s_lstm_lowN(time_input_shape, const_input_shape):
    
    opt = tf.keras.optimizers.Adam(learning_rate=0.01)
    
    # Create separate inputs for time series and constants
    time_input = Input(shape=time_input_shape)
    const_input = Input(shape=const_input_shape)

    # Feed time_input through Masking and LSTM layers
    time_mask = layers.Masking(mask_value=-999)(time_input)
    time_feats = layers.LSTM(55, kernel_regularizer=regularizers.l1_l2(2.3577105545680155e-11, 1.154892402155383e-06),
                             recurrent_regularizer=regularizers.l1_l2(1.2293748113734182e-05, 6.78100650475244e-06),
                             bias_regularizer=regularizers.l1_l2(0.09824886918067932, 5.833915572850401e-09))(time_mask)

    # Concatenate the LSTM output with the constant input
    temp_vector = layers.concatenate([time_feats, const_input])

    temp_vector = layers.Dense(36, kernel_regularizer=regularizers.l1_l2(0.0004110038571525365, 6.64597066615813e-12),
                            bias_regularizer=regularizers.l1_l2(3.424626626724603e-08, 2.319773739145603e-05), activation='relu')(temp_vector)

    life_pred = layers.Dense(1)(temp_vector)

    # Instantiate model
    model = Model(inputs=[time_input, const_input], outputs=[life_pred])

    # Compile
    model.compile(loss='huber_loss', optimizer=opt, metrics=metrics)

    return model

def s_lstm_lowN2(time_input_shape, const_input_shape):
    
    opt = tf.keras.optimizers.Adam(learning_rate=0.01)
    
    # Create separate inputs for time series and constants
    time_input = Input(shape=time_input_shape)
    const_input = Input(shape=const_input_shape)

    # Feed time_input through Masking and LSTM layers
    time_mask = layers.Masking(mask_value=-999)(time_input)
    time_feats = layers.LSTM(54, kernel_regularizer=regularizers.l1_l2(2.4199548533943016e-06, 3.1554858552496068e-12),
                             recurrent_regularizer=regularizers.l1_l2(2.3224731599685855e-12, 0.015238162130117416),
                             bias_regularizer=regularizers.l1_l2(4.885165189039142e-12, 0.0003060656017623842))(time_mask)

    # Concatenate the LSTM output with the constant input
    temp_vector = layers.concatenate([time_feats, const_input])

    temp_vector = layers.Dense(46, kernel_regularizer=regularizers.l1_l2( 6.995002799214944e-09, 1.3810793461743742e-05),
                            bias_regularizer=regularizers.l1_l2(0.007231844589114189, 0.023452023044228554), activation='relu')(temp_vector)
    temp_vector = layers.Dense(18, kernel_regularizer=regularizers.l1_l2(0.07418092340230942, 7.462029429916583e-07),
                            bias_regularizer=regularizers.l1_l2(0.001984973205253482, 0.00016718055121600628), activation='relu')(temp_vector)

    life_pred = layers.Dense(1)(temp_vector)

    # Instantiate model
    model = Model(inputs=[time_input, const_input], outputs=[life_pred])

    # Compile
    model.compile(loss='huber_loss', optimizer=opt, metrics=metrics)

    return model

def m_lstm_dev22(time_input_shape, const_input_shape):
    
    opt = tf.keras.optimizers.Adam(learning_rate=0.01)
    
    # Create separate inputs for time series and constants
    time_input = Input(shape=time_input_shape)
    const_input = Input(shape=const_input_shape)

    # Feed time_input through Masking and LSTM layers
    time_mask = layers.Masking(mask_value=-999)(time_input)
    time_feats = layers.LSTM(23, kernel_regularizer=regularizers.l1_l2(4.456753854853446e-10, 2.0863347941629806e-12),
                             recurrent_regularizer=regularizers.l1_l2(3.4960050925292308e-06, 0.040089476853609085),
                             bias_regularizer=regularizers.l1_l2(1.7867653190339894e-10, 4.038190581923118e-08))(time_mask)

    # Concatenate the LSTM output with the constant input
    temp_vector = layers.concatenate([time_feats, const_input])

    temp_vector = layers.Dense(34, kernel_regularizer=regularizers.l1_l2(4.0496755254748606e-11, 0.008097505196928978),
                            bias_regularizer=regularizers.l1_l2(1.9550006145202525e-12, 4.176363432861763e-08), activation='relu')(temp_vector)

    temp_vector = layers.Dense(36, kernel_regularizer=regularizers.l1_l2(6.01820538577158e-05, 0.0030067265033721924),
                            bias_regularizer=regularizers.l1_l2(3.972098691629178e-12, 1.8910617516354478e-10), activation='relu')(temp_vector)

    life_pred = layers.Dense(1)(temp_vector)

    # Instantiate model
    model = Model(inputs=[time_input, const_input], outputs=[life_pred])

    # Compile
    model.compile(loss='huber_loss', optimizer=opt, metrics=metrics)

    return model