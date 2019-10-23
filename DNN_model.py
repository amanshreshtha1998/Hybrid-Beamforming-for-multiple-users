import numpy as np
from tensorflow.python.keras.layers import *


Nt = 16
mul = Nt*k;


# The DNN model

input_channel_matrix_1 = Input(name='input_channel_matrix', shape=(H_input.shape[1:4]), dtype=tf.float32)

input_channel_matrix_2 = Input(name='input_channel_matrix_2', shape=(H_est.shape[1:3]), dtype = tf.complex64)

temp = Flatten()(input_channel_matrix_1)

temp = BatchNormalization()(temp)

temp = Dense(1024, activation='relu')(temp)

temp = BatchNormalization()(temp)

temp = Dense(512, activation='relu')(temp)

phase = Dense(mul)(temp)

V_a = Lambda(trans_Va, dtype=tf.complex64, output_shape=(mul,))(phase)

phase2 = Reshape((Nt,K))(V_a)

diff = Lambda(diff_func, dtype=tf.float32, output_shape=(1,))([input_channel_matrix_2 , phase2])

model = Model(inputs=[ input_channel_matrix_1, input_channel_matrix_2 ], outputs = diff )
# the y_pred is the actual rate, thus the loss is y_pred, without labels

model.compile(optimizer='adam', loss=lambda y_true, y_pred: y_pred)
model.summary()


reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=20, min_lr=0.00005)
checkpoint = callbacks.ModelCheckpoint('./temp_trained.h5', monitor='val_loss',
                                       verbose=0, save_best_only=True, mode='min', save_weights_only=True)
model.fit(x=[H_input , H_est], y=out, batch_size=1,  ## for our problem, we should keep batch size as 1 
          epochs=1000, verbose=2, validation_split=0.1, callbacks=[reduce_lr, checkpoint])



