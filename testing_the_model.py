from keras import layers

H_input_test = H_input[0]
print(H_input_test.shape)
H_input_test = np.expand_dims(H_input_test,0)
print(H_input_test.shape)

H_est_test = H_est[0]
print(H_est_test.shape)
H_est_test = np.expand_dims(H_est_test,0)
print(H_est_test.shape)

y = model.evaluate(x=[H_input_test , H_est_test] , y=out_test, batch_size=1, verbose=1)
layer_name = 'reshape_3'
intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name).output)
intermediate_output = intermediate_layer_model.predict([H_input_test , H_est_test])
print(intermediate_output.shape)
print(H_est_test.shape)
H_est_test = np.squeeze(H_est_test)
print(H_est_test.shape)

intermediate_output = np.squeeze(intermediate_output)
print(intermediate_output.shape)

H_tilde = np.matmul(H_est_test,intermediate_output)
Q , R = np.linalg.qr(H_tilde)
L = np.transpose(np.conj(R))
L_d = np.diag(np.diag(L))
L_inv = np.linalg.inv(L_d)
alpha_1 = np.matmul(Q,L_inv)
alpha = np.matmul(alpha_1,L_d)
V_D_LQR = alpha/np.linalg.norm(alpha,ord='fro')
H_eff = np.matmul(H_tilde,V_D_LQR)
beta = Nt*K
rate = np.sum(np.log10(np.power(np.absolute(H_eff), 2)/beta + 1))
print(rate)