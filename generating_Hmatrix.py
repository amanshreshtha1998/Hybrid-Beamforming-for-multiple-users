import numpy as np

m = 10000 # m: number of H matrices in the input data

H_input = []

for i in range(m):
  
  k = 8  # number of users
  n = 16 # number of antennas at the base station
  p = np.arange(-(n-1)/2,(n+1)/2,1,dtype=float)      
  spat_freq = np.zeros(n, )
  U= np.zeros((n,n),dtype=np.complex_)
  h_nlos1= np.zeros((n,k),dtype=np.complex_)
  h_nlos2= np.zeros((n,k),dtype=np.complex_)
  h= np.zeros((n,k),dtype=np.complex_)


  for ig in range(0,n):
      spat_freq[ig] = ((ig+1-(n+1)/2))
      for num in range(0,n):
          U[num][ig] = np.exp(-1j*2*np.pi*p[num]*(spat_freq[ig]/n))/np.sqrt(n)


  theta_user = -0.5*np.random.random_sample(k)
  mul_angle1 = -0.5*np.random.random_sample(k)
  mul_angle2 = -0.5*np.random.random_sample(k)


  los = (1/np.sqrt(2))*(np.random.randn(k)+1j*np.random.randn(k))
  nlos1=  (np.sqrt(0.1))*(1/np.sqrt(2))*(np.random.randn(k)+1j*np.random.randn(k))
  nlos2 = (np.sqrt(0.1))*(1/np.sqrt(2))*(np.random.randn(k)+1j*np.random.randn(k))


  for ig in range(0,k):
      for ro in range(0,n):
          h_nlos1[ro][ig] = nlos1[ig]*0.5*np.exp(-1j*2*np.pi*p[ro]*mul_angle1[ig])
          h_nlos2[ro][ig] = nlos2[ig]*0.5*np.exp(-1j*2*np.pi*p[ro]*mul_angle2[ig])
          h[ro][ig] = los[ig]*0.5*np.exp(-1j*2*np.pi*p[ro]*theta_user[ig])

  H = np.transpose(np.conjugate(U))@(h+h_nlos1+h_nlos2)


  H_input.append(np.array(np.transpose(H)))
  

H_est = np.array(H_input)
print(H_est.shape)
x = np.concatenate([np.real(H_est), np.imag(H_est)],1)
H_input = np.expand_dims(np.concatenate([np.real(H_est), np.imag(H_est)], 1), 1)
print(H_input.shape)


out = np.array([i+1 for i in range(m)])
out = np.transpose(out)
out = np.reshape(out,[m,1])
print(out.shape)

