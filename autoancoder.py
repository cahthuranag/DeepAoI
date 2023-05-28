# importing libs
import numpy as np
import tensorflow as tf
import keras
from keras.layers import Input, Dense, GaussianNoise,Lambda,Dropout
from keras.models import Model
from keras import regularizers
from tensorflow.keras.layers import BatchNormalization
from keras.optimizers import Adam,SGD
from keras import backend as K
# for reproducing reslut
import numpy as np
import tensorflow as tf

# Set random seed for NumPy
np.random.seed(1)

# Set random seed for TensorFlow
tf.random.set_seed(3)
# defining parameters
# define (n,k) here for (n,k) autoencoder
# n = n_channel 
# k = log2(M)  ==> so for (7,4) autoencoder n_channel = 7 and M = 2^4 = 16 
M = 4
k = np.log2(M)
k = int(k)
n_channel = 2
R = k/n_channel
print ('M:',M,'k:',k,'n:',n_channel)
#generating data of size N
N = 800
label = np.random.randint(M,size=N)
# creating one hot encoded vectors
data = []
for i in label:
    temp = np.zeros(M)
    temp[i] = 1
    data.append(temp)
# checking data shape
data = np.array(data)
print (data.shape)
# checking generated data with it's label
temp_check = [17,23,45,67,89,96,72,250,350]
for i in temp_check:
    print(label[i],data[i])
# defining autoencoder and it's layer
input_signal = Input(shape=(M,))
encoded = Dense(M, activation='relu')(input_signal)
encoded1 = Dense(n_channel, activation='linear')(encoded)
encoded2 = Lambda(lambda x: np.sqrt(n_channel)*K.l2_normalize(x,axis=1))(encoded1)

EbNo_train = 5.01187 #  coverted 7 db of EbNo
encoded3 = GaussianNoise(np.sqrt(1/(2*R*EbNo_train)))(encoded2)

decoded = Dense(M, activation='relu')(encoded3)
decoded1 = Dense(M, activation='softmax')(decoded)
autoencoder = Model(input_signal, decoded1)
adam = Adam(lr=0.01)
autoencoder.compile(optimizer=adam, loss='categorical_crossentropy')
# printing summary of layers and it's trainable parameters 
print (autoencoder.summary())
# traning auto encoder
autoencoder.fit(data, data,
                epochs=45,
                batch_size=32)
# saving keras model
from keras.models import load_model
# if you want to save model then remove below comment
# autoencoder.save('autoencoder_v_best.model')
# # making encoder from full autoencoder
encoder = Model(input_signal, encoded2)
# making decoder from full autoencoder
encoded_input = Input(shape=(n_channel,))

deco = autoencoder.layers[-2](encoded_input)
deco = autoencoder.layers[-1](deco)
decoder = Model(encoded_input, deco)
# generating data for checking BER
# if you're not using t-sne for visulation than set N to 70,000 for better result 
# for t-sne use less N like N = 1500
N = 50000
test_label = np.random.randint(M,size=N)
test_data = []

for i in test_label:
    temp = np.zeros(M)
    temp[i] = 1
    test_data.append(temp)
    
test_data = np.array(test_data)
# checking generated data
temp_test = 6
print (test_data[temp_test][test_label[temp_test]],test_label[temp_test])
def frange(x, y, jump):
  while x < y:
    yield x
    x += jump
# calculating BER
# this is optimized BER function so it can handle large number of N
# previous code has another for loop which was making it slow
#EbNodB_range = list(frange(0))
#ber = [None]*len(EbNodB_range)
#for n in range(0,len(EbNodB_range)):
#EbNo=10.0**(EbNodB_range[n]/10.0)
#SNR=0
EbNo=10.0**(SNR/10.0)
noise_std = np.sqrt(1/(2*R*EbNo))
noise_mean = 0
no_errors = 0
nn = N
noise = noise_std * np.random.randn(nn,n_channel)
encoded_signal = encoder.predict(test_data) 
final_signal = encoded_signal + noise
pred_final_signal =  decoder.predict(final_signal)
pred_output = np.argmax(pred_final_signal,axis=1)
no_errors = (pred_output != test_label)
no_errors =  no_errors.astype(int).sum()
ber= no_errors / nn 
print ('SNR:',SNR,'BER:',ber)
    # use below line for generating matlab like matrix which can be copy and paste for plotting ber graph in matlab
    #print(ber[n], " ",end='')            