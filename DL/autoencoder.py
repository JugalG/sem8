import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense , Input

def autoencoder_make(input_dims , encoding_dims):
  encoded_seq = Input(input_dims)
  encoded = Dense(128 , activation = 'relu')(encoded_seq)
  encoded = Dense(encoding_dims , activation = "relu")(encoded)

  decode = Dense(128 , activation = 'relu')(encoded)
  decode = Dense(input_dims[0] , activation = 'sigmoid')(decode)

  autoencoder = Model(encoded_seq , decode)

  autoencoder.compile(optimizer = 'adam' , loss = 'binary_crossentropy' , metrics = ['accuracy'])

  return autoencoder

input_dims = (784,)
encode_dim = 64


autoencoder =  autoencoder_make(input_dims ,encode_dim )

from tensorflow.keras.datasets import fashion_mnist

(train_img , train_lbs) , (test_img , test_lbs) = fashion_mnist.load_data()

train_img = train_img.reshape(train_img.shape[0],np.prod(train_img.shape[1:])).astype('float32')/255.0

test_img = test_img.reshape(test_img.shape[0],np.prod(test_img.shape[1:])).astype('float32')/255.0


history = autoencoder.fit(train_img,train_img ,epochs = 5 , validation_data = (test_img,test_img))

decoded_imgs = autoencoder.predict(test_img)


import matplotlib.pyplot as plt

testing = test_img[0:10]
decoded_test = decoded_imgs[0:10]
print(len(testing))
plt.figure(figsize = (20,3))
for i in range(10):
  plt.subplot(2,10 , i+1)
  plt.imshow(testing[i].reshape(28,28))
plt.show()

plt.figure(figsize = (20,3))
for i in range(10):
  plt.subplot(2,10 , i+1)
  plt.imshow(decoded_test[i].reshape(28, 28))
plt.show()