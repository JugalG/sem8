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




journal
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Function to load images from a folder 
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = load_img(os.path.join(folder,filename), target_size=(256, 256))
        img_array = img_to_array(img)
        images.append(img_array)
    return np.array(images)

# Load images from the 'images' folder 
input_images = load_images_from_folder('images')

# Normalize the images
input_images = input_images.astype('float32') / 255.

# Define the autoencoder model 
input_img = Input(shape=(256, 256, 3))

# Encoder
x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# Decoder
x = Conv2D(16, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Train the autoencoder
autoencoder.fit(input_images, input_images, epochs=50, batch_size=16, shuffle=True)

# Save reconstructed images
reconstructed_images = autoencoder.predict(input_images)

# Create folder for reconstructed images if it doesn't exist
if not os.path.exists('reconstructed_images'):
    os.makedirs('reconstructed_images')

# Save reconstructed images
for i, reconstructed_image in enumerate(reconstructed_images):
    plt.imsave(f'reconstructed_images/reconstructed_{i}.png', reconstructed_image)

# Compare original and reconstructed images
n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # Original Images
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(input_images[i])
    plt.title("Original")
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Reconstructed Images
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(reconstructed_images[i])
    plt.title("Reconstructed")
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
