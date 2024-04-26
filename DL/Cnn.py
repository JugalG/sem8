import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D , Dense , MaxPool2D, Flatten
from tensorflow.keras.datasets import mnist

(train_img , train_lbs) , (test_img , test_lbs) = mnist.load_data()


train_img = train_img.reshape(train_img.shape[0],28,28,1).astype('float32')/255.0

test_img = test_img.reshape(test_img.shape[0],28,28,1).astype('float32')/255.0



model = Sequential()
model.add(Conv2D(64,(3,3) , input_shape = (28,28,1) , activation = 'relu'))
model.add(MaxPool2D((2,2)))
model.add(Conv2D(32,(3,3)  , activation = 'relu'))
model.add(MaxPool2D((2,2)))
# model.add(Conv2D(16,(3,3)  , activation = 'relu'))
# model.add(MaxPool2D((2,2)))
model.add(Flatten())
model.add(Dense(64, activation = 'relu'))
model.add(Dense(10, activation = 'softmax'))


model.summary()

print(train_img.shape)
print(train_lbs.shape)

model.compile(optimizer = 'adam' , loss = "sparse_categorical_crossentropy" , metrics = ['accuracy'])

model.fit(train_img , train_lbs , epochs = 5 )

model.evaluate(test_img , test_lbs)