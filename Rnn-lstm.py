from tensorflow.keras.datasets import imdb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding , Dense , LSTM
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


# Get the word index
word_index = imdb.get_word_index()

# Calculate vocabulary size
vocab_size = len(word_index) + 1  # Add 1 for padding token

print("Vocabulary Size:", vocab_size)




mx_len = 200
max_words = 10000
embending_dims = 128
(X_train , y_train), (X_test, y_test) = imdb.load_data(num_words = max_words)




print(X_train.shape)


X_train = pad_sequences(X_train, maxlen=mx_len)
X_test = pad_sequences(X_test, maxlen=mx_len)


model = Sequential()
model.add(Embedding(input_dim = max_words , output_dim = embending_dims , input_length = mx_len))
model.add(LSTM(128))#dropout=0.2, recurrent_dropout=0.2)
model.add(Dense(1 , activation = 'sigmoid'))


model.summary()


model.compile(optimizer = 'adam' , loss = 'binary_crossentropy' , metrics = ['accuracy'])

model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2)
input_sentence = "This movie was fantastic! The acting was superb"

tokenizer = Tokenizer()
tokenizer.fit_on_texts([input_sentence])

input_sequence = tokenizer.texts_to_sequences([input_sentence])
input_sequence_padded = pad_sequences(input_sequence, maxlen=mx_len)
# Make predictions
predictions = model.predict(input_sequence_padded)

# Assuming binary classification (positive/negative sentiment)
if predictions[0][0] >= 0.5:
    print(f"The sentence '{input_sentence}' is positive.")
else:
    print(f"The sentence '{input_sentence}' is negative.")





journal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error
import yfinance as yf

# Download historical stock price data for a specific ticker symbol
dt = yf.download('AAPL', start='2021-01-01', end='2022-01-01')

# Save the data to a CSV file
dt.to_csv('stock_prices.csv')

# Function to create dataset with lookback
def create_dataset(dataset, lookback=1):
    X, y = [], []
    for i in range(len(dataset)-lookback-1):
        X.append(dataset[i:(i+lookback), 0])
        y.append(dataset[i + lookback, 0])
    return np.array(X), np.array(y)

lookback = 20  # Number of timesteps to look back
X_train, y_train = create_dataset(train_data, lookback)
X_test, y_test = create_dataset(test_data, lookback)

# Reshape input data to be 3D [samples, timesteps, features]
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Define the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32)

# Make predictions
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

# Inverse transform predictions to original scale
train_predictions = scaler.inverse_transform(train_predictions)
y_train = scaler.inverse_transform([y_train])
test_predictions = scaler.inverse_transform(test_predictions)
y_test = scaler.inverse_transform([y_test])

# Calculate root mean squared error
train_score = np.sqrt(mean_squared_error(y_train[0], train_predictions[:,0]))
print('Train RMSE: %.2f' % (train_score))
test_score = np.sqrt(mean_squared_error(y_test[0], test_predictions[:,0]))
print('Test RMSE: %.2f' % (test_score))

print(data.shape)
print(train_predictions.shape)
print(test_predictions.shape)

print(len(train_predictions) + 2 * lookback)
print(len(data['Close']))

plt.plot(range(len(train_predictions) + lookback, len(test_predictions) + len(train_predictions) + lookback), test_predictions, label='Test Predictions')
plt.plot(data['Close'], label='Original Data')
plt.plot(range(lookback, len(train_predictions) + lookback), train_predictions, label='Train Predictions')
plt.plot(range(len(train_predictions) + lookback, len(test_predictions) + len(train_predictions) + lookback), test_predictions, label='Test Predictions')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
