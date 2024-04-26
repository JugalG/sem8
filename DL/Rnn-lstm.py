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

