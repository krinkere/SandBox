import nltk
from nltk.corpus import stopwords
import string
import random
import sklearn
import json
from time import time
from keras.models import Sequential
from keras.layers import *
from keras.callbacks import TensorBoard
from TensorFlow.Chatbot.viaKeras_constants import *
from TensorFlow.Chatbot.viaKeras_utils import preserve_special_words
import pickle
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()


# import our chat-bot intents file
with open(DATA_FILE) as json_data:
    intents = json.load(json_data)

words = []
classes = []
documents = []
# ignore_words = ['?']
ignore_words = stopwords.words('english') + list(string.punctuation)

# loop through each sentence in our intents patterns
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # tokenize each word in the sentence
        w = nltk.word_tokenize(pattern)
        w = [word.lower() for word in w]
        w = preserve_special_words(w)
        # add to our words list
        words.extend(w)
        # add to documents in our corpus
        documents.append((w, intent['tag']))
        # add to our classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# stem each word and remove duplicates
words = [stemmer.stem(w) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

# remove duplicates for each class
classes = sorted(list(set(classes)))

print(len(documents), "documents")
print(len(classes), "classes", classes)
print(len(words), "unique stemmed words", words)

# create our training data
training = []

# create an empty array for our output
output_empty = [0] * len(classes)

# training set, bag of words for each sentence
for doc in documents:
    # list of tokenized words for the pattern
    pattern_words = doc[0]
    # stem each word
    pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]

    bag = []
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    ''' For whatever reason this stopped working when trying to match 102(A)(1)... '''
    # cv = sklearn.feature_extraction.text.CountVectorizer(vocabulary=words)
    # bag = cv.fit_transform([" ".join(str(x) for x in pattern_words)]).toarray().tolist()

    # cv = sklearn.feature_extraction.text.CountVectorizer(vocabulary=classes)
    # output_row = cv.fit_transform([doc[1]]).toarray().tolist()

    # training.append([bag[0], output_row[0]])
    ''' END '''

    # output is a '0' for each tag and '1' for current tag
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

# shuffle our features and turn into np.array
random.shuffle(training)
training = np.array(training)

# create train and test lists
train_x = list(training[:, 0])
train_y = list(training[:, 1])

# define tensorboard
# tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

# build model
model = Sequential()
tmp_test = len(train_x[0])
model.add(Dense(50, input_shape=(tmp_test,), activation='relu', name='Input_Layer'))
model.add(Dense(100, activation='relu', name="Hidden_Layer_1"))
model.add(Dense(50, activation='relu', name="Hidden_Layer_2"))
model.add(Dense(len(train_y[0]), activation='softmax', name="Output_Layer"))
print(model.summary())
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# Train the model3
# model.fit(train_x, train_y, epochs=1000, batch_size=8, verbose=1, callbacks=[tensorboard])
model.fit(train_x, train_y, epochs=1000, batch_size=8, verbose=1)
# Save the model to disk
model.save(MODEL_OUT_FILE)
print("Model was saved to disk.")

# save all of our data structures
pickle.dump({'words': words, 'classes': classes, 'train_x': train_x, 'train_y': train_y}, open(TRAINING_DATA_FILE, "wb"))
