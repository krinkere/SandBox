import nltk
import numpy as np
import pickle
import json
import sklearn
import random
from keras.models import load_model
from TensorFlow.Chatbot.viaKeras_constants import *
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

ERROR_THRESHOLD = 0.25

# import our chat-bot intents file
with open(DATA_FILE) as json_data:
    intents = json.load(json_data)

# restore all of our data structures
data = pickle.load(open(TRAINING_DATA_FILE, "rb"))
words = data['words']
classes = data['classes']
train_x = data['train_x']
train_y = data['train_y']

# load our saved model
model = load_model(MODEL_OUT_FILE)

# create a data structure to hold user context
context = {}


def clean_up_sentence(sentence):
    # tokenize the pattern
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = preserve_special_words(sentence_words)
    # stem each word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words


# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    ''' For whatever reason this stopped working when trying to match 102(A)(1)... '''
    # # bag of words
    # cv = sklearn.feature_extraction.text.CountVectorizer(vocabulary=words)
    # bag = cv.fit_transform([" ".join(str(x) for x in sentence_words)]).toarray()[0]
    ''' END '''

    # bag of words
    bag = [0]*len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    bag = np.array(bag)

    return bag


def classify(sentence):
    # generate probabilities from the model
    ''' For whatever reason this stopped working when trying to match 102(A)(1)... '''
    # sentence_vector = np.array([bow(sentence).tolist()])
    # results = model.predict(sentence_vector)[0]
    ''' END '''
    sentence_vector = np.array([bow(sentence)])
    results = model.predict(sentence_vector)[0]
    # filter out predictions below a threshold
    results = [[i, r] for i, r in enumerate(results) if r > ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((classes[r[0]], r[1]))
    # return tuple of intent and probability
    return return_list


def response(sentence, userID='123', show_details=False):
    results = classify(sentence)
    # if we have a classification then find the matching intent tag
    if results:
        top_choice = results[0][0]
        for i in intents['intents']:
            # find a tag matching the first result
            if i['tag'] == top_choice:
                # set context for this intent if necessary
                if 'context_set' in i:
                    if show_details:
                        print('context:', i['context_set'])
                    context[userID] = i['context_set']

                # check if this intent is contextual and applies to this user's conversation
                if not 'context_filter' in i or \
                        (userID in context and 'context_filter' in i and i['context_filter'] == context[userID]):
                    if show_details:
                        print('tag:', i['tag'])
                    # a random response from the intent
                    return print(random.choice(i['responses']))


def get_context():
    return context


def preserve_special_words(tokens):
    for index, token in enumerate(tokens):
        combined_token = token
        longest_matched_sub_index = -1
        for sub_index, sub_token in enumerate(tokens[index+1:]):
            combined_token += sub_token
            if combined_token in USPTO_RESERVED_WORDS:
                longest_matched_sub_index = index + sub_index + 1

        if longest_matched_sub_index != -1:
            left = tokens[:index]
            right = tokens[longest_matched_sub_index+1:]
            joined = [''.join(tokens[index:longest_matched_sub_index+1])]
            tokens = left + joined + right

    return tokens



