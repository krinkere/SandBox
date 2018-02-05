import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import rcParams
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Input, Dense
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import regularizers
from sklearn.preprocessing import StandardScaler
from Fraud.auto_encoder_keras_globals import LABELS

sns.set(style='whitegrid', palette='muted', font_scale=1.5)
rcParams['figure.figsize'] = 14, 8
RANDOM_SEED = 42

# load the data
df = pd.read_csv("data/creditcard.csv")

# exploration
print("Data size:", df.shape[0])
print("Number of features:", df.shape[1])
print("Any missing values?", df.isnull().values.any())

count_classes = pd.value_counts(df['Class'], sort=True)
count_classes.plot(kind='bar', rot=0)
plt.title("Transaction class distribution")
plt.xticks(range(2), LABELS)
plt.xlabel("Class")
plt.ylabel("Frequency")
plt.show()

df_frauds = df[df.Class == 1]
df_normal = df[df.Class == 0]
print("Number of frauds transactions:", df_frauds.shape[0])
print("Number of normal transactions:", df_normal.shape[0])
print("Fraud stats")
print(df_frauds.Amount.describe())
print("Normal stats")
print(df_normal.Amount.describe())

f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
f.suptitle('Amount per transaction by class')
bins = 50
ax1.hist(df_frauds.Amount, bins = bins)
ax1.set_title('Fraud')
ax2.hist(df_normal.Amount, bins = bins)
ax2.set_title('Normal')
plt.xlabel('Amount ($)')
plt.ylabel('Number of Transactions')
plt.xlim((0, 20000))
plt.yscale('log', nonposy='clip')
plt.show()

f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
f.suptitle('Time of transaction vs Amount by class')
ax1.scatter(df_frauds.Time, df_frauds.Amount)
ax1.set_title('Fraud')
ax2.scatter(df_normal.Time, df_normal.Amount)
ax2.set_title('Normal')
plt.xlabel('Time (in Seconds)')
plt.ylabel('Amount')
plt.show()

# remove time since it did not prove to be useful
data = df.drop(['Time'], axis=1)

# it was seen that amount for normal transactions is several magnitudes higher, so we need to scale it to be -1 to 1
print("old data")
print(data.Amount.head(3))
data['Amount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1, 1))
print("new data")
print(data.Amount.head(3))

# once again the idea of using autocoders for fraud detection is to train it against clean data, i.e. no fraud
# come up with a model that would reconstruct the normal data
# now apply it to a random set. If recontstuction error is high, then it is an outlier.
X_train, X_test = train_test_split(data, test_size=0.2, random_state=RANDOM_SEED)
X_train = X_train[X_train.Class == 0]
X_train = X_train.drop(['Class'], axis=1)
y_test = X_test['Class']
X_test = X_test.drop(['Class'], axis=1)
X_train = X_train.values
X_test = X_test.values
# save for later when we to evaluate the model
pickle.dump(X_test, open("models/test_data.pkl", "wb"))
pickle.dump(y_test, open("models/test_data_classification.pkl", "wb"))
# define a model with 2 encoders/decoders
input_dim = X_train.shape[1]
print(input_dim)
encoding_dim = 14
input_layer = Input(shape=(input_dim, ))
encoder = Dense(encoding_dim,
                activation="tanh",
                activity_regularizer=regularizers.l1(10e-5))(input_layer)
encoder = Dense(int(encoding_dim / 2), activation="relu")(encoder)
decoder = Dense(int(encoding_dim / 2), activation='tanh')(encoder)
decoder = Dense(input_dim, activation='relu')(decoder)
auto_encoder = Model(inputs=input_layer, outputs=decoder)
nb_epoch = 1000
batch_size = 32
auto_encoder.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
# periodically save to the model
check_pointer = ModelCheckpoint(filepath="models/model.h5", verbose=0, save_best_only=True)
# logs for tensor board
tensor_board = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)
# train the model and save history
history = auto_encoder\
                    .fit(X_train, X_train, epochs=nb_epoch, batch_size=batch_size, shuffle=True,
                         validation_data=(X_test, X_test), verbose=1, callbacks=[check_pointer, tensor_board])\
                    .history

plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()