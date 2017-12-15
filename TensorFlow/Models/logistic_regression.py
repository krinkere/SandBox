# Import all libraries
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from matplotlib import style
import matplotlib
import tensorflow as tf

# Matplotlib Config
matplotlib.rcParams['figure.figsize'] = (10.0, 8.0)
style.use('ggplot')

# Create Dataset
x, y_ = make_classification(150, n_features=2, n_redundant=0)
# Plot the dataset
plt.scatter(x[:,0], x[:,1], c=y_, cmap=plt.cm.coolwarm)
plt.show()

y = y_.reshape((150, 1))


# Function to plot decision boundary
def plot_decision_boundary(pred_func, X):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.copper)
    plt.scatter(X[:, 0], X[:, 1], c=y_, cmap=plt.cm.coolwarm)
    plt.show()


# Define Placeholders for X and Y
# None represents the number of training examples.
# But assume there is a feature vector of 50 feature and we have a dataset of 100 samples. Assume we want to train a
#       model two times with different number of samples, say 30 and 40. Here the size of the training set has one
#       dimension fixed (number of features=number of columns) and another dimension (number of rows=number of training
#       samples) of variable size. Setting its size to 30, then we restrict the input to be of size (30, 50) and thus we
#       won`t be able to re-train the model with 40 samples. The same holds for using 40 as number of rows. The solution
#       is to just set the number of columns but leave the number of rows unspecified by setting it to None as follows:
X = tf.placeholder(tf.float32, shape=[None, 2])
Y = tf.placeholder(tf.float32, shape=[None, 1])

# Weights and Biases
# (tf.random_normal - Generate Random number from Normal Distribution)
W = tf.Variable(tf.random_normal([2, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# As a summary, a placeholder is used to store input data that are to be initialized once and used multiple times.
# But Variable is used for trainable parameters that are to be changed multiple times after being initialized.

# Hypothesis
# (tf.matmul(A, B) - Multiply matrices A & B - A * B)
# (tf.sigmoid - Calculate Sigmoid Function)
hypothesis = tf.sigmoid(tf.matmul(X, W) + b)

# Cost Function
# (tf.reduce_mean - Equivalent to np.mean)
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) *
                       tf.log(1 - hypothesis))

# Optimize Cost Function using Gradient Descent
# (tf.train.GradientDescentOptimizer - Initialize Gradient Descent Optimizer Object)
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

# Prediction and Accuracy
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

# Start Session
sess = tf.Session()

# Initialize Variables
sess.run(tf.global_variables_initializer())

# Train the model
for step in range(10001):
    cost_val, _ = sess.run([cost, train], feed_dict={X: x, Y: y})
    if step % 1000 == 0:
        # Print Cost Function
        print(step, cost_val)

# Accuracy report
h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X: x, Y: y})
print("\nAccuracy: ", a)

# Plot decision boundary
plot_decision_boundary(lambda x: sess.run(predicted, feed_dict={X:x}), x)