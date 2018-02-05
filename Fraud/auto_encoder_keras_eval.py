from keras.models import load_model
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import (confusion_matrix, precision_recall_curve, auc, roc_curve, cohen_kappa_score)
from Fraud.auto_encoder_keras_globals import LABELS

# load model along with test data and classifications
auto_encoder = load_model('models/model.h5')
X_test = pickle.load(open("models/test_data.pkl", "rb"))
y_test = pickle.load(open("models/test_data_classification.pkl", "rb"))

# predict values given test data
predictions = auto_encoder.predict(X_test)
# calculate your traditional squared error
mse = np.mean(np.power(X_test - predictions, 2), axis=1)
# dataframe containing actual true classification and reconstruction error
predictions_df = pd.DataFrame({'reconstruction_error': mse, 'true_class': y_test})
print(predictions_df.describe())

fig = plt.figure()
ax = fig.add_subplot(111)
normal_error_df = predictions_df[(predictions_df['true_class'] == 0)]
_ = ax.hist(normal_error_df.reconstruction_error.values, bins=10)
plt.xlabel('reconstruction error')
plt.ylabel('count')
plt.title('No Fraud')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)
fraud_error_df = predictions_df[predictions_df['true_class'] == 1]
_ = ax.hist(fraud_error_df.reconstruction_error.values, bins=10)
plt.xlabel('reconstruction error')
plt.ylabel('count')
plt.title('Fraud')
plt.show()

# ROC curves are very useful tool for understanding the performance of binary classifiers.
# However, our case is a bit out of the ordinary. We have a very imbalanced dataset.
# Nonetheless, let's have a look at our ROC curve:
# The ROC curve plots the true positive rate versus the false positive rate, over different threshold values.
# Basically, we want the blue line to be as close as possible to the upper left corner.
# While our results look pretty good, we have to keep in mind of the nature of our dataset.
# ROC doesn't look very useful for us.
fpr, tpr, thresholds = roc_curve(predictions_df.true_class, predictions_df.reconstruction_error)
roc_auc = auc(fpr, tpr)
plt.title('Receiver Operating Characteristic (not useful for very imbalanced dataset)')
plt.plot(fpr, tpr, label='AUC = %0.4f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([-0.001, 1])
plt.ylim([0, 1.001])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

# recall & precision
precision, recall, th = precision_recall_curve(predictions_df.true_class, predictions_df.reconstruction_error)
plt.plot(recall, precision, 'b', label='Precision-Recall curve')
plt.title('Recall vs Precision')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.show()

plt.plot(th, precision[1:], 'b', label='Threshold-Precision curve')
plt.title('Precision for different Reconstruction error values')
plt.xlabel('Reconstruction error')
plt.ylabel('Precision')
plt.show()

plt.plot(th, recall[1:], 'b', label='Threshold-Recall curve')
plt.title('Recall for different Reconstruction error values')
plt.xlabel('Reconstruction error')
plt.ylabel('Recall')
plt.show()

# visualize test data. anything over certain threshold (indicated in red line) is an outlier
threshold = 2.9
print('chosen initial value for threshold', threshold)

groups = predictions_df.groupby('true_class')
fig, ax = plt.subplots()
for name, group in groups:
    ax.plot(group.index, group.reconstruction_error, marker='o', ms=3.5, linestyle='',
            label="Fraud" if name == 1 else "Normal")
ax.hlines(threshold, ax.get_xlim()[0], ax.get_xlim()[1], colors="r", zorder=100, label='Threshold')
ax.legend()
plt.title("Visualization of classification: Fraud or Not based on threshold limit of %s" % str(threshold))
plt.ylabel("Reconstruction error")
plt.xlabel("Data point index")
plt.show()

# display same result via confusion matrix
y_pred = [1 if e > threshold else 0 for e in predictions_df.reconstruction_error.values]
conf_matrix = confusion_matrix(predictions_df.true_class, y_pred)

plt.figure(figsize=(12, 12))
sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
plt.title("Confusion matrix")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()

# evaluate via Kappa
print(conf_matrix)
print('Accuracy: ',
      str(np.round(100 * float((conf_matrix[0][0]+conf_matrix[1][1]) / float(conf_matrix[0][0] + conf_matrix[1][1] + conf_matrix[1][0] + conf_matrix[0][1])), 2))+'%')
print('Cohen Kappa for 2.9: ', str(np.round(cohen_kappa_score(predictions_df.true_class, y_pred), 3)))
print('Recall: ' + str(np.round(100*float((conf_matrix[1][1]))/float((conf_matrix[1][0]+conf_matrix[1][1])), 2))+'%')

threshold = 4.5
y_pred = [1 if e > threshold else 0 for e in predictions_df.reconstruction_error.values]
print('Cohen Kappa for 4.5: ', str(np.round(cohen_kappa_score(predictions_df.true_class, y_pred), 3)))

max_kappa = 0
selected_threshold = -1
predictions_df.sort_values(by=['reconstruction_error'], inplace=True)
min_threshold = predictions_df.reconstruction_error.values[0]
max_threshold = predictions_df.reconstruction_error.values[-1]
print(min_threshold)
print(max_threshold)

threshold = min_threshold
while threshold < max_threshold:
    y_pred = [1 if e > threshold else 0 for e in predictions_df.reconstruction_error.values]
    kappa = np.round(cohen_kappa_score(predictions_df.true_class, y_pred), 3)
    if kappa > max_kappa:
        max_kappa = kappa
        selected_threshold = threshold
    print('Cohen Kappa for', threshold, "is", str(kappa))
    threshold += 10

print("threshold of", selected_threshold, "yielded best Cohen Kappa value of", max_kappa)

###
### duplicate
###

# visualize test data. anything over certain threshold (indicated in red line) is an outlier
threshold = selected_threshold

groups = predictions_df.groupby('true_class')
fig, ax = plt.subplots()
for name, group in groups:
    ax.plot(group.index, group.reconstruction_error, marker='o', ms=3.5, linestyle='',
            label= "Fraud" if name == 1 else "Normal")
ax.hlines(threshold, ax.get_xlim()[0], ax.get_xlim()[1], colors="r", zorder=100, label='Threshold')
ax.legend()
plt.title("Visualization of classification: Fraud or Not based on threshold limit of %s" % str(threshold))
plt.ylabel("Reconstruction error")
plt.xlabel("Data point index")
plt.show()

# display same result via confusion matrix
y_pred = [1 if e > threshold else 0 for e in predictions_df.reconstruction_error.values]
conf_matrix = confusion_matrix(predictions_df.true_class, y_pred)

plt.figure(figsize=(12, 12))
sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
plt.title("Confusion matrix")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()

conf_matrix = confusion_matrix(predictions_df.true_class, y_pred)
print(conf_matrix)
print('Accuracy: ',
      str(np.round(100 * float((conf_matrix[0][0]+conf_matrix[1][1]) / float(conf_matrix[0][0] + conf_matrix[1][1] + conf_matrix[1][0] + conf_matrix[0][1])), 2))+'%')
print('Cohen Kappa for', str(threshold), ': ', str(np.round(cohen_kappa_score(predictions_df.true_class, y_pred), 3)))
print('Recall: ' + str(np.round(100*float((conf_matrix[1][1]))/float((conf_matrix[1][0]+conf_matrix[1][1])), 2))+'%')

