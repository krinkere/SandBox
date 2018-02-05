"""
install h2o - http://h2o-release.s3.amazonaws.com/h2o/rel-wheeler/4/index.html
"""
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from pylab import rcParams
import h2o
from h2o.estimators.deeplearning import H2OAutoEncoderEstimator

rcParams['figure.figsize'] = 20, 12

# Start H2O on your local machine
h2o.init()

ecg_data = h2o.import_file("http://h2o-public-test-data.s3.amazonaws.com/smalldata/anomaly/ecg_discord_test.csv")

print(ecg_data.shape)

train_ecg = ecg_data[:20:, :]
test_ecg = ecg_data[:23, :]


def plot_stacked_time_series(df, title):
    stacked = df.stack()
    stacked = stacked.reset_index()
    total = [data[0].values for name, data in stacked.groupby('level_0')]
    pd.DataFrame({idx: pos for idx,pos in enumerate(total)}).plot(title=title)
    plt.legend(bbox_to_anchor=(1.05, 1))
    plt.show()

plot_stacked_time_series(ecg_data.as_data_frame(), "ECG data set")


def plot_bidimensional(model, test, recon_error, layer, title):

    bidimensional_data = model.deepfeatures(test, layer).cbind(recon_error).as_data_frame()

    cmap = cm.get_cmap('Spectral')

    fig, ax = plt.subplots()
    bidimensional_data.plot(kind='scatter',
                            x='DF.L{}.C1'.format(layer+1),
                            y='DF.L{}.C2'.format(layer+1),
                            s=500,
                            c='Reconstruction.MSE',
                            title=title,
                            ax=ax,
                            colormap=cmap)
    layer_column = 'DF.L{}.C'.format(layer + 1)
    columns = [layer_column + '1', layer_column + '2']
    for k, v in bidimensional_data[columns].iterrows():
        ax.annotate(k, v, size=20, verticalalignment='bottom', horizontalalignment='left')
    fig.canvas.draw()
    plt.show()

seed = 13

"""
Encoding
Let’s sketch out an example encoder:
784 (input) ----> 1000 ----> 500 ----> 250 ----> 100 -----> 30


"""

model = H2OAutoEncoderEstimator(
    activation="Tanh",
    hidden=[50, 20, 2, 20, 50],
    epochs=100,
    seed=seed,
    reproducible=True)

model.train(
    x=train_ecg.names,
    training_frame=train_ecg
)

recon_error = model.anomaly(test_ecg)
plot_bidimensional(model, test_ecg, recon_error, 2, "2D representation of data points seed {}".format(seed))

print(model)

plt.figure()
df = recon_error.as_data_frame(True)
df["sample_index"] = df.index
df.plot(kind="scatter", x="sample_index", y="Reconstruction.MSE", title="reconstruction error", s = 500)
plt.show()
