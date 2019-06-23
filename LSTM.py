import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axisartist.parasite_axes import HostAxes, ParasiteAxes
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve


def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()

def load_data(file_name, min_mmr=0, gamemode=22):
    raw = np.array(pd.read_csv(file_name))
    raw = raw[raw[:, 4] > min_mmr]
    raw = raw[raw[:, 6] == gamemode]

    data = np.zeros((raw.shape[0], 10), dtype=int)
    label = np.array(raw[:, 1], dtype=int)

    for i, rec in enumerate(raw):
        # radiant team picks
        radiant = list(map(int, rec[2].split(',')))
        dire = list(map(int, rec[3].split(',')))
        # radiant.sort()
        # dire.sort()
        data[i] = radiant[:1] + dire[:2] + radiant[1:3] + dire[2:4] + radiant[3:] + dire[4:]

    return data, label


def load_data2(file_name):
    data = np.array(pd.read_csv(file_name))
    train_data, test_data, _, _ = train_test_split(data, np.zeros(data.shape[0]), train_size=0.9, random_state=42)

    test_label = test_data[:, 0]
    test_data = test_data[:, 1:]
    test_data = np.where(test_data == 1)[1].reshape(test_data.shape[0], 10)
    test_data = np.where(test_data < 115, test_data, test_data - 115)
    train_label = train_data[:, 0]
    train_data = train_data[:, 1:]
    train_data = np.where(train_data == 1)[1].reshape(train_data.shape[0], 10)
    train_data = np.where(train_data < 115, train_data, train_data - 115)

    return train_data, train_label, test_data, test_label


# test_data, test_label = load_data("706e_test_dataset.csv", min_mmr=0)
# train_data, train_label = load_data("706e_train_dataset.csv", min_mmr=0)
# # test_data = test_data[:, :, None]
# print(test_data.shape)
# print(test_label.shape)
train_data, train_label, test_data, test_label = load_data2("data_filtered_5.csv")
print(test_data)

model = keras.Sequential([
    tf.keras.layers.Embedding(115, 64, input_length=10),
    # tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    # tf.keras.layers.LSTM(64, dropout=0.2, recurrent_dropout=0.2),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    keras.layers.Dense(2, activation='softmax')
])

model.compile(optimizer=keras.optimizers.Adam(0.01),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

epochs = 5
loss = []
acc = []
test_loss = []
test_acc = []
train_time = 0.
for i in range(epochs):
    print("epochs: ", i)
    start = time.time()
    history = model.fit(train_data, train_label, batch_size=256)
    train_time += (time.time() - start)
    t_loss, t_acc = model.evaluate(test_data, test_label)
    test_acc.append(t_acc)
    test_loss.append(t_loss)
    loss.append(history.history["loss"])
    acc.append(history.history["acc"])
    print("loss:" + str(history.history["loss"]))
    print("acc:" + str(history.history["acc"]))
    print("test_acc:" + str(test_acc))
#     if i % 5 == 0:
#         model.save("discriminator")
#
model.save("LSTM.h5")

y_pred = model.predict(test_data)
y_pred = y_pred[:, 1]
print(test_label.shape, y_pred.shape)
roc = roc_auc_score(test_label, y_pred)
fpr, tpr, thresholds = roc_curve(test_label, y_pred)
print('\rroc-auc: %s' % (str(round(roc, 4))), end=100 * ' ' + '\n')
plot_roc_curve(fpr, tpr)

print('Test accuracy:', test_acc)
print('time:', train_time)

# done model evaluation, paint the figure out
fig = plt.figure()
host = HostAxes(fig, [0.15, 0.1, 0.65, 0.8])
par1 = ParasiteAxes(host, sharex=host)

host.parasites.append(par1)

host.axis["right"].set_visible(False)
par1.axis["right"].set_visible(True)

par1.axis["right"].major_ticklabels.set_visible(True)
par1.axis["right"].label.set_visible(True)

fig.add_axes(host)

# host.set_xlim(0, 2)
# host.set_ylim(0, 2)

host.set_ylabel("loss")
host.set_xlabel("epoch(accuracy = %.4f)" % (test_acc[-1]))
par1.set_ylabel("acc")

p1, = host.plot(loss, label="loss")
p2, = host.plot(test_loss, label="test_loss")
p3, = par1.plot(acc, label="acc")
p4, = par1.plot(test_acc, label="test acc")

host.legend()

host.axis["left"].label.set_color(p1.get_color())
par1.axis["right"].label.set_color(p2.get_color())
fig.savefig("loss.png")
plt.show()
