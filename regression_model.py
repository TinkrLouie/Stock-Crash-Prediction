import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Model
from keras.optimizers import Adam
from keras.losses import MeanSquaredError
from keras.layers import *
import matplotlib.pyplot as plt

batch_size = 128
nb_epochs = 20

df = pd.read_csv('ncskew.csv')

n_feats = np.size(df, 1) - 1

train, test = train_test_split(df, train_size=0.8, random_state=42)
X_train = train.iloc[:, :n_feats]
y_train = train.iloc[:, n_feats:]
X_test = test.iloc[:, :n_feats]
y_true = test.iloc[:, n_feats:]

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

hidden_units1 = 160
hidden_units2 = 480
hidden_units3 = 256
learning_rate = 0.001

#Input Layer, Hidden Layers and Output Layer
input_layer = Input(shape=(n_feats,))
layer1 = Dense(hidden_units1, kernel_initializer='normal', activation='relu')(input_layer)
layer1 = Dropout(0.2)(layer1)
layer2 = Dense(hidden_units2, kernel_initializer='normal', activation='relu')(layer1)
layer2 = Dropout(0.2)(layer2)
layer3 = Dense(hidden_units3, kernel_initializer='normal', activation='relu')(layer2)
layer3 = Dropout(0.2)(layer3)
output_layer = Dense(1, kernel_initializer='normal', activation='linear')(layer3)

#Create ANN Model
ann = Model(inputs=input_layer, outputs=output_layer)

ann.summary()

# loss function
mse = MeanSquaredError()
ann.compile(
    loss=mse,
    optimizer=Adam(learning_rate=learning_rate),
    metrics=[mse]
)
# train the model
history = ann.fit(
    X_train,
    y_train,
    epochs=nb_epochs,
    batch_size=batch_size,
    validation_split=0.2,
    validation_data=(X_test, y_true)
)

#Save ANN Model
ann.save("ANN_ncskew.h5")
# Evaluate the model on the test data using `evaluate`
print("Evaluate on test data")
results = ann.evaluate(X_test, y_true, batch_size=batch_size)
print("test loss, test acc:", results)

# summarize history for accuracy
plt.plot(history.history['mean_squared_error'])
plt.plot(history.history['val_mean_squared_error'])
plt.title('model mean squared error')
plt.ylabel('mean_squared_error')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()