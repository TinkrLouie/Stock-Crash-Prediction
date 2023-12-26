import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import keras
from sklearn import metrics
from keras.optimizers import Adam
from keras.losses import MeanSquaredLogarithmicError

batch_size = 64
nb_epochs = 100

df = pd.read_csv('ncskew.csv')

n_feats = np.size(df, 1) - 1

X = df.iloc[:, :n_feats].values
y = df.iloc[:, n_feats:].values
train, test = train_test_split(df, train_size=0.8, random_state=42)
X_train = train.iloc[:, :n_feats].values
y_train = train.iloc[:, n_feats:].values
X_test = test.iloc[:, :n_feats].values
y_test = test.iloc[:, n_feats:].values

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#ann = keras.models.Sequential()
#
##1st layer
#ann.add(keras.layers.Dense(units=n_feats, activation="linear"))
##2nd layer
#ann.add(keras.layers.Dense(units=n_feats, activation="linear"))
##Adding Output Layer
#ann.add(keras.layers.Dense(units=1))

hidden_units1 = 160
hidden_units2 = 480
hidden_units3 = 256
learning_rate = 0.002


# Creating model using the Sequential in tensorflow
def build_model_using_sequential():
    model = keras.Sequential([
        keras.layers.Dense(hidden_units1, kernel_initializer='normal', activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(hidden_units2, kernel_initializer='normal', activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(hidden_units3, kernel_initializer='normal', activation='relu'),
        keras.layers.Dense(1, kernel_initializer='normal', activation='linear')
    ])
    return model


# build the model
ann = build_model_using_sequential()

#compile
#custom_optimizer = keras.optimizers.SGD(learning_rate=0.002)
#ann.compile(optimizer=custom_optimizer, loss='mean_squared_error')
#
##Compiling ANN
##ann.compile(optimizer="Adam", loss='mean_squared_error')
#
#ann.fit(X_train, y_train, batch_size=batch_size, epochs=nb_epochs, validation_data=(X_test, y_test), verbose=2)

# loss function
msle = MeanSquaredLogarithmicError()
ann.compile(
    loss=msle,
    optimizer=Adam(learning_rate=learning_rate),
    metrics=[msle]
)
# train the model
ann.fit(
    X_train,
    y_train,
    epochs=nb_epochs,
    batch_size=batch_size,
    validation_split=0.2
)


ann.save("ANN_ncskew.h5")