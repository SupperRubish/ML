import os
import random

import joblib
import pandas as pd
import numpy as np
from keras.callbacks import EarlyStopping
from keras.layers import LSTM
from keras.src.optimizers.schedules import ExponentialDecay
# from keras.optimizers.schedules.learning_rate_schedule import ExponentialDecay

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, Dropout, MaxPooling1D
from keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from keras.optimizers import SGD, Adam
from scipy.signal import butter, lfilter
import clean_data

# We will load each file and assign labels to each gesture class
# The labels are based on the filenames which indicate the gesture

# Assign labels based on the file names
labels_dict = {
        'circle': 0,  # Assuming 'circle.xls' corresponds to the 'circle' gesture
        'come': 1,  # Assuming 'come.xls' corresponds to the 'come here' gesture
        'go': 2,  # Assuming 'go.xls' corresponds to the 'go away' gesture
        'wave': 3  # Assuming 'wave.xls' corresponds to the 'wave' gesture
    }
names = ["c", "p", "l"]  # Names of group members

# Load each dataset and create a combined dataframe with labels
dataframes = []
num=0
currentPath = os.getcwd()  # Assigns the path of the current working directory to the variable 'currentPath'
print(currentPath)
for i in names:
    for j in range(1,11):
        i=str(i)
        j=str(j)
        file_circle=currentPath+f"/data/circle/{i}_circle_{j}.xls"  # Load each 'circle' dataset
        file_come = currentPath+f'/data/come/{i}_come_{j}.xls'  # Load each 'come' dataset
        file_go= currentPath+f'/data/go/{i}_go_{j}.xls'  # Load each 'go' dataset
        file_wave= currentPath+f'/data/wave/{i}_wave_{j}.xls'  # Load each 'wave' dataset

        # Drop the 'Time(s)' column and read the first 1400 rows of data
        circle_data = (clean_data.cleanData(file_circle)).drop(['Time (s)'], axis=1).values[:1400]
        come_data = (clean_data.cleanData(file_come)).drop(['Time (s)'], axis=1).values[:1400]
        go_data = (clean_data.cleanData(file_go)).drop(['Time (s)'], axis=1).values[:1400]
        wave_data = (clean_data.cleanData(file_wave)).drop(['Time (s)'], axis=1).values[:1400]

        # circle_data = (pd.read_excel(file_circle)).drop(['Time (s)'], axis=1).values[:1400]
        # come_data = (pd.read_excel(file_come)).drop(['Time (s)'], axis=1).values[:1400]
        # go_data = (pd.read_excel(file_go)).drop(['Time (s)'], axis=1).values[:1400]
        # wave_data = (pd.read_excel(file_wave)).drop(['Time (s)'], axis=1).values[:1400]

        for k in range(0,1400,100):  # Label each 100 rows as a gesture

            #circle
            circle=circle_data[k:k+100].copy()
            dataframes.append([circle,0])
            #come
            come=come_data[k:k+100].copy()
            dataframes.append([come,1])
            #go
            go=go_data[k:k+100].copy()
            dataframes.append([go,2])
            #wave
            wave=wave_data[k:k+100].copy()
            dataframes.append([wave,3])
            num+=4

# Shuffle the dataset randomly
random.shuffle(dataframes)

# Unpack the tuples in the 'dataframes' list and recombines the elements into two tuples
actions, labels = zip(*dataframes)

# Stack all gesture data into a two-dimensional array for normalization
actions_stacked = np.vstack(actions)

# Initialize the normalizer
scaler = StandardScaler()

# Train the normalizer and normalize the data
actions_scaled = scaler.fit_transform(actions_stacked)

# Save normalizer
joblib.dump(scaler, 'scaler.save')

# Reshape the normalized data back into original gesture
# 100 represents every 100 rows as a gesture and 4 represents 4 features
actions_scaled_reshaped = actions_scaled.reshape(num, 100, 4)

# Transform integer labels into a binary(one-hot) format
y = to_categorical(labels)


# Divide the data into training set and test set
# Ratio of training set and test set is 4:1
X_train, X_test, y_train, y_test = train_test_split(actions_scaled_reshaped, y, test_size=0.2, random_state=42)

print(X_train)
print(y_train)

# Build 1D CNN model

# Sequential model is one of Keras models for linear stacking layers which means you can add layer after layer in order,
# And each layer has only one input and one output
model = Sequential()

# Add a one-dimensional convolutional layer(Conv1D). This is the core of 1D CNN, which can extract features of data.
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 4)))

# Add a dropout layer to reduce over-fitting. Dropout improves the generalisation of the model by randomly discarding
# a portion of the neuron output from the networking during training
# 0.5 indicates discard rate of 50%, means 50% of the neuron outputs are randomly selected during training and set to 0
model.add(Dropout(0.5))

# Add a one-dimensional maximum pooling layer(MaxPooling1D). This is used to reduce the dimensionality of the data
# and reduce the amount of computation while maintaining important information.
# That is the largest of every 2 values is chosen as the output
model.add(MaxPooling1D(pool_size=2))

# Add a Flatten layer to spread the output of the previous layer. This is a common technique used when transitioning
# from a convolutional or pooling layer to a fully connected layer(Dense layer)
# model.add(Flatten())
model.add(LSTM(100))
model.add(Dense(100, activation='relu'))
model.add(Dense(y.shape[1], activation='softmax'))

# Compilation model
# Optimiser and learning rate
# SGD
# from tensorflow.keras.optimizers import SGD
#
# #Use SGD optimiser and specify learning rate
# optimizer = SGD(learning_rate=0.01)
#
# model.compile(loss='categorical_crossentropy',
#               optimizer=optimizer,
#               metrics=['accuracy'])

# RMSprop
# from tensorflow.keras.optimizers import RMSprop
#
# # Use the RMSprop optimizer and specify the learning rate
# optimizer = RMSprop(learning_rate=0.001)
#
# model.compile(loss='categorical_crossentropy',
#               optimizer=optimizer,
#               metrics=['accuracy'])

# Learning rate decay
# from tensorflow.keras.optimizers import Adam

# Initialise a learning rate decay function
lr_schedule = ExponentialDecay(
    initial_learning_rate=1e-2,
    decay_steps=10000,
    decay_rate=0.9)

# Use the Adam optimiser with a learning rate scheduler
optimizer = Adam(learning_rate=lr_schedule)

# The smaller the learning rate, the closer to the optimal solution, the slower the efficiency
# optimizer = SGD(learning_rate=0.01)

# model.compile(loss='categorical_crossentropy',
#               optimizer=optimizer,
#               metrics=['accuracy'])
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
# Train the model
history = model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test),callbacks=early_stopping)

# Evaluate the model by showing the accuracy
accuracy = model.evaluate(X_test, y_test, verbose=0)[1]
print(f'Test accuracy: {accuracy}')

model.save('my_model.h5')