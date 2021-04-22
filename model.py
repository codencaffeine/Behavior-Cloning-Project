import csv
import cv2
import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import random
import pickle
import pandas as pd

pkl = False

if pkl:
    print("Loading Data from pickle...")
    with open('data.pkl','rb') as f:
        [new_images, new_measurements] = pickle.load(f)
else:
    print("Loading Data from files...")
    dir = './data/IMG/'

    df = pd.read_csv('./data/driving_log.csv', header=None)

    images = []
    image_flipped = []
    measurements = []
    measurement_flipped = []

    for index, row in df.iterrows():
        for col in range(3):
            source_path = row[col]
            filename = source_path.split('/')[-1]
            current_path = dir + filename
            image = cv2.imread(current_path)
            images.append(image)
            image_flipped.append(np.fliplr(image))

        measurement = float(row[3])
        measurements.append(measurement)
        measurements.append(measurement+0.2)
        measurements.append(measurement-0.2)

    print("Total Images: ", len(images)," | Total Measurements: ",len(measurements))

    aug_images = images + image_flipped
    measurement_flipped = [-1*x for x in measurements]
    aug_measurements = measurements + measurement_flipped

    nbin = 25
    hist, bins = np.histogram(aug_measurements, nbin)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width)
    plt.savefig("./examples/old_hist.png")
    plt.show()
    avg_value = np.mean(hist)
    print("Mean of Measurement: ", avg_value)

    keep_prob = []
    for i in range(nbin):
        if hist[i] <= avg_value:
            keep_prob.append(1)
        else:
            keep_prob.append(avg_value/hist[i])

    remove_ind = []
    for i in range(len(aug_measurements)):
        for j in range(nbin):
            if aug_measurements[i] > bins[j] and aug_measurements[i] <= bins[j+1]:
                if random.random() < (1-keep_prob[j]):
                    remove_ind.append(i)
    
    df1= pd.DataFrame(list(zip(aug_images, aug_measurements)))
    df1.drop(remove_ind, inplace=True)

    new_images = df1[0].tolist()
    new_measurements = df1[1].tolist()

    print("Saving pickle... ")
    with open('data.pkl','wb') as f:
        pickle.dump([new_images, new_measurements], f)

nbin = 25
hist, bins = np.histogram(new_measurements, nbin)
width = 0.7 * (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2
plt.bar(center, hist, align='center', width=width)
plt.savefig("./examples/new_hist.png")
plt.show()
avg_value = np.mean(hist)

X_train = np.array(new_images)
y_train = np.array(new_measurements)

##### Model Architecture #####

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D
from keras.layers import Cropping2D, Dropout, Activation, MaxPooling2D
from keras.optimizers import Adam, SGD, RMSprop
from tensorflow import keras

activation_fn = "relu"
learning_rate = 1e-4
epochs = 5
loss_fn = keras.losses.MeanSquaredError()
optimizer_fn = Adam(learning_rate=learning_rate)

# optimizer_fn = SGD(learning_rate=learning_rate)
# optimizer_fn = RMSprop(learning_rate=learning_rate)

model = Sequential([
    Lambda(lambda x: x/255.0 - 0.5, input_shape = (160,320,3)),
    Cropping2D(cropping = ((70,25),(0,0))),

    Conv2D(24, kernel_size = (5, 5),strides = (2,2), activation = activation_fn),
    Conv2D(36, kernel_size = (5, 5),strides = (2,2), activation = activation_fn),
    Conv2D(48, kernel_size = (5, 5),strides = (2,2), activation = activation_fn),
    Conv2D(64, kernel_size = (3, 3), activation = activation_fn),
    Conv2D(64, kernel_size = (3, 3), activation = activation_fn),

    Flatten(),

    Dense(1164, activation = activation_fn),
    Dense(100, activation = activation_fn),
    Dense(50, activation = activation_fn),
    Dense(10, activation = activation_fn),
    Dense(1),
])

model.compile(optimizer = optimizer_fn,
              loss      = loss_fn,
              metrics=['mse'])
              
history_object = model.fit(X_train, y_train, epochs=epochs, validation_split=0.2, shuffle=True)

# plt.plot(history_object.history['accuracy'])
# plt.xlabel("Epochs")
# plt.ylabel("Accuracy")
# plt.show()

plt.plot(history_object.history['mse'])
plt.xlabel("Epochs")
plt.ylabel("MSE")
plt.show()

## Save model as h5
model.save('model.h5')
print("Model Saved!")
