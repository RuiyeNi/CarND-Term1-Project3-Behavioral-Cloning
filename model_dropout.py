import csv
import cv2
import numpy as np

left_flag = True
right_flag = True
savename = 'model_dropout3.h5'

lines = []
datafile =  'data' 
with open('./' + datafile +'/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)


# Read in images
images = []
measurements = []
count = 0
correction_left = 0.4  # 0.4
correction_right = 0.25
sample_rate = 0.8
for line in lines:
    count += 1
    sample_prob = np.random.rand()
    if count !=1:
        # center camera image
        source_path = line[0]
        filename = source_path.split('/')[-1]
        current_path = './' + datafile + '/IMG/' + filename
        image = cv2.imread(current_path)
        # Color space needs to be consistent with drive.py 
        # Very important!!
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        measurement = float(line[3])

        if (np.abs(measurement) > 0) or (np.abs(measurement) == 0 and sample_prob <= sample_rate):
            # center camera image
            images.append(image)
            measurements.append(measurement)

            if left_flag:
                # left camera image
                source_path = line[1]
                filename = source_path.split('/')[-1]
                current_path = './' + datafile + '/IMG/' + filename
                left_image = cv2.imread(current_path)
                images.append(left_image)
                measurements.append(measurement + correction_left)

            if right_flag:
                # right camera image
                source_path = line[2]
                filename = source_path.split('/')[-1]
                current_path = './' + datafile + '/IMG/' + filename
                right_image = cv2.imread(current_path)
                images.append(right_image)
                measurements.append(measurement - correction_right)



# Augmented images: flip
augmented_images, augmented_measurements = [], []
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image, 1))
    augmented_measurements.append(measurement*-1.0)


# Convert training data to array
X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
# Normalize data
model.add(Lambda(lambda x:x/255.0 - 0.5, input_shape=(160, 320, 3)))
# Crop images
model.add(Cropping2D(cropping=((70, 25), (0, 0))))

# Default network
#model.add(Convolution2D(6, 5, 5, activation="relu"))
#model.add(MaxPooling2D())
#model.add(Convolution2D(6, 5, 5, activation="relu"))
#model.add(MaxPooling2D())
#model.add(Flatten())
#model.add(Dense(120))
#model.add(Dense(84))
#model.add(Dense(1))


# NIVIDA, 5 convolution layers, 3 fully connected layers
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation="relu"))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation="relu"))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation="relu"))
model.add(Convolution2D(64, 3, 3, subsample=(1, 1), activation="relu"))
model.add(Convolution2D(64, 3, 3, subsample=(1, 1), activation="relu"))
model.add(Flatten())
model.add(Dense(1164))
# Add dropout to deal with overfitting
model.add(Dropout(0.4))
model.add(Dense(100))
model.add(Dense(50))
# Add dropout to deal with overfitting
model.add(Dropout(0.1))
model.add(Dense(10))
model.add(Dense(1))
# Get model summary
model.summary()

# Compile and fit model
model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=3, verbose=1)

# Save model
model.save(savename)