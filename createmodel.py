import cv2
import numpy as np
import os
from random import shuffle
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.utils import to_categorical

dataset = 'C:/Users/acale/Desktop/FYP2/Code/Development/Final/EuroCoinDataset/train'
img_size = 100


def assign_label(img):
    current_label = img.split('.')[-3]
    
    if current_label == '2e': return [0]
    elif current_label == '1e': return [1]
    elif current_label == '50c': return [2]
    elif current_label == '20c': return [3]
    elif current_label == '10c': return [4]
    elif current_label == '5c': return [5]
    elif current_label == '2c': return [6]
    elif current_label == '1c': return [7]

def load_dataset():
    training_data = []
    for img in os.listdir(dataset):
        label = assign_label(img)
        path = os.path.join(dataset,img)
        img = cv2.imread(path)
        img = cv2.resize(img, (img_size,img_size))
        RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        training_data.append([np.array(RGB_img),np.array(label)])
    shuffle(training_data)
    return training_data

train_data = load_dataset()

# train contains the whole dataset apart from the last 800 images
train = train_data[:-800]
# test contains only the last 800 images of the dataset
test = train_data[-800:]

# Turn the data into Numpy Arrays
x_train = np.array([i[0] for i in train])
y_train = np.array([i[1] for i in train])

x_test = np.array([i[0] for i in test])
y_test = np.array([i[1] for i in test])

# Convert the data into categorical data
y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test)

# Normalise the pixels
x_train = x_train / 255
x_test = x_test / 255

# Build the model
model = Sequential() 
model.add(Conv2D(32, (5, 5), activation='relu', input_shape=(img_size,img_size,3)))	# Uses 32 5x5 filters
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dense(8, activation='softmax')) # because 8 labels
model.summary()

# Compile and train model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(x_train, y_train_one_hot, batch_size=256, epochs=10, validation_split=0.3 )

model.evaluate(x_test, y_test_one_hot)[1]

model.save('EuroCNN100.h5')

# Plot Accuracy of Model
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Accuracy of Model')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()

# Plot Loss of Model
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.show()