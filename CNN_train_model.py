import numpy as np
import cv2
import os
import pickle

from keras import Sequential
from keras.src.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.src.optimizers import Adam
from keras.src.preprocessing.image import ImageDataGenerator
from keras.src.utils import to_categorical
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

################
trainingPath = "./dataset/numbers/trainingSet"
testRatio = 0.2
images = []
classNumber = []


################

def preProcessImage(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.equalizeHist(image)
    image = image / 255

    return image


def myModel():
    noOfFilters = 60
    sizeOfFilter1 = (5, 5)
    sizeOfFilter2 = (3, 3)
    sizeOfPool = (2, 2)
    noOfNode = 500
    model = Sequential()
    model.add(Conv2D(noOfFilters, sizeOfFilter1, input_shape=(28, 28, 1), activation='relu'))
    model.add(Conv2D(noOfFilters, sizeOfFilter1, activation='relu'))
    model.add(MaxPooling2D(pool_size=sizeOfPool))
    model.add(Conv2D(noOfFilters // 2, sizeOfFilter2, activation='relu'))
    model.add(Conv2D(noOfFilters // 2, sizeOfFilter2, activation='relu'))
    model.add(MaxPooling2D(pool_size=sizeOfPool))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(noOfNode, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(numberOfClasses, activation='softmax'))
    model.compile(Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


myList = os.listdir(trainingPath)
numberOfClasses = len(myList)

for x in range(0, numberOfClasses):
    myImagePath = os.listdir(trainingPath + "/" + str(x))
    for y in myImagePath:
        myImage = cv2.imread(trainingPath + "/" + str(x) + "/" + str(y))
        myImage = cv2.resize(myImage, (28, 28))
        images.append(myImage)
        classNumber.append(x)

images = np.array(images)
classNumber = np.array(classNumber)

X_train, X_test, y_train, y_test = train_test_split(images, classNumber, test_size=testRatio)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=testRatio)

# Preprocess images
X_train = np.array(list(map(preProcessImage, X_train)))
X_test = np.array(list(map(preProcessImage, X_test)))
X_val = np.array(list(map(preProcessImage, X_val)))

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], X_val.shape[2], 1)

datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, shear_range=0.1, zoom_range=0.2)
datagen.fit(X_train)

# Onehot encoding
y_train = to_categorical(y_train, numberOfClasses)
y_val = to_categorical(y_val, numberOfClasses)
y_test = to_categorical(y_test, numberOfClasses)

model = myModel()
print(model.summary())

print("SHAPE TRAIN:", X_train.shape)

history = model.fit(datagen.flow(X_train, y_train, batch_size=50),
                    epochs=20,
                    validation_data=(X_val, y_val), shuffle=True)

plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.legend(['training', 'validation'])
plt.xlabel('Epoch')

plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.legend(['training', 'validation'])
plt.xlabel('Epoch')

plt.show()

score = model.evaluate(X_test, y_test, verbose=0)
print('Test Score = ', score[0])
print('Accuracy', score[1])

model.save("model/CNN_model.h5")
