import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import sklearn

# x_train = array containing all of the mnist images (60,000 28x28)
# y_train = array containing labels for the images
# x_test = array containing 10,000 images for testing
# y_test = array containing labels for test data
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# reshapes the data to be 28x28x1 instead of just 28x28
x_train = x_train.reshape((x_train.shape[0], 28, 28, 1)).astype('float32') / 255.0
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1)).astype('float32') / 255.0

# CNN model
model = keras.Sequential(
    [
        keras.Input(shape=(28, 28, 1)),  # tells the model what the input tensor will be
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"), # convolutional layer with an output of 32 feature maps (26x26x32)- relu outputs max signed value
        layers.MaxPooling2D(pool_size=(2, 2)), # takes the max in a 2x2 area and outputs it- becoems 13x13x32
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"), # second convolutional layer with an output of 64 feature maps- 11x11x64
        layers.MaxPooling2D(pool_size=(2, 2)), # second max pool with output 5x5x64
        layers.Flatten(), # flattens the max pool output to array with data about every individual pixel (5*5*64 = 1600)
        layers.Dropout(0.5), # randomly drops out 50% of the neurons (changes their input to 0)
        layers.Dense(10, activation="softmax"), # classifies to 0-9 - output = softmax(dot(input, kernel) + bias)
    ]
)

# lists the output size for each layer
model.summary()

# specifies the optimizer, loss function, and metrics to evaluate by
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# trains the model with batch sizes of 128 (takes 422 batches to pass through the dataset in an epoch)
# 10% of the data is used for validation (approx 6k)- uses this to evaluate the loss
model.fit(x_train, y_train, batch_size=128, epochs=10, validation_split=0.1)

# evaluates the model and prints accuracy, loss, validation accuracy, and validation loss per epoch
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"\nTest accuracy: {test_acc}")

# generates preds and makes a confusion matrix with them
y_preds_probs = model.predict(x_test)
y_preds = np.argmax(y_preds_probs, axis=1)
conf_matrix = tf.math.confusion_matrix(labels=y_test, predictions=y_preds)
sklearn.metrics.ConfusionMatrixDisplay(conf_matrix)