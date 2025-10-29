import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Load and preprocess the MNIST dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Reshape data to include a channel dimension (for grayscale images)
x_train = x_train.reshape((x_train.shape[0], 28, 28, 1)).astype('float32') / 255.0
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1)).astype('float32') / 255.0

# Define the CNN model
model = keras.Sequential(
    [
        keras.Input(shape=(28, 28, 1)),  # Input layer with image dimensions
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"), # First convolutional layer
        layers.MaxPooling2D(pool_size=(2, 2)), # First max-pooling layer
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"), # Second convolutional layer
        layers.MaxPooling2D(pool_size=(2, 2)), # Second max-pooling layer
        layers.Flatten(), # Flatten the output for the dense layers
        layers.Dropout(0.5), # Dropout layer to prevent overfitting
        layers.Dense(10, activation="softmax"), # Output layer for 10 classes (digits 0-9)
    ]
)

# Compile the model
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# Train the model
model.fit(x_train, y_train, batch_size=128, epochs=10, validation_split=0.1)

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"\nTest accuracy: {test_acc}")