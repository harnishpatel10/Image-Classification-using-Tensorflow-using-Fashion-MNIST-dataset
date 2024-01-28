# Importing necessary libraries
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import pandas as pd

# Part 1: Prepare the Dataset

# Download the fashion mnist dataset and split it into training and testing sets
(X_train_full, y_train_full), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# Split the training set into a validation set and a (smaller) training set
X_train, y_train = X_train_full[:-5000], y_train_full[:-5000]
X_valid, y_valid = X_train_full[-5000:], y_train_full[-5000:]

# Normalize pixel values to the range [0, 1]
X_train, X_valid, X_test = X_train / 255., X_valid / 255., X_test / 255.

# Display the shape and data type of the training set
print("X_train shape:", X_train.shape)
print("X_train dtype:", X_train.dtype)

# Display the labels of the first training instance
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

print("Label for the first training instance:", class_names[y_train[0]])

# Part 2: Setting Hyperparameters and defining model

# Set the random seed for reproducibility
tf.random.set_seed(42)

# Create a sequential model
model = keras.Sequential([
    keras.layers.InputLayer(input_shape=[28, 28]),
    keras.layers.Flatten(),
    keras.layers.Dense(300, activation="relu"),
    keras.layers.Dense(100, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])

# Display model summary and information about its layers
model.summary()
model.layers
hidden1 = model.layers[1]
hidden1.name

# Get weights and biases of the first hidden layer
weights, biases = hidden1.get_weights()

# Display weights and biases information
print("Weights shape:", weights.shape)
print("Biases shape:", biases.shape)

# Compile the model
model.compile(loss="sparse_categorical_crossentropy",
              optimizer="sgd",
              metrics=["accuracy"])

# Part 3: Start Training

# Train the model
history = model.fit(X_train, y_train, epochs=30,
                    validation_data=(X_valid, y_valid))

# Display training history parameters and plot training history
print("Training history parameters:", history.params)
print("Training epochs:", history.epoch)
pd.DataFrame(history.history).plot(
    figsize=(8, 5), xlim=[0, 29], ylim=[0, 1], grid=True, xlabel="Epoch",
    style=["r--", "r--.", "b-", "b-*"])
plt.legend(loc="lower left")

# Save the model
model.save("fashion_mnist_model")

# Part 4: Start Testing
# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy*100:.2f}%")

# Using model to make predictions
X_new = X_test[:3]
y_proba = model.predict(X_new)
y_pred = y_proba.argmax(axis=-1)
class_predictions = np.array(class_names)[y_pred]

# Display predictions
for i in range(len(X_new)):
    plt.imshow(X_new[i], cmap="binary")
    plt.title(f"Predicted: {class_predictions[i]}, Actual: {class_names[y_test[i]]}")
    plt.show()

