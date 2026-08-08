# THE FIRST CNN FOR MNIST

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers


# 1. Load MNIST data -------------------------
# if this does not work, you may have to run Python’s certificate installer. I went to Python application and double clicked on "Install Certificates.command" to install it.
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()


# 2. Prepare the data -------------------------

# Convert pixel values from 0-255 to 0-1
# CNNs train much better when input values are small (gradients can be unstable with larger numbers such as 255)
X_train = X_train / 255.0
X_test = X_test / 255.0


# CNN expects: height, width, channels
# MNIST images are grayscale, so channels = 1
# Latest TensorFlow/Keras need four dimentions: (batch_size, height, width, channels), here -1 will be computed as 60000 images, so the overall result is 60000, 28, 28, 1
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)


# 3. Show one image -------------------------
plt.imshow(X_train[0].reshape(28, 28), cmap="gray")
plt.title("Example MNIST Digit")
plt.axis("off")
plt.show()


# 4. Create CNN model -------------------------
# we must use 10 in the last dense layer as we have 10 number of classes.
model = keras.Sequential([
    keras.Input(shape=(28, 28, 1)),

    layers.Conv2D(
        filters=8,
        kernel_size=(3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D(
        pool_size=(2, 2)
    ),

    layers.Flatten(),

    layers.Dense(
        10,
        activation="softmax"
    )
])


# 5. Compile -------------------------
# "sparse_categorical_crossentropy" loss function is good for multi-class classification problems
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# 6. Train -------------------------
model.fit(
    X_train,
    y_train,
    epochs=5,
    verbose=1
)


# 7. Evaluate -------------------------
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print("\nTest accuracy:", accuracy)


# 8. Predict one image -------------------------
predictions = model.predict(X_test[0:1])
predicted_digit = np.argmax(predictions[0])
print("\nTrue digit:", y_test[0])
print("Predicted digit:", predicted_digit)
print("Probabilities:", predictions[0])


# 9. Show prediction -------------------------
plt.imshow(X_test[0].reshape(28, 28), cmap="gray")
plt.title(f"True: {y_test[0]}, Predicted: {predicted_digit}")
plt.axis("off")
plt.show()