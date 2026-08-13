import os
import cv2
import numpy as np

from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers


# Try to fill the parts with ... or you are welcome to write it in a new way.

# 1. Parameters -------------------------

# 1. Parameters -------------------------
SIZE = 64

parasitized_path = "cell_images/Parasitized"
uninfected_path = "cell_images/Uninfected"

# 2. Create dataset arrays -------
X = []
y = []

# 3. Read parasitized images ----------
count = 0

# maximum number of images to use from each class
max_class = 2000

for filename in os.listdir(parasitized_path):
    if count >= max_class:
        break

    filepath = os.path.join(parasitized_path, filename)
    image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

    if image is None:
        continue

    image = cv2.resize(image, (SIZE, SIZE))

    X.append(image)

    # Label 1 = parasitized
    y.append(1)

    count += 1

# 4. Read uninfected images ----------
count = 0

for filename in os.listdir(uninfected_path):
    if count >= max_class:
        break

    filepath = os.path.join(uninfected_path, filename)

    image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

    if image is None:
        continue

    image = cv2.resize(image, (SIZE, SIZE))

    X.append(image)

    # Label 0 = uninfected
    y.append(0)

    count += 1

# 5. Convert to NumPy arrays ----------
X = np.array(X)
y = np.array(y)

# 6. Print information ----------------
print("X shape:", X.shape)
print("y shape:", y.shape)

print("Number of parasitized images:", np.sum(y == 1))
print("Number of uninfected images:", np.sum(y == 0))

# 7. Prepare the data for CNN -------------------------
X = X / 255.0

# CNN expects: number of images, height, width, channels
X = X.reshape(-1, SIZE, SIZE, 1)

print("X shape after reshape:", X.shape)


# 8. Split into training and testing data -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training images:", X_train.shape[0])
print("Testing images:", X_test.shape[0])


# 9. Create CNN model -------------------------
model = keras.Sequential([

    keras.Input(shape=(SIZE, SIZE, 1)),

    layers.Conv2D(
        filters=32,
        kernel_size=(3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D(
        pool_size=(2, 2)
    ),

    layers.Conv2D(
        filters=64,
        kernel_size=(3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D(
        pool_size=(2, 2)
    ),

    layers.Flatten(),

    layers.Dense(
        128,
        activation="relu"
    ),

    layers.Dense(
        1,
        activation="sigmoid"
    )
])

# 10. Compile -------------------------
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# 11. Train -------------------------
model.fit(
    X_train,
    y_train,
    batch_size=20,
    epochs=35,
    verbose=1
)

# 12. Evaluate -------------------------
loss, accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print("\nTest accuracy:", accuracy)