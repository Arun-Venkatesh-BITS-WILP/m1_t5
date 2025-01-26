import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10


def load_data():
    # Load CIFAR-10 dataset
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    # Normalize pixel values
    X_train = tf.image.rgb_to_grayscale(X_train)
    X_test = tf.image.rgb_to_grayscale(X_test)
    X_train = tf.cast(X_train, tf.float32) / 255.0
    X_test = tf.cast(X_test, tf.float32) / 255.0

    return X_train, y_train, X_test, y_test


def train_model():
    X_train, y_train, X_test, y_test = load_data()

    # Build the CNN model
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dense(10, activation="softmax"),
    ])

    # Compile the model
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    # Train the model
    model.fit(
        X_train, y_train, batch_size=256, epochs=1,
        validation_data=(X_test, y_test), verbose=1
    )

    # Save the model
    model.save("models/cnn_model.h5")
    print("CNN model trained!!!")


if __name__ == "__main__":
    train_model()
