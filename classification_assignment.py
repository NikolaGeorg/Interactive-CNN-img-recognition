from tensorflow.keras.datasets import cifar10
from keras import layers, models, backend
import os
import datetime
from keras.utils import to_categorical
import tensorflow as tf

data_labels = [
    'airplane',
    'car',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck'
]

num_classes = len(data_labels)


def cnn(num_classes):
    data_augmentation = models.Sequential([
        layers.RandomFlip("horizontal"),
    ])

    model = models.Sequential([
        layers.Input((32, 32, 3)),
        layers.Rescaling(1. / 255),
        data_augmentation,

        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2)),
        #layers.Dropout(0.2),

        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2)),
        #layers.Dropout(0.3),

        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2)),
        #layers.Dropout(0.35),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.summary()

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

def train(x_train, y_train, x_test, y_test, num_classes, epochs):
    x_train = x_train.reshape((-1, 32, 32, 3)).astype("float32")
    x_test = x_test.reshape((-1, 32, 32, 3)).astype("float32")
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    model = cnn(num_classes)
    logdir = os.path.join("logs", f"[cnn_model]: {datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}")
    callbacks = []

    if backend.backend() == "tensorflow":
        callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1))

    model.fit(
        x_train, y_train,
        epochs=epochs,
        batch_size=64,
        validation_split=0.2,
        validation_freq=1,
        callbacks=callbacks
    )

    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f"[cnn_model] Test Accuracy: {test_acc:.4f}, Test Loss: {test_loss:.4f}")

    model.save("cifar10_cnn.model.keras")
    print("Моделът е запазен")
    return model, test_acc

if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    model, model_acc = train(x_train, y_train, x_test, y_test, num_classes, 15)


