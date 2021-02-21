import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras import backend as K
import keras as ks
import tensorflow as tf

plt.style.use('seaborn-whitegrid')

model_type = "fc"

(train_X, train_y), (test_X, test_y) = mnist.load_data()

train_X = ks.utils.normalize(train_X, axis=1)
test_X = ks.utils.normalize(test_X, axis=1)

train_X = np.expand_dims(train_X, -1)
test_X = np.expand_dims(test_X, -1)

model = ks.models.Sequential()

if model_type == "fc":
    model.add(ks.layers.Flatten())
    model.add(ks.layers.Dense(64, activation="relu"))
    # model.add(ks.layers.Dense(32, activation="relu"))
elif model_type == "cnn":
    # model.add(ks.Input(shape=(28, 28, 1)))
    model.add(ks.layers.Conv2D(32, input_shape=(28, 28, 1), kernel_size=(3, 3), activation="relu"))
    model.add(ks.layers.MaxPool2D(pool_size=(2, 2)))
    model.add(ks.layers.Flatten())
    model.add(ks.layers.Dropout(0.17))

# # Build the output layer
model.add(ks.layers.Dense(10, activation="softmax"))

# model.build(input_shape=(28, 28, 1))

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

history = model.fit(x=train_X, y=train_y, epochs=17, validation_data=(test_X, test_y))
model.summary()


test_loss, test_acc = model.evaluate(x=test_X, y=test_y)
print("\nTest accuracy:", test_acc)
print("Test loss:", test_loss)

predictions = model.predict([test_X])
print(np.argmax(predictions[1706]))
plt.imshow(test_X[1706], cmap="gray")
plt.show()


def display_numbers():
    for i in range(9):
        plt.subplot(330 + 1 + i)
        plt.imshow(train_X[i], cmap=plt.get_cmap("gray"))
    plt.show()


# plot diagnostic learning curves
def summarize_diagnostics(histories):
    # plot loss
    plt.subplot(2, 1, 1)
    plt.title('Cross Entropy Loss')
    plt.plot(histories.history['loss'], label='train')
    plt.plot(histories.history['val_loss'], label='test')
    plt.legend(loc=0)
    # plot accuracy
    plt.subplot(2, 1, 2)
    plt.title('Classification Accuracy')
    plt.plot(histories.history['accuracy'], label='train')
    plt.plot(histories.history['val_accuracy'], label='test')
    plt.legend(loc=3)
    plt.show()


if __name__ == "__main__":
    summarize_diagnostics(history)
