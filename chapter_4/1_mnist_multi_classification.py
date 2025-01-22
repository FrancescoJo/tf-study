import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tensorflow.keras.datasets.mnist import load_data
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

TF_RANDOM_SEED = 1234

def main():
    (x_train, y_train), (x_test, y_test) = load_data(path="mnist.npz")

    data_size_x = len(x_train)
    data_size_y = len(y_train)

    if data_size_x != data_size_y:
        print(f"Data size mismatch!! Cannot continue!! (Image data size: {x_train}, Label data size: {y_train}")
        exit(1)

    # sample_size = 3
    # data_size = data_size_x
    # random_idx = np.random.randint(data_size, size=sample_size)
    #
    # for idx in random_idx:
    #     image = x_train[idx, :]
    #     label = y_train[idx]
    #
    #     plt.figure()
    #     plt.imshow(image)
    #     plt.title("data at #%d, label: %d" % (idx, label))

    # Separate data for training and validation
    x_train, x_validate, y_train, y_validate = train_test_split(x_train, y_train,
                                                                test_size = 0.3,
                                                                random_state = TF_RANDOM_SEED)

    num_x_train = x_train.shape[0]
    num_x_validate = x_validate.shape[0]
    num_x_test = x_test.shape[0]

    # Square sized image of 28 pixels
    # Our test data are truecolor image, therefore we divide them by 255 to normalise in range of 0..1
    x_train = (x_train.reshape((num_x_train, 28 * 28))) / 255
    x_validate = (x_validate.reshape((num_x_validate, 28 * 28))) / 255
    x_test = (x_test.reshape((num_x_test, 28 * 28))) / 255
    y_train = to_categorical(y_train)
    y_validate = to_categorical(y_validate)
    y_test = to_categorical(y_test)

    model = Sequential()
    # ????
    model.add(Dense(64, activation = "relu", input_shape = (784, )))  # 28 * 28 flattened to 784
    model.add(Dense(32, activation = "relu"))
    model.add(Dense(10, activation = "softmax"))

    model.compile(optimizer = "adam",
                  loss = "categorical_crossentropy",
                  metrics=["acc"])
    # Train it!
    history = model.fit(x_train, y_train,
                        epochs = 30,
                        batch_size = 128,
                        validation_data = (x_validate, y_validate))

    model.evaluate(x_test, y_test)

if __name__ == "__main__":
    tf.random.set_seed(TF_RANDOM_SEED)
    main()
