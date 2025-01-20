import tensorflow as tf
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import mse

# ...
# Epoch 183/500
# 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 22ms/step - acc: 0.7500 - loss: 0.0972
# Epoch 184/500
# 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 22ms/step - acc: 1.0000 - loss: 0.0969
# ...
# [array([[0.5335133 ], [0.42863002]], dtype=float32), array([0.27245152], dtype=float32)]
# chapter_3/4_perceptron_or_gate.py  11.91s user 5.93s system 126% cpu 14.142 total
def main():
    # Describe an OR gate behaviour as following: Our perceptron accepts 2-D tensor(x) and sums it, and produces
    # output as y.
    x = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    y = np.array([[0]   , [1]   , [1]   , [1]   ])

    model = Sequential()
    # Construct a single dense layered perceptron
    model.add(Dense(1,
                    input_shape = (2, ),
                    activation = 'linear'))

    # Prepare model
    model.compile(optimizer = SGD(),
                  loss = mse,
                  metrics = ['acc'])    # This constraints output evaluation  as list

    # Learn it!
    model.fit(x, y, epochs = 500)

    # Evaluate results: must contains two weights and one bias
    print(model.get_weights())

if __name__ == "__main__":
    tf.random.set_seed(1234)  # For reproduction
    main()
