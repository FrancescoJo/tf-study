import tensorflow as tf
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import mse

# ...
# Epoch 9999/10000
# 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 22ms/step - acc: 0.2500 - loss: 0.2505
# Epoch 10000/10000
# 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 22ms/step - acc: 0.2500 - loss: 0.2505
# [array([[-0.03931295], [ 0.1412804 ]], dtype=float32), array([0.4395261], dtype=float32)]
# chapter_3/6_perceptron_xor_gate_defunct.py  200.87s user 90.61s system 125% cpu 291.48 total
#
# NOT WORKING: 25% Accuracy in 10000 epochs
def main():
    # Describe an XOR gate behaviour as following: Our perceptron accepts 2-D tensor(x) and sums it, and produces
    # output as y.
    # Not working: XOR gate requires multi-layered perceptrons, since
    # determining the results on vector space requires more than 1 perceptron,
    # compared to AND/OR gates.
    #
    # OR gate                AND Gate               XOR Gate
    #
    #   \ 1        1         0      \ 1           \ 1      \ 0
    #    \|                  |       \             \|       \
    #     \                  |        \           ==\========\===
    #     |\                 |         \            |\        \
    # --- 0 \----- 1     --- 0 ------ 0 \       --- 0 \----- 1 \
    #        \                           \             \        \
    #
    # Note that the solution and its slope of diagonal line above means weight and bias.
    x = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    y = np.array([[0]   , [1]   , [1]   , [0]   ])

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
    model.fit(x, y, epochs = 1000)

    # Evaluate results: must contains two weights and one bias
    print(model.get_weights())

if __name__ == "__main__":
    tf.random.set_seed(1234)  # For reproduction
    main()
