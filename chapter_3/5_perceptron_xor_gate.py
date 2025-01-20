import tensorflow as tf
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.losses import mse

# ...
# Epoch 72/200
# 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 22ms/step - acc: 0.7500 - loss: 0.2282
# Epoch 73/200
# 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 22ms/step - acc: 1.0000 - loss: 0.2278
# ...
# [array([[-0.17035264, -0.13393143,  0.58737016,  0.4435201 ,  0.11558281,  0.4944732 ,  0.15395641,  0.35936502,
#           0.47144875, -0.19508076, -0.5572989 , -0.14963008, -0.5543906 ,  0.27645743, -0.31230122, -0.00313818
#         ],
#         [ 0.03665588, -0.10173348, -0.01859038,  0.42513743, -0.00221925,  0.17029312,  0.1734926 , -0.35807487,
#           0.30020636,  0.43054062, -0.14208063,  0.36132455,  0.12578875,  0.00242301,  0.52568424, -0.50161386
#        ]], dtype=float32
#  ),
#  array([-0.03731607,  0         ,  0.02550827, -0.2238425 ,  0.01288853,  0.0145911 ,  0.00948213,  0.00092558,
#          0.01536523,  0.19674464,  0         ,  0.15108676, -0.12718405,  0.00431087, -0.03469678,  0
#        ], dtype=float32
#  ),
#  array([[-0.1936881 ], [ 0.23965895], [-0.23778093], [-0.25285316], [-0.3445593 ], [ 0.17504027], [ 0.05339933], [ 0.368486  ],
#         [ 0.27364916], [-0.0552451 ], [-0.01376647], [-0.34040788], [-0.28809926], [-0.06346657], [ 0.4824327 ], [ 0.06535721]
#         ], dtype=float32
#  ),
#  array([-0.01371722], dtype=float32)
# ]
# chapter_3/5_perceptron_xor_gate.py  6.67s user 3.69s system 137% cpu 7.513 total
def main():
    # Describe an XOR gate behaviour as following: Our perceptron accepts 2-D tensor(x) and sums it, and produces
    # output as y.
    x = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    y = np.array([[0]   , [1]   , [1]   , [0]   ])

    model = Sequential()
    # Construct a multiple dense layered perceptron
    # 16 perceptrons for 200 epochs shows stable results for 100 runs
    model.add(Dense(16,
                    input_shape = (2, ),
                    activation = 'relu'))         # sigmoid, tanh, ReLU
    model.add(Dense(1, activation = 'sigmoid'))

    # Prepare model
    model.compile(optimizer = RMSprop(),
                  loss = mse,
                  metrics = ['acc'])    # This constraints output evaluation  as list

    # Learn it!
    model.fit(x, y, epochs = 200)

    # Evaluate results: must contains two weights and one bias
    print(model.get_weights())

if __name__ == "__main__":
    tf.random.set_seed(1234)  # For reproduction
    main()
