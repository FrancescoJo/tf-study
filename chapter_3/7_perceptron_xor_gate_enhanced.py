import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.losses import mse
import matplotlib.pyplot as plt

def create_and_train_model(learning_rate=0.001, neurons=16, epochs=200, batch_size=4):
    # Input data for XOR gate
    x = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    # Create model
    model = Sequential([
        Dense(neurons, input_shape=(2,), activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    # Compile model with specified learning rate
    optimizer = RMSprop(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=mse, metrics=['accuracy'])

    # Train model
    history = model.fit(
        x, y,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

    return model, history

def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Plot loss
    ax1.plot(history.history['loss'])
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')

    # Plot accuracy
    ax2.plot(history.history['accuracy'])
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')

    plt.tight_layout()
    return fig

def evaluate_model(model):
    x_test = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    predictions = model.predict(x_test)
    print("\nModel predictions:")
    for inputs, pred in zip(x_test, predictions):
        print(f"Input: {inputs}, Predicted output: {pred[0]:.3f}")

def main():
    learning_rates = [0.01, 0.001, 0.0001]
    best_loss = float('inf')
    best_params = None
    best_model = None

    for lr in learning_rates:
        print(f"\nTraining with learning rate: {lr}")
        model, history = create_and_train_model(
            learning_rate=lr,
            neurons=8,  # Reduced from 16
            epochs=100  # Reduced from 200
        )

        final_loss = history.history['loss'][-1]
        if final_loss < best_loss:
            best_loss = final_loss
            best_params = {'learning_rate': lr}
            best_model = model

    print(f"\nBest parameters found: {best_params}")
    print(f"Best loss achieved: {best_loss:.6f}")

    evaluate_model(best_model)

if __name__ == "__main__":
    tf.random.set_seed(1234)
    main()
