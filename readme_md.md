# MNIST Digit Classification using TensorFlow & Keras

An end-to-end Deep Learning project that builds, trains, and evaluates a Neural Network to classify handwritten digits (0–9) from the classic MNIST dataset. 

Achieved **98%+ test accuracy** by implementing proper data normalization and a multi-layer Artificial Neural Network (ANN).

---

## 📌 Project Overview

Handwritten digit recognition is a benchmark problem in computer vision and machine learning. This project demonstrates how to construct an image classification pipeline using TensorFlow/Keras to achieve high precision on 28x28 grayscale digit images.

### Key Highlights
- **Dataset:** MNIST (60,000 training images, 10,000 testing images)
- **Model Architecture:** Multi-Layer Perceptron (Sequential ANN)
- **Activation Functions:** ReLU (Hidden Layer) & Softmax (Output Layer)
- **Optimizer:** Adam
- **Loss Function:** Sparse Categorical Cross-entropy
- **Accuracy:** ~98.6%

---

## 🧠 Model Architecture

The neural network is built using Keras's `Sequential` API:

| Layer | Type | Output Shape | Activation | Details |
| :--- | :--- | :--- | :--- | :--- |
| **Input** | Flatten / Input | (784,) | None | Converts 28x28 pixel grid into 1D vector |
| **Hidden** | Dense | (128,) | ReLU | Extracts high-level non-linear feature maps |
| **Output** | Dense | (10,) | Softmax | Outputs class probabilities for digits 0–9 |

---

## 🛠️ Data Preprocessing

Pixel values in raw grayscale images range from `0` to `255`. Unscaled pixel values can cause exploding gradients and slow down convergence. 

To resolve this, min-max scaling is applied:
$$\text{X\_train} = \frac{\text{X\_train}}{255.0}$$
$$\text{X\_test} = \frac{\text{X\_test}}{255.0}$$

This normalizes feature values to a float range between `0.0` and `1.0`, significantly improving optimization stability and training velocity.

---

## 💻 Code Snippet

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense

# 1. Load Data
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

# 2. Preprocess & Normalize
X_train = X_train / 255.0
X_test = X_test / 255.0

X_train_flattened = X_train.reshape(len(X_train), 784)
X_test_flattened = X_test.reshape(len(X_test), 784)

# 3. Build Model Architecture
model = keras.Sequential([
    Input(shape=(784,)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# 4. Compile Model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 5. Train Model
model.fit(X_train_flattened, y_train, epochs=5)

# 6. Evaluate Model
test_loss, test_acc = model.evaluate(X_test_flattened, y_test)
print(f"Test Accuracy: {test_acc * 100:.2f}%")
```

---

## 📊 Results & Performance

| Epoch | Loss | Training Accuracy |
| :---: | :---: | :---: |
| 1 | 0.2562 | 92.75% |
| 2 | 0.1109 | 96.72% |
| 3 | 0.0764 | 97.68% |
| 4 | 0.0584 | 98.20% |
| **5** | **0.0453** | **98.62%** |

---

## 🚀 How to Run locally

### Prerequisites
Ensure you have Python 3.8+ installed along with the required libraries:

```bash
pip install tensorflow numpy matplotlib seaborn
```

### Steps
1. Clone this repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/Digit-Classification.git
   cd Digit-Classification
   ```
2. Open the notebook in Jupyter or VS Code:
   ```bash
   jupyter notebook "digit classification.ipynb"
   ```
3. Run all cells sequentially to train and evaluate the model.

---

## 📜 License
This project is open-source and available under the [MIT License](LICENSE).