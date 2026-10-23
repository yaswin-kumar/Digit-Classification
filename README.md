MNIST Digit Classification using TensorFlow & KerasAn end-to-end Deep Learning project that builds, trains, and evaluates a Neural Network to classify handwritten digits (0–9) from the classic MNIST dataset.Achieved 98%+ test accuracy by implementing proper data normalization and a multi-layer Artificial Neural Network (ANN).📌 Project OverviewHandwritten digit recognition is a benchmark problem in computer vision and machine learning. This project demonstrates how to construct an image classification pipeline using TensorFlow/Keras to achieve high precision on 28x28 grayscale digit images.Key HighlightsDataset: MNIST (60,000 training images, 10,000 testing images)Model Architecture: Multi-Layer Perceptron (Sequential ANN)Activation Functions: ReLU (Hidden Layer) & Softmax (Output Layer)Optimizer: AdamLoss Function: Sparse Categorical Cross-entropyAccuracy: ~98.6%🧠 Model ArchitectureThe neural network is built using Keras's Sequential API:LayerTypeOutput ShapeActivationDetailsInputFlatten / Input(784,)NoneConverts 28x28 pixel grid into 1D vectorHiddenDense(128,)ReLUExtracts high-level non-linear feature mapsOutputDense(10,)SoftmaxOutputs class probabilities for digits 0–9🛠️ Data PreprocessingPixel values in raw grayscale images range from 0 to 255. Unscaled pixel values can cause exploding gradients and slow down convergence.To resolve this, min-max scaling is applied:
$$\text{X\_train} = \frac{\text{X\_train}}{255.0}$$$$\text{X\_test} = \frac{\text{X\_test}}{255.0}$$This normalizes feature values to a float range between 0.0 and 1.0, significantly improving optimization stability and training velocity.💻 Code Snippetimport tensorflow as tf
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
📊 Results & PerformanceEpochLossTraining Accuracy10.256292.75%20.110996.72%30.076497.68%40.058498.20%50.045398.62%🚀 How to Run locallyPrerequisitesEnsure you have Python 3.8+ installed along with the required libraries:pip install tensorflow numpy matplotlib seaborn
StepsClone this repository:git clone https://github.com/YOUR_USERNAME/Digit-Classification.git
cd Digit-Classification
Open the notebook in Jupyter or VS Code:jupyter notebook "digit classification.ipynb"
Run all cells sequentially to train and evaluate the model.📜 LicenseThis project is open-source and available under the MIT License.# Digit-Classification
