## Neural Networks From Scratch – Minimal Autograd Engine & MLP Implementation

This project is a **from-scratch neural network framework** inspired by the core ideas behind libraries like PyTorch, built without using any deep learning frameworks. It is designed to teach and demonstrate how automatic differentiation and basic neural network components work under the hood.

### Features
- **Custom Autograd Engine**  
  Implements a `Value` class that:
  - Stores scalar values and their gradients
  - Tracks the computation graph
  - Performs reverse-mode automatic differentiation via backpropagation
- **Neural Network Building Blocks**  
  - `Neuron`: single neuron with optional nonlinearity
  - `Layer`: fully connected layer with configurable activation
  - `MLP`: multi-layer perceptron that can stack layers into deep architectures
- **No Deep Learning Frameworks**  
  Relies only on Python’s standard library (`math`, `random`) for the core implementation.

### Example: Binary Iris Classification
The included `iris_classification.ipynb` notebook demonstrates:
- Loading and preprocessing the Iris dataset (binary classification: classes 0 and 1)
- Scaling features with `StandardScaler`
- Building an MLP:  
  ```python
  MLP = mininet.MLP(4, [16, 16, 1])
  
### Training Loop
- Forward pass  
- Mean squared error loss for targets in {-1, 1}  
- Backward pass with `.backward()`  
- Manual gradient descent parameter updates  

### Evaluation
- Accuracy  
- ROC curve and AUC  
- Confusion matrix  
- Loss curve over epochs  

### Tech Stack
- **Core Implementation**: Python standard library (`math`, `random`)  
- **Data Handling**: NumPy, pandas, scikit-learn  
- **Visualization**: matplotlib, seaborn  
