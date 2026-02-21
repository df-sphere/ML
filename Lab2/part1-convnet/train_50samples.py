"""
Training script for 50 samples CIFAR-10 overfitting experiment
"""

from modules import ConvNet
from optimizer import SGD
from cs7643.solver import Solver
from data import get_CIFAR10_data
import matplotlib.pyplot as plt

# Load CIFAR-10 data
root = './data/cifar10/cifar-10-batches-py'
X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data(root)

print(f'Number of training samples: {len(X_train)}')
print(f'Training on only 50 samples to verify model can overfit')

# Define network architecture
model_list = [
    dict(type='Conv2D', in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),
    dict(type='ReLU'),
    dict(type='MaxPooling', kernel_size=2, stride=2),
    dict(type='Linear', in_dim=8192, out_dim=10)
]

criterion = dict(type='SoftmaxCrossEntropy')
model = ConvNet(model_list, criterion)

# Create optimizer with momentum
optimizer = SGD(model, learning_rate=0.0001, reg=0.001, momentum=0.9)

# Create trainer
trainer = Solver()

# Train on only 50 samples
print('\nStarting training...')
loss_history, train_acc_history = trainer.train(
    X_train[:50], y_train[:50], model, batch_size=10, num_epochs=10,
    verbose=True, optimizer=optimizer
)

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(train_acc_history)
plt.legend(['train'], loc='upper left')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title('Training Accuracy on 50 CIFAR-10 Samples')
plt.grid(True)
plt.savefig('train.png')
print(f'\nPlot saved to train.png')
print(f'Final training accuracy: {train_acc_history[-1]:.4f}')
print(f'Target accuracy: ~0.9')
