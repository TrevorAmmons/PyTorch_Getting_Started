import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Define the Class
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(in_features = 28 * 28, out_features = 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

# Get Device for Training
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backens.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# Create an instance of NeuralNetwork and move it to the device.
model = NeuralNetwork().to(device)
print(model)

# Pass the model input data
input_data = torch.rand(1, 28, 28, device = device)
logits = model(input_data)
predicted_probability = nn.Softmax(dim = 1)(logits)
y_pred = predicted_probability.argmax(1)
print(f"Predicted class: {y_pred}")

# Model Layers
input_image = torch.rand(3, 28, 28)
print(input_image.size())

# Convert each 2D 28 x 28 image into a contiguous array of 784 pixel values
# The minibatch dimension (at dim=0) is maintained
flatten = nn.Flatten()
flat_image = flatten(input_image)
print(flat_image.size())

# Pass through linear layer
layer1 = nn.Linear(in_features = 28 * 28, out_features = 20)
hidden1 = layer1(flat_image)
print(hidden1.size())

# Non-linear activations used between linear layers
print(f"Before ReLU: {hidden1}\n\n")
hidden1 = nn.ReLU()(hidden1)
print(f"After ReLU: {hidden1}\n\n")

# nn.Sequential is an ordered container of modules. The data is passed through all the modules in the same order as defined.
seq_modules = nn.Sequential(
    flatten,
    layer1,
    nn.ReLU(),
    nn.Linear(20, 10)
)

input_image = torch.rand(3, 28, 28)
logits = seq_modules(input_image)

# nn.Softmax converts linear values of [-inf, inf] to [0, 1] for the probability of each class
softmax = nn.Softmax(dim = 1)
pred_probab = softmax(logits)

# Model Parameters
print(f"Model structure: {model}\n\n")
for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values: {param[:2]}\n")