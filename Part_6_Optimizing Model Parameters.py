import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

training_data = datasets.FashionMNIST(
    root = "data",
    train = True,
    download = True,
    transform = ToTensor()
)

test_data = datasets.FashionMNIST(
    root = "data",
    train = False,
    download = True,
    transform = ToTensor()
)

train_dataloader = DataLoader(training_data, batch_size = 64)
test_dataloader = DataLoader(test_data, batch_size = 64)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork()

# Hyperparameters

# Hyperparameters are adjustable parameters that let you control the model optimization process.
# Different values can impact model training and convergence rates.abs
# Number of Epochs - the number of times to iterate over the dataset
# Batch Size - the number of data samples propagated through the network before teh parameters are updated.
# Learning Rate - how much to update models parameters at each batch/epoch. Smaller values yield slow learning speed,
#                 while large values may result in unpredictable behavior during training.abs

learning_rate = 1e-3
batch_size = 64
epochs = 5

# Optimization loop

# Each iteration of the optimization loop is called an epoch, it has two main parts including
# (1) The Train Loop - iterate over the training dataset and try to converge to optimal parameters.
# (2) The Validation/Test Loop - iterate over teh test dataset to check if model performance is improving.

# Loss Function

# nn.MSELoss (Mean Square Error) for regression tasks
# nn.NLLLoss (Negative Log Likelihood) for classification
# nn.CrossEntropyLoss combines nn.LogSoftmaz and nn.NLLLoss

# pass our model's output logits to nn.CrossEntropyLoss, normalizing the logits and computing the prediction error
loss_fn = nn.CrossEntropyLoss()

# Optimizer

# We will use Stochastic Gradient Descent. Optimization logic is encaspulated in the optimizer object.
# We will pass the learning rate hyperparameter.
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

# In the training loop, optimization happens in three steps
# (1) Call optimizer.zero_grad() to reset the gradients of model parameters. Gradients by default add up; to prevent double-counting, we explicity zero them at each iteration
# (2) Backpropagate the prediction loss with a call to loss.backward(). PyTorch deposits the gradients of the loss w.r.t each parameter.
# (3) Once we have our gradients, we call optimizer.step() to adjust the parameters by the gradients collected in teh backward pass.

# Full Implementation

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices.
    model.train()

    for batch, (X, y) in enumerate(dataloader):
        # Copmute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item, batch * batch_size + len(X)
            print(f"Loss: {loss} [{current:>5d}/{size:>5d}]")
    
def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaulating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # This also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad = True
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

# We initialize the loss function and optimizer, and pass it to train_loop and test_loop.
# Feel free to increase the number of epochs to track the model's improving performance.

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

epochs = 10

for t in range(epochs):
    print(f"Epoch {t + 1}\n------------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)

print("Done!")