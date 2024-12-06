import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

training_data = datasets.FashionMNIST(
    root = "data",
    train = True,
    download = True,
    # Converts a PIL image or Numpy ndarray into a FloatTensor and scales the image's pixel
    # intensity values in the range [0., 1.]
    transform = ToTensor(),
    # This takes the labels and turns them into one-hot coded tensors, a tensor of size equal
    # to the number of labels, where the value at the index equal to the label is 1.
    target_transform = Lambda(lambda y: torch.zeros(10, dtype = torch.float).scatter_(0, torch.tensor(y), value = 1))
)

