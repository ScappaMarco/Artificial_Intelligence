#dependencies
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch import nn, save, load
from torch.optim import Adam
from torch.utils.data import DataLoader

#defianiamo i due dataset principali: uno per il training e uno per il test
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=ToTensor()) #train set

test_set = datasets.MNIST(root='./data', train=False, download=True, transform=ToTensor()) #test set
