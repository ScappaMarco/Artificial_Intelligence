#dependencies
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch import nn, save, load
from torch.optim import Adam
from torch.utils.data import DataLoader
import Neuralnetwork #classe della neural network definita da me

#definzizione dataset e dataloader per il set di dati per il training
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=ToTensor()) #train set
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True) #fissiamo la dimensione dei batch a 32 per volta

#definzione dataset e dataloader per il set di dati per il test
test_set = datasets.MNIST(root='./data', train=False, download=True, transform=ToTensor()) #test set
test_loader = DataLoader(test_set, batch_size=32, shuffle=True) #fissiamo la dimensione dei batrch a 32

mlp = Neuralnetwork() #istanza della rerte neurale creata

#definizione delle loss function e della funzione di ottimizzazione
loss_func = nn.CrossEntropyLoss() #loss
optimizer = Adam(mlp.parameters(), lr=0.0001) #optimization

#adesso bisogna creare il ciclo per addestrare la rete neurale tramite n epoche
num_epochs = 10 #numero di training steps
for epoch in range(num_epochs):
    mlp.train()
    for images, labels in train_loader:
        prediction = mlp(images)
        loss = loss_func(prediction, labels)

        #backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoca numero {epoch}, con loss = {loss.item()}")
    




