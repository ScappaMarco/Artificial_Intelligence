#dependencies
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch import nn, save, load
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from neuralNetwork import ImageClassifier #classe della neural network definita da me

#definzizione dataset e dataloader per il set di dati per il training - Il full_train_dataset verrà poi suddiviso in train_dataset e validation_dataset
full_train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=ToTensor()) #train set - train = True
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=ToTensor()) #test set - train = False

#estraiamo dal training set il set di validation, facendo un 80% e 20% del training set
train_size = int(0.8 * len(full_train_dataset))
val_size = len(full_train_dataset) - train_size

train_dataset, validation_dataset = random_split(full_train_dataset, [train_size, val_size])

#definizione dei dataloader, con batch size di 64 per training e validation e 256 per il test
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True) 
validation_loader = DataLoader(validation_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False) 

mlp = ImageClassifier() #istanza della rerte neurale creata

#definizione delle loss function e della funzione di ottimizzazione
loss_func = nn.CrossEntropyLoss() #loss
optimizer = Adam(mlp.parameters(), lr=0.0001) #optimization

#adesso bisogna creare il ciclo per addestrare la rete neurale tramite n epoche
num_epochs = 10 #numero di training steps
for epoch in range(num_epochs):
    mlp.train()
    for images, labels in train_loader:
        predizione_train = mlp(images)
        loss = loss_func(predizione_train, labels)

        #backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    #qui va la fase di validation
    mlp.eval()
    loss_totale = 0.0
    correct = 0
    totali = 0

    #dopo ogni epoca di training il modello esegue una fase di validazione
    with torch.no_grad():
        for images, labels in validation_loader:
            predizione_validation = mlp(images)
            loss_totale += loss_func(predizione_validation, labels).item()
            _, valore_predizione = torch.max(predizione_validation, 1)
            correct += (valore_predizione == labels).sum().item()
            totali += labels.size(0)

    validation_los = loss_totale / len(validation_loader)
    accuracy = correct / totali

    print(f"Epoca numero {epoch} / {num_epochs}, con loss = {loss.item()}")
    print(f"Validation loss = {validation_los}, e accuracy = {accuracy}%")

#adesso comincia la parte di test