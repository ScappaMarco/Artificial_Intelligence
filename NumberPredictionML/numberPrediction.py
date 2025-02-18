#dependencies
from torchvision import datasets
from PIL import Image
from torchvision.transforms import ToTensor, Compose, Resize, Grayscale
from torch import nn, save, load
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from neuralNetwork import ImageClassifier #classe della neural network definita da me
import matplotlib.pyplot as plt

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


print("INIZIO FASE DI TRAINING E VALIDATION")
print("---------------------------------------------------------------")

#adesso bisogna creare il ciclo per addestrare la rete neurale tramite n epoche
train_losses = [] #sarà utile dopo per mostrare il grafico della loss function
validation_accuarcies = []
num_epochs = 10 #numero di training steps
for epoch in range(num_epochs):
    mlp.train()
    epoch_train_loss = 0.0
    for train_images, train_labels in train_loader:
        predizione_train = mlp(train_images)
        loss = loss_func(predizione_train, train_labels)

        #backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_train_loss += loss.item()

    train_losses.append(loss.item())

    #qui va la fase di validation
    mlp.eval()
    loss_totale = 0.0
    correct = 0
    totali = 0

    #dopo ogni epoca di training il modello esegue una fase di validazione
    with torch.no_grad():
        for validation_images, validation_labels in validation_loader:
            predizione_validation = mlp(validation_images)
            loss_totale += loss_func(predizione_validation, validation_labels).item()
            _, valore_predizione = torch.max(predizione_validation, 1)
            correct += (valore_predizione == validation_labels).sum().item()
            totali += validation_labels.size(0)

    validation_los = loss_totale / len(validation_loader)
    accuracy = correct / totali

    validation_accuarcies.append(accuracy)

    print(f"Epoca numero {epoch + 1} / {num_epochs}, con loss = {loss.item()}")
    print(f"Validation loss = {validation_los}, e accuracy = {accuracy}%")
    print("---------------------------------------------------------------")

'''
#sezione dedicata alla visualizzazione della loss function e dell'accuracy
fig, ax1 = plt.subplots(figsize=(10, 6))

#grafico loss function
ax1.plot(range(1, num_epochs + 1), train_losses, label='Training Loss', color='blue', marker='o')
ax1.set_xlabel('Epoche')
ax1.set_ylabel('Loss', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

#grafico accuracy
ax2 = ax1.twinx()
ax2.plot(range(1, num_epochs + 1), validation_accuarcies, label='Training Accuracy', color='red', marker='s')
ax2.set_ylabel('Accuracy (%)', color='red')
ax2.tick_params(axis='y', labelcolor='red')

plt.title('Training Loss e Accuracy per Epoca')
fig.legend(loc='upper right')

plt.grid(True)

plt.show()
'''

print("INIZIO FASE DI TESTING")
print("---------------------------------------------------------------")
#adesso comincia la parte di test
mlp.eval()
test_loss = 0.0
correct_test = 0
test_totali = 0

with torch.no_grad():
    for test_images, test_labels in test_loader:
        prediction_test = mlp(test_images)
        test_loss += loss_func(prediction_test, test_labels).item()
        _, valore_predizione_test = torch.max(prediction_test, 1)
        correct_test += (valore_predizione_test == test_labels).sum().item()
        test_totali += test_labels.size(0)

    test_accuracy = correct_test / test_totali
    print(f"Test accuracy = {test_accuracy * 100:.2f}")

#salvataggio modello con wb = write binary
with open('classification_model.pt', 'wb') as f:
    save(mlp.state_dict(), f)

#codice di prova del modello con delle foto prese dal dataset di MNIST
'''
transform = Compose([
    Resize((28, 28)),
    Grayscale(num_output_channels=1),  
    ToTensor()
])

with open('classification_model.pt', 'rb') as f:
    mlp.load_state_dict(load(f))
    img = Image.open('Examples/image_7.png')
    img_tensor = transform(img).unsqueeze(0)

    print(mlp(img_tensor))
'''