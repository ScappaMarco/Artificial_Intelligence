from torch import nn

#Definizione modello di classificazione di tipo multiclass
class ImageClassifier(nn.Module):
    #definzione rete neurale tramile MLP (strati lineari)
    def __init__(self):
        super().__init__()
        '''
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),  # Conv1: (28x28) → (28x28)
            nn.ReLU(),  
            
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),  # Conv2: (28x28) → (28x28)
            nn.ReLU(),  
            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),  # Conv3: (28x28) → (28x28)
            nn.ReLU(),  
            
            nn.Flatten(),  # Appiattisce i dati (28x28x128 → 100352)
            nn.Linear(28 * 28 * 128, 10)  # Layer finale per classificazione (100352 → 10 classi)
        )
        '''
        self.model = nn.Sequential(
            nn.Flatten(), #tramiote la funzione Flatten() appiattisco l'immagine 28x28 in un vettore 
            nn.Linear(28*28, 128), #primo strato connesso (28x28 neuroni di input, 128 neuroni di output)
            nn.ReLU(), #funziine di attivazione Rectified Liner Unit 
            nn.Linear(128, 64), #secondo strato (128 neuroni di input (output primo strato) e 64 neuroni di output)
            nn.ReLU(), #funziine di attivazione Rectified Liner Unit 
            nn.Linear(64, 10) #terzo e ultimo strato, in cui abbiamo 64 neuroni di input (output strato precedente) e 10 neuroni di output, che corrispondono alle 10 classi disponibili (0-9)
        )
    
    def forward(self, x):
        return self.model(x)