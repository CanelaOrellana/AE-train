import numpy as np
from typing import List, Dict, Any
import seaborn as sns
sns.set(color_codes=True)
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import time

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_squared_error
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import r2_score

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)



'''
A continuación se presentan funciones auxiliares para el funcionamiento de las clases de AutoEncoders
'''

def print_plus(text, level = 0, color = "n", end = "\n", limpiar = True):
    if color == "r": inicio = "\33[91m"     #rojo
    elif color == "g": inicio  = "\33[92m"  #verde
    elif color == "p": inicio = "\33[95m"   #morado
    elif color == "b": inicio  = "\33[1m"   #azul
    elif color == "n": inicio  = "\33[0m"   #negrita
    elif color == "y":  inicio = "\033[93m" #amarillo
    salto = " "* level
    if limpiar:
        text = f"{text}{' ' * (110 - len(text))}"
    print(f"{salto}{inicio}{text}\33[0m", end=end)


def ftrain(model, dataloader, optimizer, criterion):
  epoch_loss = 0
  epoch_r2 = 0
  model.train() #poner el modelo el modo entrenamiento
  #Training loop
  for batch in dataloader:
    batch = batch[0]    # input data y label
    optimizer.zero_grad() # limpiar gradientes
             
    output = model(batch) #output
    loss = criterion(output, batch) # loss
    r2 = r2_score(output[-1].cpu().detach().numpy(), batch[-1].cpu().detach().numpy())
    loss.backward() # Computar gradientes
    
    #EN CASO DE UTILIZAR CLIPPING DEBERÍA IR AQUÍ
    optimizer.step() # step
    epoch_loss += loss.item()
    epoch_r2 += r2
    
  if len(dataloader) > 0:
    return epoch_loss / len(dataloader), epoch_r2 / len(dataloader)
  else:
    raise ValueError("El dataloader está vacío. No hay muestras para evaluar.")


def fevaluate(model, dataloader, criterion):
  epoch_loss = 0
  epoch_r2 = 0
  model.eval() #poner el modelo en modo evaluación
  with torch.no_grad():
    for batch in dataloader:
      batch = batch[0]  # input data y label

      output = model(batch) # output
      loss = criterion(output, batch)
      r2 = r2_score(output[-1].cpu().detach().numpy(), batch[-1].cpu().detach().numpy())

      epoch_loss += loss.item()
      epoch_r2 += r2

  if len(dataloader) > 0:
    return epoch_loss / len(dataloader), epoch_r2 / len(dataloader)
  else:
    raise ValueError("El dataloader está vacío. No hay muestras para evaluar.")

'''
Clase AutoEncoder
'''
#AE Caso Base
class AutoEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, 
    h_layer_sizes: List[int], 
    timesteps: int, 
    learning_rate: float = 0.001, 
    dropout: float = 0.2, 
    n_features_in = 9, 
    batch_size: int = 32, 
    optim: str = 'Adam',
    epochs: int = 50, 
    verbose: int = 2,
    patience: int = 5):
        self.h_layer_sizes = h_layer_sizes #Lista con tamaños de las capas ocultas
        self.learning_rate = learning_rate #nro
        self.dropout = dropout #nro
        self.batch_size = batch_size #nro
        self.epochs = epochs #nro
        self.activation = '' #torch.nn.Sigmoid()
        self.optim = optim
        self.verbose = verbose 
        self.timesteps = timesteps
        self.n_features_in = n_features_in
        self.patience = patience #para early stopping
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Verificar CUDA
        self.model_ = self.build_model().to(self.device)  # Mover el modelo a la GPU
        
    def build_model(self):
        '''
        Método que permite construir la arquitectura del modelo
        '''
        class Encoder(nn.Module): #Se define el Encoder
            def __init__(self, timesteps, n_features_in, h_layer_sizes, dropout):
                super(Encoder, self).__init__()
                self.timesteps = timesteps
                self.n_features_in = n_features_in
                layers = []
                input_size = n_features_in
                for size in h_layer_sizes:
                    layers.append(nn.LSTM(input_size = input_size,# if i ==0 else h_layer_sizes[i-1],
                                        hidden_size = size, 
                                        batch_first=True, 
                                        dropout=0))
                    layers.append(nn.Dropout(dropout))
                    input_size = size

                self.model = nn.ModuleList(layers)

            def forward(self, x):
                for layer in self.model:
                    x, _ = layer(x) if isinstance(layer, nn.LSTM) else (layer(x), None)
                return x

        class Decoder(nn.Module): #Se define el Decoder
            def __init__(self, timesteps, h_layer_sizes, dropout, n_features_in, activation):
                super(Decoder, self).__init__()
                self.timesteps = timesteps
                self.n_features_in = n_features_in
                self.activation = activation
                layers = []
                input_size = h_layer_sizes[0]

                for size in h_layer_sizes:
                    layers.append(nn.LSTM(input_size = input_size,# if i ==0 else h_layer_sizes[i-1],
                                        hidden_size = size, 
                                        batch_first=True, 
                                        dropout=0))
                    layers.append(nn.Dropout(dropout))
                    input_size = size

                layers.append(nn.Linear(input_size, n_features_in))
                self.model = nn.ModuleList(layers)

            def forward(self, x):
                for i, layer in enumerate(self.model):
                    if isinstance(layer, nn.LSTM):
                        x, _ = layer(x)
                    else:
                        x = layer(x)
                        if self.activation:  # Aplicar la función de activación en caso de estar definida
                            x = self.activation(x)
    
                return x

        class Autoencoder(nn.Module): #Se define el AutoEncoder completo
            def __init__(self, h_layer_sizes, timesteps, dropout, n_features_in, activation):
                super(Autoencoder, self).__init__()
                self.timesteps = timesteps
                self.n_features_in = n_features_in
                self.encoder = Encoder(timesteps, n_features_in, h_layer_sizes, dropout)
                self.decoder = Decoder(timesteps, h_layer_sizes[::-1], dropout, n_features_in, activation)

            def forward(self, x):
                encoded = self.encoder(x)
                reconstructed = self.decoder(encoded)
                return reconstructed

        model = Autoencoder(self.h_layer_sizes, self.timesteps, self.dropout, self.n_features_in, self.activation)
        self.loss_fn = nn.MSELoss()
        if self.optim == 'Adam':
            self.optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        if self.optim == 'AdamW':
            self.optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate)
        if self.optim == 'RAdam':
            self.optimizer = torch.optim.RAdam(model.parameters(), lr=self.learning_rate)

        return model

    
    def fit(self, X, p_train = 0.9, y=None):
        #Datos entrenamiento
        X_tensor = torch.FloatTensor(X).to(self.device)  # Mover los datos a la GPU
        dataset = torch.utils.data.TensorDataset(X_tensor)

        #Se define el índice para dividir los datos de entrenamiento y validación
        train_size = int(p_train * len(dataset))

        #Se crean los subsets sin desordenar los datos
        train_dataset = torch.utils.data.Subset(dataset, list(range(0, train_size)))
        val_dataset = torch.utils.data.Subset(dataset, list(range(train_size, len(dataset))))

        #Se crean los DataLoaders para entrenamiento y validación
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False, drop_last=True)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False)

        #Definiciones necesarias para early stopping
        best_loss = float('inf')
        paciencia = 0
        
        #Prints de información
        if self.verbose == 0:
            # print_plus('Entrenando', color = "p", end="\r")
            print_plus(f'h_layer_sizes: {self.h_layer_sizes}, lr: {self.learning_rate}, drop out: {self.dropout}, patience: {self.patience}', end=" ")
        ts = time.time()
        
        train_losses = []
        val_losses = []
        train_r2 = []
        val_r2 = []

        #Bucle por épocas
        for epoch in range(self.epochs):
            
            #Bucle entrenamiento
            loss_train, r2_train = ftrain(self.model_, train_dataloader, self.optimizer, self.loss_fn)
            train_losses.append(loss_train) #append del loss promedio de la época
            train_r2.append(r2_train) #append del r2 promedio de la época
            #Bucle validación
            loss_val, r2_val = fevaluate(self.model_, val_dataloader, self.loss_fn)
            val_losses.append(loss_val) #append del loss promedio de la época
            val_r2.append(r2_val) #append del r2 promedio de la época
            
            # Monitoreo early stopping
            if loss_val < best_loss:
                best_loss = loss_val
                paciencia = 0
            else:
                paciencia += 1
                if paciencia >= self.patience:
                    #Se imprime en qué época se detuvo el entrenamiento por early stopping
                    print_plus(f"Época: {epoch + 1}/{self.epochs}", color = "p", end="\n", limpiar=False)
                    te = time.time()
                    print_plus(f"time: {te-ts}", color = "y", end="\r")     
                    break

            if self.verbose > 0 and epoch % self.verbose == 0:
                print(f'Epoch {epoch}/{self.epochs}, Loss validación: {loss_val}, R^2 validación: {r2_val}, R^2 entrenamiento: {r2_train}')

        if self.verbose == 5:
            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16,6))
            axes[0].plot(range(1, len(train_losses) + 1), train_losses, label='Loss entrenamiento', color='b')
            axes[0].plot(range(1, len(train_losses) + 1), val_losses, label='Loss validación', color='magenta')
            axes[0].set_title('Función de Costo')
            axes[0].set_xlabel('Épocas')
            axes[0].set_ylabel('Función de costo')
            axes[0].legend()

            axes[1].plot(range(1, len(train_r2) + 1), train_r2, label='Entrenamiento', color='b')
            axes[1].plot(range(1, len(train_r2) + 1), val_r2, label='Validación', color='magenta')
            axes[1].set_title('R^2 por época')
            axes[1].set_xlabel('Épocas')
            axes[1].set_ylabel('R^2')
            axes[1].legend()

            plt.show()
            plt.close(fig)
        
        te = time.time()
        print_plus(f"time: {te-ts}", color = "y", end="\r")
        return self

    def predict(self, X):
        self.model_.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)  # Mover los datos a la GPU
            return self.model_(X_tensor).cpu().numpy()  # Mover el resultado de nuevo a la CPU para la conversión a numpy

    def score(self, X) -> float:
        X_pred = self.predict(X)
        X = X.reshape((X.shape[0], -1))
        X_pred = X_pred.reshape((X_pred.shape[0], -1))
        r2 = r2_score(X, X_pred)
        mse = mean_squared_error(X, X_pred)
        return 1 - mse

    def save_weights(self, filepath='autoencoder_weights.pth'):
        torch.save(self.model_.state_dict(), filepath)
        print_plus(f"Pesos guardados exitosamente en '{filepath}'", color = "g", end="\r")


    def transform(self, X) -> np.ndarray:
        self.model_.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            encoded = self.model_.encoder(X_tensor)
            return encoded.cpu().numpy()
        # raise NotImplementedError("Transform method is not implemented in PyTorch version.")

    def inverse_transform(self, X_tf: np.ndarray):
        self.model_.eval()
        with torch.no_grad():
            X_tf_tensor = torch.FloatTensor(X_tf).to(self.device)
            decoded = self.model_.decoder(X_tf_tensor)
            return decoded.cpu().numpy()
        # raise NotImplementedError("Inverse transform method is not implemented in PyTorch version.")

#Wrapper CB
class AutoEncoderWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, h_layer_sizes, timesteps, learning_rate, dropout, n_features_in, batch_size, optim, epochs, verbose, patience):
        self.h_layer_sizes = h_layer_sizes
        self.timesteps = timesteps
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.n_features_in = n_features_in
        self.batch_size = batch_size
        self.optim = optim
        self.epochs = epochs
        self.verbose = verbose
        self.patience = patience
        self.model_ = None

    def _create_autoencoder(self):
        return AutoEncoder(
            h_layer_sizes=self.h_layer_sizes,
            timesteps=self.timesteps,
            learning_rate=self.learning_rate,
            dropout=self.dropout,
            batch_size=self.batch_size,
            optim=self.optim,
            epochs=self.epochs,
            verbose=self.verbose,
            n_features_in=self.n_features_in,
            patience = self.patience,
        ) 
        
    def fit(self, X, p_train=0.9, y=None):
        self.model_ = self._create_autoencoder()
        self.model_.fit(X, p_train)
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def score(self, X, y=None):
        return self.model_.score(X)

    def save_weights(self, filepath = 'AE.pth'):
        return self.model_.save_weights(filepath)

'''
Clase ClassifierAutoEncoder
'''
# P R O X I M A M E N T E