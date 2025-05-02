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
Clase JointAEClassifier
'''

def f_train(model, dataloader, optimizer, criterion_r, criterion_c, alpha):
  epoch_loss = 0
  epoch_rmse = 0
  running_corrects = 0
  model.train()
  #Training loop
  scaler = torch.cuda.amp.GradScaler()
  for batch_X, batch_y in dataloader:
    if batch_X.dim() == 2:  # Verifica si faltan dimensiones en los datos
        batch_X = batch_X.unsqueeze(1)  # Añade dimensión en el eje de timesteps
    optimizer.zero_grad() # limpiar gradientes

    with torch.autocast(device_type="cuda"):
      out_r, out_c = model(batch_X) #output
      out_c = out_c.float()
      _, preds = torch.max(out_c, 1)
      loss_r = criterion_r(out_r, batch_X)
      loss_c = criterion_c(out_c, batch_y)
      loss = alpha*loss_r + (1-alpha)*loss_c # loss total
      # print(f'Loss reconstrucción: {loss_r}')
      # print(f'Loss clasificación: {loss_c}')
      # print(f'Loss compuesta: {loss}')

      # rmse = torch.sqrt(loss_r)
    
    loss.backward() # Computar gradientes
    #EN CASO DE UTILIZAR CLIPPING DEBERÍA IR AQUÍ
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step() # step
    # for name, param in model.named_parameters():
    #   if torch.isnan(param).any():
    #     print(f"Parámetro {name} contiene NaN después de optimizer.step()")

    epoch_loss += loss.item()
    epoch_rmse += torch.sqrt(loss_r).item()
    # running_corrects += torch.sum((out_c == batch_y).float())
    running_corrects += torch.sum(preds == batch_y)

  if len(dataloader) > 0:
      return epoch_loss / len(dataloader), epoch_rmse / len(dataloader), running_corrects.double() / len(dataloader.dataset)
  else:
    raise ValueError("El dataloader está vacío. No hay muestras para evaluar.")
        
  # return epoch_loss / len(dataloader), epoch_rmse / len(dataloader)


def f_evaluate(model, dataloader, criterion_r, criterion_c, alpha):
  epoch_loss = 0
  epoch_rmse = 0
  running_corrects = 0

  model.eval()    
  with torch.no_grad(): #disable the autograd engine (save computation and memory)
    for batch_X, batch_y in dataloader:
      with torch.autocast(device_type="cuda"):
        out_r, out_c = model(batch_X) #output
        # out_c = torch.argmax(out_c, dim=1).float()
        out_c = out_c.float()

        _, preds = torch.max(out_c, 1)
        loss_r = criterion_r(out_r, batch_X)
        loss_c = criterion_c(out_c, batch_y)
        loss = alpha*loss_r + (1-alpha)*loss_c # loss total

        rmse = torch.sqrt(loss)

      epoch_loss += loss.item()
      epoch_rmse += torch.sqrt(loss_r).item()
      # running_corrects += torch.sum((out_c == batch_y).float())
      running_corrects += torch.sum(preds == batch_y)
  if len(dataloader) > 0:
    return epoch_loss / len(dataloader), epoch_rmse / len(dataloader), running_corrects.double() / len(dataloader.dataset)
  else:
    raise ValueError("El dataloader está vacío. No hay muestras para evaluar.")

  # return epoch_loss / len(dataloader), epoch_rmse / len(dataloader)


class JointAEClassifier(BaseEstimator, TransformerMixin):
    def __init__(self, 
    h_layer_sizes: List[int], 
    timesteps, 
    learning_rate: float = 0.001, 
    learning_rate_c: float = 0.005,
    dropout_ae: float = 0.2,
    dropout_c: float = 0.2, 
    n_features_in = 9, 
    batch_size: int = 32,
    alpha: float = 0.5, 
    optim: str = 'Adam',
    stat: str = 'median',
    epochs: int = 50, 
    verbose: int = 2,
    patience: int = 5,
    n_classes: int = 12):
        self.h_layer_sizes = h_layer_sizes #Lista con tamaños de las capas ocultas
        self.learning_rate = learning_rate #nro(0,1)
        self.learning_rate_c = learning_rate_c #nro (0,1)
        self.dropout_ae = dropout_ae #nro(0,1)
        self.dropout_c = dropout_c #nro(0,1)
        self.batch_size = batch_size #nro
        self.alpha = alpha #nro(0,1)
        self.epochs = epochs #nro
        self.activation = '' #torch.nn.Sigmoid()
        self.optim = optim
        self.stat = stat
        self.verbose = verbose 
        self.timesteps = timesteps
        self.n_features_in = n_features_in
        self.patience = patience #para early stopping
        self.n_classes = n_classes #nro
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Verificar CUDA
        self.model_ = self.build_model().to(self.device)  # Mover el modelo a la GPU

    def build_model(self):
        class Encoder(nn.Module):
            def __init__(self, timesteps, n_features_in, h_layer_sizes, dropout_ae):
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
                    layers.append(nn.Dropout(dropout_ae))
                    input_size = size

                self.model = nn.ModuleList(layers)

            def forward(self, x):
                for layer in self.model:
                    if isinstance(layer, nn.LSTM):
                        x, _ = layer(x)
                    else:
                        x = layer(x)
                return x

        class Decoder(nn.Module):
            def __init__(self, timesteps, h_layer_sizes, dropout_ae, n_features_in, activation):
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
                    layers.append(nn.Dropout(dropout_ae))
                    input_size = size

                layers.append(nn.Linear(input_size, n_features_in))
                self.model = nn.ModuleList(layers)

            def forward(self, x):
                for layer in self.model:
                    if isinstance(layer, nn.LSTM):
                        x, _ = layer(x)
                    elif isinstance(layer, nn.Linear):
                        x = layer(x)
                        if self.activation:
                            x = self.activation(x)
                    else:
                        x = layer(x)
    
                return x
        # NEWW
        class Classifier(nn.Module):
            def __init__(self, input_size, n_classes, dropout_c, timesteps, stat):
                super(Classifier, self).__init__()
                self.drout = nn.Dropout(dropout_c)
                self.stat = stat
                self.activation = nn.ReLU()
                self.fc1 = nn.Linear(input_size,int(input_size // 1.5))
                self.fc2 = nn.Linear(int(input_size // 1.5), int(input_size//2.25))
                self.fc3 = nn.Linear(int(input_size//2.25), n_classes) # Capa de salida

            def forward(self, x):
                # print(f'x shape: {x.shape}')
                if isinstance(x, nn.LSTM):
                        x, _ = x
                        # print(f'x shape2: {x.shape}')
                if self.stat == 'mean':
                    x = torch.mean(x, dim=1) #promedio
                if self.stat == 'std':
                    x = torch.std(x, dim=1) #desviación estandar
                if self.stat == 'median':
                    x, _ = torch.median(x,dim=1) #mediana
                if self.stat == 'var':
                    x = torch.var(x, dim=1) #varianza
                if self.stat == 'min':
                    x, _ = torch.min(x, dim=1) #mínimo
                if self.stat == 'max':
                    x, _ = torch.max(x, dim=1) #máximo
                if self.stat == 'mode':
                    x, _ = torch.mode(x, dim=1) #moda
                if self.stat == '':
                    x = x[:,-1,:]  # Tomar el último timestep
                # print(f'x shape3: {x.shape}')
                x = self.activation(self.fc1(x))
                x = self.activation(self.fc2(x))
                x = self.drout(x)
                x = self.fc3(x)
                return x

        class AEClassifier(nn.Module):
            def __init__(self, h_layer_sizes, timesteps, dropout_ae, dropout_c, n_features_in, activation, n_classes, batch_size, stat):
                super(AEClassifier, self).__init__()
                self.timesteps = timesteps
                self.n_features_in = n_features_in
                self.encoder = Encoder(timesteps, n_features_in, h_layer_sizes, dropout_ae)
                self.decoder = Decoder(timesteps, h_layer_sizes[::-1], dropout_ae, n_features_in, activation)
                self.classifier = Classifier(h_layer_sizes[-1], n_classes, dropout_c, timesteps, stat)  # Último tamaño de capa oculta como input

            def forward(self, x):
                encoded = self.encoder(x)
                # print_plus(f'Shape salida del encoder: {encoded.shape}', end='\n' )
                reconstructed = self.decoder(encoded)
                classified = self.classifier(encoded)
                # print_plus(f'Shape classified: {classified.shape}', color="r")
                return reconstructed, classified

        model = AEClassifier(self.h_layer_sizes, self.timesteps, self.dropout_ae, self.dropout_c, self.n_features_in, self.activation, self.n_classes, self.batch_size, self.stat)
        self.loss_fn_r = nn.MSELoss()
        self.loss_fn_c = nn.CrossEntropyLoss()
        if self.optim == 'Adam':
            # self.optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
            self.optimizer = torch.optim.Adam([
                {'params':model.encoder.parameters(), 'lr':self.learning_rate},
                {'params':model.decoder.parameters(), 'lr':self.learning_rate},
                {'params':model.classifier.parameters(), 'lr':self.learning_rate_c}])
        if self.optim == 'AdamW':
            # self.optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate)
            self.optimizer = torch.optim.AdamW([
                {'params':model.encoder.parameters(), 'lr':self.learning_rate},
                {'params':model.decoder.parameters(), 'lr':self.learning_rate},
                {'params':model.classifier.parameters(), 'lr':self.learning_rate_c}])
        if self.optim == 'RAdam':
            # self.optimizer = torch.optim.RAdam(model.parameters(), lr=self.learning_rate)
            self.optimizer = torch.optim.RAdam([
                {'params':model.encoder.parameters(), 'lr':self.learning_rate},
                {'params':model.decoder.parameters(), 'lr':self.learning_rate},
                {'params':model.classifier.parameters(), 'lr':self.learning_rate_c}])
        # La siguiente linea reemplaza los ifs en caso de que se quiera eliminar el uso de 2 learning rates distintos    
        # self.optimizer = getattr(torch.optim, self.optim)(model.parameters(), lr=self.learning_rate)

        #Uso de scheduler para modificar el learning rate 
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', factor = 0.5, patience=4)
        return model

    def fdataloader(self, X, y, p_train=0.9):
        # Mover datos a la GPU
        X_tensor = torch.FloatTensor(X).to(self.device)  # Mover los datos a la GPU
        y_tensor = torch.LongTensor(y).to(self.device)
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)

        unique_labels = torch.unique(y_tensor)

        train_indices = []
        val_indices = []

        for label in unique_labels:
            label_indices = torch.where(y_tensor == label)[0]
            label_size = len(label_indices)
            train_size = int(p_train * label_size)

            train_indices.extend(label_indices[:train_size].tolist())
            val_indices.extend(label_indices[train_size:].tolist())
        
        #Crear datasets en torch    
        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        val_dataset = torch.utils.data.Subset(dataset, val_indices)

        # Crear dataloaders para train y test 
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False)

        # Retorna o almacena los datasets divididos
        return train_dataloader, val_dataloader

    
    def fit(self, X, y, p_train = 0.9):
        #Obtener dataloaders        
        train_dataloader, val_dataloader = self.fdataloader(X, y, p_train)
        
        #Definiciones necesarias para early stopping
        best_loss = float('inf')
        paciencia = 0
        
        #Prints de información
        if self.verbose == 0:
            print_plus(f'h_layer_sizes: {self.h_layer_sizes}, lr: {self.learning_rate}, drop out: {self.dropout_c}, patience: {self.patience}, alpha: {self.alpha}', end=" ")
        ts = time.time()
        
        train_losses = []
        train_rmses = []
        train_acc = []
        val_losses = []
        val_rmses = []
        val_acc = []

        #Bucle por épocas
        for epoch in range(self.epochs):
            
            # #Bucle entrenamiento
            loss_train, rmse_train, acc_train = f_train(self.model_, train_dataloader, self.optimizer, self.loss_fn_r, self.loss_fn_c, self.alpha)
            train_losses.append(loss_train) #append del loss promedio de la época
            train_rmses.append(rmse_train) #append del rmse promedio de la época
            train_acc.append(acc_train.item())
            #Bucle validación
            loss_val, rmse_val, acc_val= f_evaluate(self.model_, val_dataloader, self.loss_fn_r, self.loss_fn_c, self.alpha)
            # print(f'Loss train: {loss_train}')
            self.scheduler.step(loss_val)
            # Prints para ver cómo cambian los learning rate
            # for i, param_group in enumerate(self.optimizer.param_groups):
            #     print(f"Epoch {epoch+1}, Param Group {i}: Learning rate is {param_group['lr']}")
            val_losses.append(loss_val) #append del loss promedio de la época
            val_rmses.append(rmse_val) #append del rmse promedio de la época
            val_acc.append(acc_val.item())

            # Monitoreo (early stopping)
            if loss_val < best_loss:
                # print('hi')
                best_loss = loss_val
                paciencia = 0
            else:
                paciencia += 1
                # print(paciencia)
                if paciencia >= self.patience:
                    print_plus(f"Época: {epoch + 1}/{self.epochs}", color = "p", end="\n", limpiar=False)         
                    break

            if self.verbose > 0 and epoch % self.verbose == 0:
                print(f'Epoch {epoch}/{self.epochs}, Loss: {loss_val}, Accuracy en validación: {acc_val}')

        if self.verbose == 5:
            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16,6))
            axes[0].plot(range(1, len(train_losses) + 1), train_losses, label='Loss entrenamiento', color='b')
            axes[0].plot(range(1, len(train_losses) + 1), val_losses, label='Loss validación', color='c')
            axes[0].set_title('Función de Costo')
            axes[0].set_xlabel('Épocas')
            axes[0].set_ylabel('Función de costo')
            axes[0].legend()

            axes[1].plot(range(1, len(train_acc) + 1), train_acc, label='Accuracy entrenamiento', color='b')
            axes[1].plot(range(1, len(train_acc) + 1), val_acc, label='Accuracy validación', color='c')
            axes[1].set_title('Accuracy')
            axes[1].set_xlabel('Épocas')
            axes[1].set_ylabel('Accuracy')
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
            reconstructed, classified = self.model_(X_tensor)
            reconstructed = reconstructed.cpu().numpy()
            classified = classified.argmax(dim=1).cpu().numpy()
            return reconstructed, classified  # Mover el resultado de nuevo a la CPU para la conversión a numpy

    def score(self, X, y) -> float:
        X_pred, X_clas = self.predict(X)
        X = X.reshape((X.shape[0], -1))
        X_pred = X_pred.reshape((X_pred.shape[0], -1))
        r2 = r2_score(X, X_pred)
        # predictions = self.predict_classifier(X)
        accuracy = (X_clas == y).mean()
        score = 0.5*r2 + 0.5*accuracy
        return score

    def score_classefier(self, X, y) -> float:
        predictions = self.predict_classifier(X)
        accuracy = (predictions == y).mean()
        return accuracy

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


class JointAEClassifierWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, h_layer_sizes, timesteps, learning_rate, learning_rate_c, dropout_ae, dropout_c, n_features_in, batch_size, stat, alpha, optim, epochs, verbose, patience):
        self.h_layer_sizes = h_layer_sizes
        self.timesteps = timesteps
        self.learning_rate = learning_rate
        self.learning_rate_c = learning_rate_c
        self.dropout_ae = dropout_ae
        self.dropout_c = dropout_c
        self.n_features_in = n_features_in
        self.batch_size = batch_size
        self.stat = stat
        self.alpha = alpha
        self.optim = optim
        self.epochs = epochs
        self.verbose = verbose
        self.patience = patience
        # self.tolerance = tolerance
        self.model_ = None

    def _create_aeclassifier(self):
        return JointAEClassifier(
            h_layer_sizes=self.h_layer_sizes,
            timesteps=self.timesteps,
            learning_rate=self.learning_rate,
            learning_rate_c=self.learning_rate_c,
            dropout_ae=self.dropout_ae,
            dropout_c=self.dropout_c,
            batch_size=self.batch_size,
            alpha=self.alpha,
            optim=self.optim,
            stat=self.stat,
            epochs=self.epochs,
            verbose=self.verbose,
            n_features_in=self.n_features_in,
            patience = self.patience,
            # tolerance = self.tolerance
        ) 
        
    def fit(self, X, y, p_train=0.9):
        self.model_ = self._create_aeclassifier()
        self.model_.fit(X, y, p_train)
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def score(self, X, y=None):
        return self.model_.score(X, y)

    def save_weights(self, filepath = 'AE.pth'):
        return self.model_.save_weights(filepath)