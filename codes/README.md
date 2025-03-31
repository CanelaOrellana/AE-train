# Carpeta de Códigos

Aquí se encuentra:
* el código `AutoEncoders.py`
* el código `CasoBase_train.ipynb`
* la carpeta **Detección de Anomalías**

_ _ _
_ _ _
## ✨AutoEncoders.py✨

En este script de python, se encuentran las clases que definen a los objetos AutoEncoder y ClassifierAutoEncoder. A continuación les explico qué se puede hacer con cada una de ellas, en qué consisten los objetos, quée métodos traen y cuales son los hiperparámetros que pueden utilizar para obtener el autoencoder de sus sueños. 

_pd: obvio no olviden que cuando clonen este repo o copien los códigos, podrán agregar y quitar los métodos que quieran_

### Clase ✨<span style="color:teal">AutoEncoder</span>✨


Posee los hiperparámetros:
- `h_layer_sizes` <span style="color:green">List [ int ]</span> : recibe una lista, la cual contiene los tamaños de las capas ocultas del encoder (y por ende, del decoder)  

- `timesteps` <span style="color:green">int</span> : representa la cantidad de pasos en el tiempo que se reciben, es el tamaño de la ventana temporal. (Si se trabajara con frases, sería el número de palabras)  

- `learning_rate` <span style="color:green">float</span> :  corresponde al learning rate de toda la vida  

- `dropout` <span style="color:green">float</span> : corresponde al dropout que se aplica luego de la capa de salida del encoder y del decoder.  
_pd: nunca obtuve mejor rendimiento utilizándolo_  

- `n_features_in` <span style="color:green">int</span> : es el número de características de entrada de los datos que se están ingresando, en mi caso, correspondía a la cantidad de sensores que estaba considerando en mis datos de entrenamiento  

- `batch_size` <span style="color:green">int</span> : es el batch_size de toda la vida.  
_pd: en mi experiencia es mejor utiliza números pequeños (menores a 10), no sólo por eel desempeño obtenido, sino que también por la capacidad de la gpu que yo utilicé_  

- `optim` <span style="color:green">string</span>:  puede ser <span style="color:orange">'Adam'</span>, <span style="color:orange">'RAdam'</span> o <span style="color:orange">'AdamW'</span> y permite elegir qué optimizador desea usar en el entrenamiento futuro.  
_pd: sólo se configuró para esas 3 posibilidades, si se desea utilizar otro optimizador, se debe modificar un poco el código.  
pd 2: siempre me funcionó mejor RAdam (tiene sentido)_  

- `epochs` <span style="color:green">int</span> : representa el número de épocas (máximo) durante las que se llevará a cabo el entrenamiento.  

- `verbose` <span style="color:green">int</span> : permite controlar cuántos mensajes informativos se quiere recibir durante el entrenamiento. Como regla general, se reciben mensajes en las épocas múltiplos del número que se utilice, el mensaje incluye: época, loss en validación, R^2 en validación y R^2 en entrenamiento.  
También hay algunos números especiales:
    - **0**: Se imprime el número de capas, leearning rate, dropout y paciencia que se están utilizando en el entrenamiento (No se imprime nada durante los bucles de entrenamiento).  
    _pd: el objetivo de esto era seguirle la pista a los entrenamientos al utilizar GridSearch_  
    - **5**: Además de los mensajes cada 5 épocas, al finalizar el entrenamiento se muestran los gráficos de **loss vs épocas** y de **r^2 vs épocas**. 

- `patience` <span style="color:green">int</span> : representa la cantidad de épocas que se tendrá paciencia si el modelo no está bajando su función de _loss_, en caso de que se superen las épocas de paciencia, se detendrá tempranamente el entrenamiento del modelo, ya que no se están obteniendo mejoras y no queremos malgastar ni tiempo ni recursos de computo.

Métodos que posee:

- `build_model`: permite construir la arquitectura del modelo según los hiperparámetros antes definidos. Posee 3 clases locales
    - Encoder: permite crear un objeto Encoder a partir de los hiperparámetros antes definidos, incluyendo su método fordward.    
    - Decoder: permite crear un objeto Decoder a partir de los hiperparámetros antes definidos, incluyendo su método fordward.  
    - Autoencoder: permite crear al objeto Autoencoder, que consiste en un Encoder seguido de un Decoder, incluyendo su método fordward.  
En este método retorna el modelo de arquitectura autoencoder, también se definen aquí, la función de loss y el optimizador del modelo.  

-  `fit`: este método permite llevar a cabo los bucles de entrenamiento del modelo, utilizando de apoyo a las funciones `ftrain` y `fevaluate`. Para funcionar recibe:
    - **X**: conjunto de datos de entrenamiento, deben estar en el formato esperado []
    - `p_train` <span style="color:green">float</span> : representa el porcentaje de X que se usará para entrenamiento, el porcentaje restante se utilizará para validación.  
    _pd: estos AEs están diseñados para el procesamiento de señales, por lo que no se realiza shuffle de los conjuntos. Tenerlo presente si se desea utilizar estos códigos con otro tipo de datos_  

- `predict`: este método permite predecir utilizando el modelo entrenado (o no).

- `score`: este método permite calcular una métrica para evaluar la predicción del modelo. En el código se puede modificar para utilizar mse o r^2.  
_pd: este método es utilizado por GridSearch para definir qué modelos de los entrenados son mejores_

- `save_weights`: este método permite guardar los pesos del modelo entrenado, para en un futuro poder simplemente cargarlos en el modelo, sin necesidad de reentrenar.  

- `transform`: este método permite transformar los datos a la forma del espacio latente del autoencoder.  
_pd: existe por completitud de los métodos que se implementan en los modelos de scikit learn, que es para los que funciona GridSearch de manera nativa_  
_pd 2: pero por su puesto, el espacio latente que se produce es de gran interés 💗_

- `inverse_transform`: este método permite transformar los datos desde el espacio latente al espacio de entrada al autoencoder.  
_pd: también existe por completitud de los métodos que se implementan en los modelos de scikit learn, que es para los que funciona GridSearch de manera nativa_

#### Clase ✨AutoEncoderWrapper✨
Esta clase sirve para _envolver_ nuestro autoencoder creado con la clase explicada anteriormente. Por lo que recibe exactamente los mismos hiperparámetros que <span style="color:teal">AutoEncoder</span>. Sus métodos se describen a continuación

- `_create_autoencoder`: este método retorna un objeto de la clase AutoEncoder, con los hiperparámetros recibidos.
- `fit`: este método utiliza el método anterior para crear un autoencoder y lo entrena llamando al método <span style="color:orange">fit</span> de AutoEncoder.
- `predict`: este método llama al método <span style="color:orange">predict</span> de AutoEncoder.
- `score`: este método llama al método <span style="color:orange">score</span> de AutoEncoder.
- `save_weights`: este método llama al método <span style="color:orange">save\_weights</span> de AutoEncoder.

Este objeto permite utilizar los métodos de `GridSearch` o `HalvingGridSearch` para encontrar los mejores hiperparámetros


### Clase ✨<span style="color:teal">ClassifierAutoEncoder</span>✨

P R O X I M A M E N T E

_ _ _
_ _ _
## ✨CasoBase_train.ipynb✨

En este notebook se muestran ejemplos para
- Realizar los entrenamientos utilizando grillas
- Realizar un entrenamiento específico
- Cargar los pesos para un uso posterior del modelo




_ _ _
_ _ _
## ✨Detección de Anomalías✨

En esta carpeta aún no pueden encontrar nada, pero próximamente estarán los códigos de cómo se puede realizar detección de anomalías a partir de los autoencoders creados 