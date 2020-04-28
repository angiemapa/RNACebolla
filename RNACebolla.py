import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Flatten, Dense, Activation
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, Conv2D
from tensorflow.keras.preprocessing import image
from tensorflow.keras import backend as k
#from mlxtend.evaluate import confusion_matrix
import matplotlib.image as mping
import matplotlib.pyplot as plt
from tensorflow.python.keras.optimizers import Adam 

k.clear_session()
#colocamos la ruta donde esta la carpeta de las imagenes
pat = (r".\prueba")
datos_entrenamiento = (r".\Validacion\faseA")
datos_validacion = (r".\Validacion\faseC")

epocas = 50
longitud, altura = 150,150#redimencionar el tamaño de la imagen
batch_size = 10 #cantidad de imagenes que procesa a la vez
filtrosConv1 = 32 #numero de filtros que aplicamos tas la primera capa
filtrosConv2 = 64 #numero de filtros que aplicamos tras la segunda capa2
size_filtro1 = (3,3)#para primera convolucion
size_filtro2 = (2,2)
size_pool = (2,2)#para mejorar el vance de la convolucio
netapas = 2 #debemos colocar todas las etapcas que vamos a evaluar de nuestro planta 
lr = 0.0004 #tendremo que ir probando con el error para ver cual esta el mejor resultado

#Restructurando nuestos datos de imagenes
entrenamiento_restructurada = ImageDataGenerator(
    rescale = 1./255, #rescalamos los pixeles de la imagen entre 0-1
    shear_range = 0.3, #inclinar imagenes
    zoom_range = 0.3, #Porciones de imagenes
    horizontal_flip = True)

validacion_restructurada = ImageDataGenerator(
    rescale = 1./255)

#abrir y alistar todo la carpeta de entrenamiento
imagen_entrenamiento = entrenamiento_restructurada.flow_from_directory(
    datos_entrenamiento,
    target_size= (altura,longitud),
    batch_size = batch_size,
    class_mode = 'categorical')

imagen_validacion = validacion_restructurada.flow_from_directory(
    datos_validacion,
    target_size= (altura,longitud),
    batch_size = batch_size,
    class_mode = 'categorical')
print(imagen_entrenamiento.class_indices)

pasos_entrenamiento = imagen_entrenamiento.n//imagen_entrenamiento.batch_size
pasos_validacion = imagen_validacion.n//imagen_validacion.batch_size
print(pasos_entrenamiento,pasos_validacion)

#Creamos la red neuronal convolucional

cnn = Sequential()
cnn.add(Convolution2D(64, #cambie 64
               kernel_size=(3, 3), 
               padding ='same',
               input_shape=(longitud,altura,3),
               activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2, 2)))

cnn.add(Convolution2D(128, kernel_size=(3, 3), activation='relu')) #128
cnn.add(MaxPooling2D(pool_size=(2, 2)))
cnn.add(Convolution2D(256, kernel_size=(3, 3), activation='relu')) #256
cnn.add(MaxPooling2D(pool_size=(2, 2)))
cnn.add(Convolution2D(512, kernel_size=(3, 3), activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2, 2)))
cnn.add(Convolution2D(1024, kernel_size=(3, 3), activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2, 2)))

cnn.add(Flatten())
cnn.add(Dense(256, activation='relu'))
cnn.add(Dropout(0.5))
cnn.add(Dense(netapas, activation='softmax'))        

cnn.compile(loss='categorical_crossentropy',
            optimizer='sgd', 
            metrics=['accuracy'])

H = cnn.fit_generator(imagen_entrenamiento,
                     steps_per_epoch=pasos_entrenamiento,
                     epochs=epocas,
                     validation_data=imagen_validacion,
                     validation_steps=pasos_validacion)


#ruta carpeta para guardar modelo
target_dir = r'.\modelo'
#si no existe, lo guarda
if not os.path.exists(target_dir):
  os.mkdir(target_dir)
cnn.save(r'.\modelo\modelo.h5') #nombre modelo
cnn.save_weights(r'.\modelo\pesos.h5') #nombre pesos