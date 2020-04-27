from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model


longitud, altura = 150,150
modelo = r'C:\Users\PAOLITA\Documents\Pao\URL\2020\IA\modelo\modelo.h5'
pesos_modelo = r'C:\Users\PAOLITA\Documents\Pao\URL\2020\IA\modelo\pesos.h5'
cnn = load_model(modelo)
cnn.load_weights(pesos_modelo)

def predict(file,answer):
  x = load_img(file, target_size=(longitud, altura))
  x = img_to_array(x)
  x = np.expand_dims(x, axis=0)
  array = cnn.predict(x)
  result = array[0]
  asnwer = np.argmax(result)*10
    
  
  if answer <20:
    print("pred: cebolla en fase1")
  elif answer <30:
    print("pred: cebolla en fase2")
  elif answer <40:
    print("pred: cebolla en fase3")
  else:
    print("no es ceboolla")

  return answer

num = 31 #numero de la imagen
predict("C:\\Users\\PAOLITA\\Pictures\\prueba\\"+str(num)+".jpg",num)
cnn.summary()