from keras.models import load_model
from PIL import Image
import numpy as np
import sys

filename = sys.argv[1]

image = Image.open(filename)
image= image.resize((32,32),Image.ANTIALIAS)
image = np.asarray(image)
image = np.expand_dims(image,axis=0)

model = load_model("my_model.h5")

prediction = model.predict(image)[0]

print("Bird: {0}, cat: {1}, deer: {2}, dog: {3}, frog: {4}, horse: {5}".format(prediction[0],prediction[1],prediction[2],prediction[3],prediction[4],prediction[5]))
