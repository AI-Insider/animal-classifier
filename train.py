from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, LeakyReLU
from keras.preprocessing.image import ImageDataGenerator


model = Sequential()

model.add(Conv2D(64,3,padding="same",input_shape=(32,32,3)))
model.add(LeakyReLU())
model.add(MaxPooling2D())

model.add(Conv2D(128,3,padding="same"))
model.add(LeakyReLU())
model.add(MaxPooling2D())

model.add(Conv2D(256,3,padding="same"))
model.add(LeakyReLU())
model.add(MaxPooling2D())


model.add(Flatten())
model.add(Dense(2048))
model.add(LeakyReLU())
model.add(Dropout(rate=0.4))
model.add(Dense(6,activation="softmax"))


model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        'data/train',
        target_size=(32,32),
        batch_size=20,
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        'data/test',
        target_size=(32,32),
        batch_size=20,
        class_mode='categorical')

model.fit_generator(
        train_generator,
        steps_per_epoch=1500,
        epochs=25,
        validation_data=validation_generator,
        validation_steps=300)

model.save("my_model.h5")
