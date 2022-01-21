from __future__ import print_function
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten,BatchNormalization
from keras.layers import Conv2D,MaxPooling2D
import os

num_classes = 5
pic_rows,pic_cols = 48,48
batch_size = 32
cons=64
model = Sequential()

train_data_dir = '/Users/durgeshthakur/Deep Learning Stuff/Emotion Classification/fer2013/train'
validation_data_dir = '/Users/durgeshthakur/Deep Learning Stuff/Emotion Classification/fer2013/validation'

train_datagen = ImageDataGenerator(
					rescale=1./255,
					rotation_range=30,
					shear_range=0.3,
					zoom_range=0.3,
					width_shift_range=0.4,
					height_shift_range=0.4,
					horizontal_flip=True,
					fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
					train_data_dir,
					color_mode='grayscale',
					target_size=(pic_rows,pic_cols),
					batch_size=batch_size,
					class_mode='categorical',
					shuffle=True)

validation_generator = validation_datagen.flow_from_directory(
							validation_data_dir,
							color_mode='grayscale',
							target_size=(pic_rows,pic_cols),
							batch_size=batch_size,
							class_mode='categorical',
							shuffle=True)


cnn_block()
flat_maker()
pressing()



print(model.summary())

from keras.optimizers import RMSprop,SGD,Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

checkpoint = ModelCheckpoint('Train.h5',
                             monitor='val_loss',
                             mode='min',
                             save_best_only=True,
                             verbose=1)

earlystop = EarlyStopping(monitor='val_loss',
                          min_delta=0,
                          patience=3,
                          verbose=1,
                          restore_best_weights=True
                          )

reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                              factor=0.2,
                              patience=3,
                              verbose=1,
                              min_delta=0.0001)

callbacks = [earlystop,checkpoint,reduce_lr]

model.compile(loss='categorical_crossentropy',
              optimizer = Adam(lr=0.001),
              metrics=['accuracy'])

nb_train_samples = 24176
nb_validation_samples = 3006
epochs=25

history=model.fit_generator(
                train_generator,
                steps_per_epoch=nb_train_samples//batch_size,
                epochs=epochs,
                callbacks=callbacks,
                validation_data=validation_generator,
                validation_steps=nb_validation_samples//batch_size)


def cnn_block():

    # Block-1

    model.add(Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal', input_shape=(pic_rows, pic_cols, 1)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal', input_shape=(pic_rows, pic_cols, 1)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    # Block-2
    list = [64, 128, 256]

    # Using for loop
    for i in list:
        model.add(Conv2D(i, (3, 3), padding='same', kernel_initializer='he_normal'))
        model.add(Activation('elu'))
        model.add(BatchNormalization())
        model.add(Conv2D(i, (3, 3), padding='same', kernel_initializer='he_normal'))
        model.add(Activation('elu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

def flat_maker():
    model.add(Flatten())
    model.add(Dense(cons, kernel_initializer='he_normal'))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

def pressing():
    model.add(Dense(cons, kernel_initializer='he_normal'))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, kernel_initializer='he_normal'))
    model.add(Activation('softmax'))









































