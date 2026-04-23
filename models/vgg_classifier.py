import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten
from keras.applications import VGG16
from keras.optimizers import Adam

def get_model(img_rows, img_cols):

    base_model = VGG16(weights='imagenet' , include_top=False , input_shape=(img_rows, img_cols, 3))

    output = base_model.layers[-1].output
    output = keras.layers.Flatten()(output)
    base_model = Model(base_model.input, output)

    # Freeze the base VGG16 model layers
    for layer in base_model.layers:
        layer.trainable = False

    # Build the custom model
    model = Sequential()
    model.add(base_model)
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model