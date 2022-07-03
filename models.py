from keras import optimizers
import keras


class Model:

    @staticmethod
    def build_arhitecture(image_size=224, output_number=4, learning_rate=0.001):
        keras.backend.clear_session()  # clear model numbers
        input_layer = keras.layers.Input(shape=(image_size, image_size, 3))

        conv_layer_1 = keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same")(input_layer)

        conv_layer_2 = keras.layers.Conv2D(16, (3, 3), activation="relu", padding="same")(conv_layer_1)

        pooling_layer = keras.layers.MaxPool2D(pool_size=(2, 2))(conv_layer_2)

        flatten = keras.layers.Flatten()(pooling_layer)
        dense = keras.layers.Dense(200, activation="relu")(flatten)
        dropout = keras.layers.Dropout(0.25)(dense)

        classifier = keras.layers.Dense(output_number, activation="softmax")(dropout)

        model = keras.Model(inputs=input_layer, outputs=classifier)
        opt = keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=['accuracy'])

        print(model.summary())
        return model
