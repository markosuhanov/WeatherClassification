import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import seaborn as sn
from sklearn.metrics import confusion_matrix


class Utils:

    @staticmethod
    def load_train_dataframe():
        train_df = pd.read_csv('./data/train_labels.csv', dtype=str)
        print("Train Dataframe: " + str(train_df.shape))
        print("Top 5 records")
        print(train_df.head())
        return train_df

    @staticmethod
    def load_test_dataframe():
        test_df = pd.read_csv('./data/test_labels.csv', dtype=str)
        print("Test Dataframe: " + str(test_df.shape))
        print("Top 5 records")
        print(test_df.head())
        return test_df

    @staticmethod
    def load_train_valid_images(train_df, image_size=224, val_split=0.25):
        datagen = ImageDataGenerator(rescale=1. / 255.,
                                     validation_split=val_split,
                                     rotation_range=20,
                                     width_shift_range=0.2,
                                     height_shift_range=0.2,
                                     horizontal_flip=True,
                                     vertical_flip=True)

        train_generator = datagen.flow_from_dataframe(
            dataframe=train_df,
            directory="./data/train/",
            x_col="file_name",
            y_col="weather",
            subset="training",
            batch_size=64,
            seed=11,
            shuffle=True,
            class_mode="categorical",
            target_size=(image_size, image_size))

        validation_generator = datagen.flow_from_dataframe(
            dataframe=train_df,
            directory="./data/train/",
            x_col="file_name",
            y_col="weather",
            subset="validation",
            batch_size=64,
            seed=11,
            shuffle=True,
            class_mode="categorical",
            target_size=(image_size, image_size))

        return train_generator, validation_generator

    @staticmethod
    def load_test_images(test_df, image_size=224):
        datagen = ImageDataGenerator(rescale=1. / 255.)
        test_generator = datagen.flow_from_dataframe(
            dataframe=test_df,
            directory="./data/test/",
            x_col="file_name",
            y_col="weather",
            batch_size=64,
            seed=11,
            shuffle=False,
            target_size=(image_size, image_size))

        return test_generator

    @staticmethod
    def print_accuracy(model, train_images, valid_images, test_images):
        train_accuracy = model.evaluate(train_images)
        print('Train accuracy:', train_accuracy[1])

        valid_accuracy = model.evaluate(valid_images)
        print('Validation accuracy:', valid_accuracy[1])

        test_accuracy = model.evaluate(test_images)
        print('Test accuracy:', test_accuracy[1])

    @staticmethod
    def print_confusion_matrix(result_df, class_indices):
        conf_matrix = confusion_matrix(result_df.weather, result_df.predictions)

        conf_matrix_df = pd.DataFrame(conf_matrix, class_indices.items(), class_indices.items())
        conf_matrix_df.index.name = 'Actual'
        conf_matrix_df.columns.name = 'Predicted'
        plot = sn.heatmap(conf_matrix_df, annot=True, annot_kws={"size": 16})
        fig = plot.get_figure()
        fig.savefig("./results/confusion_matrix.png")

        print("Confusion Matrix:")
        print(conf_matrix)

    @staticmethod
    def save_results(test_df, predictions, class_indices):
        result = test_df.copy()
        predictions = np.argmax(predictions, axis=1)
        predictions_name = [class_indices[x] for x in predictions]
        result['predictions'] = predictions_name
        result.to_csv("./results/test_result.csv")
        return result
