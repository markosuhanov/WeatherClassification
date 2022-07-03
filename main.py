import numpy as np
import tensorflow as tf
import random as python_random
from models import Model
from utils import Utils

IMAGE_SIZE = 224
RANDOM_SEED = 11


def main():
    # Set seed to make train reproducible
    np.random.seed(RANDOM_SEED)
    python_random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    # Load train and test dataframe
    train_df = Utils.load_train_dataframe()
    test_df = Utils.load_test_dataframe()

    # Load image data generators for train, valid and test
    train_images, valid_images = Utils.load_train_valid_images(train_df, IMAGE_SIZE)
    test_images = Utils.load_test_images(test_df, IMAGE_SIZE)

    # Get model
    model = Model.build_arhitecture(IMAGE_SIZE)

    # Fit model
    history = model.fit(train_images, validation_data=valid_images, epochs=10)

    # Save model
    model.save('./results/model')

    # # Load model
    # import keras
    # model = keras.models.load_model('./results/model')

    # Print accuracy
    Utils.print_accuracy(model, train_images, valid_images, test_images)

    # Get cass indices reverse
    class_indices = {v: k for k, v in test_images.class_indices.items()}

    # Predict
    predictions = model.predict(test_images)

    # Save results
    result_df = Utils.save_results(test_df, predictions, class_indices)

    # Print confusion matrix
    Utils.print_confusion_matrix(result_df, class_indices)


if __name__ == '__main__':
    main()
