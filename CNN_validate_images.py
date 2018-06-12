"""
Classify a few images through our CNN.
"""
import os
import numpy as np
import operator
import random
import glob
from UCFdata import DataSet
from processor import process_image
from keras.models import load_model

def main(nb_images=5):
    """Spot-check `nb_images` images."""
    data = DataSet()

    # load the trained model that has been saved in CNN_train_UCF101.py
    checkpoint = sorted(os.listdir('data/checkpoints/'))[-1] # get the last checkpoint
    filename = os.path.join('data/checkpoints/', checkpoint)
    model = load_model(filename)

    # Get all our test images.
    images = glob.glob('./data/test/**/*.jpg')

    for _ in range(nb_images):
        print('-'*80)
        # Get a random row.
        sample = random.randint(0, len(images) - 1)
        image = images[sample]

        # Turn the image into an array.
        print(image)
        image_arr = process_image(image, (299, 299, 3))
        image_arr = np.expand_dims(image_arr, axis=0)

        # Predict.
        predictions = model.predict(image_arr)

        # Show how much we think it's each one.
        label_predictions = {}
        for i, label in enumerate(data.classes):
            label_predictions[label] = predictions[0][i]

        sorted_lps = sorted(label_predictions.items(), key=operator.itemgetter(1), reverse=True)

        for i, class_prediction in enumerate(sorted_lps):
            # Just get the top five.
            if i > 4:
                break
            print("%s: %.2f" % (class_prediction[0], class_prediction[1]))
            i += 1

if __name__ == '__main__':
    main()
