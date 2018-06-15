"""
Classify test images set through our CNN.
Use keras 2+ and tensorflow 1+
It takes a long time for hours.
"""
import os
import numpy as np
import operator
import random
import glob
from UCFdata import DataSet
from processor import process_image
from keras.models import load_model, Model
from keras.preprocessing.image import ImageDataGenerator

data = DataSet()
def main(nb_images=5):
    # CNN model evaluate

    test_data_gen = ImageDataGenerator(rescale=1. / 255)
    batch_size = 32
    test_generator = test_data_gen.flow_from_directory('./data/test/', target_size=(299, 299),
                                                       batch_size=batch_size, classes=data.classes,
                                                       class_mode='categorical')

    test_data_num = test_generator.classes.shape[0]
    print 'There is a total of {} test samples to extract features.'.format(test_data_num)

    # load the trained model that has been saved in CNN_train_UCF101.py
    checkpoint = sorted(os.listdir('data/checkpoints/'))[-1] # get the last checkpoint
    filename = os.path.join('data/checkpoints/', checkpoint)
    model = load_model(filename)

    # Extracting the activation of the last layer.  Keras framework
    # does not allow us to get the last layer before the application
    # of the softmax (AFAIK), so then we need to get the activation of
    # the penultimate layer and the weights and biases of the last
    # layer and perform the computation to obtain the activation
    # before the softmax application.
    lastlayer = model.layers[-1]
    penlayer = model.layers[-2]
    model_penlayer = Model(model.input, penlayer.output)
    output_penlayer = model_penlayer.predict_generator(generator=test_generator, steps=np.ceil(float(test_data_num) / batch_size))
    weights_lastlayer, bias_lastlayer = lastlayer.get_weights()
    features = np.dot(output_penlayer, weights_lastlayer) + bias_lastlayer
    print 'There is a total of {} feature vectors generated.'.format(features.shape[0])

    with open('UCF101.dat', 'w') as fd:
        for label, feature_vector in zip(test_generator.classes, features):
            fd.write('{}'.format(label))
            for feature in feature_vector:
                fd.write(' {}'.format(feature))
            fd.write('\n')

if __name__ == '__main__':
    main()
