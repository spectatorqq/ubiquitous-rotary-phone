"""
Classify a few images through our CNN.
"""
import numpy as np
import operator
import random
import glob
from HMDB51data import DataSet
from processor import process_image
from keras.models import load_model

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
import os

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

root_dir = 'F:\\Learning_Python\\document\\hmdb51_org\\HMDB51\\'


def main(nb_images=5):
    """Spot-check `nb_images` images."""
    data = DataSet()
    model = load_model('./data/checkpoints/inception.0026-2.54.hdf5')  # replaced by your model name

    # Get all our test images.
    images = glob.glob(root_dir + 'test/*/*.jpg')

    for _ in range(nb_images):
        print('-' * 80)
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
        # print(sorted_lps)
        # [('walk', 0.8944378), ('turn', 0.04855361), ('stand', 0.040282194), ('wave', 0.009945527),
        #  ('talk', 0.006243166), ('smile', 0.00053773355)]
        # print("*" * 20)
        for i, class_prediction in enumerate(sorted_lps):
            # Just get the top five.
            if i > 4:
                break
            print("%s: %.2f" % (class_prediction[0], class_prediction[1]))
            i += 1


if __name__ == '__main__':
    main()
