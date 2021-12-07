from anaspingpong.utils import Utils
import os
import tensorflow as tf
from tensorflow.keras import models, layers
import numpy as np

EXTEND_TILES = 2
ZOOM = 20
SOURCE = "http://ecn.t0.tiles.virtualearth.net/tiles/a{quad}.jpeg?g=129&mkt=en&stl=H"
#source = "https://mt0.google.com/vt?lyrs=h&x={x}&s=&y={y}&z={z}"
MODEL = tf.keras.models.load_model('../model/checkpoint_Fbeta_entire_model/')
BATCH_SIZE = 25
IMAGE_SIZE = (512, 512)
THRESHOLD = .5

def get_dataset(data_dir):
    dataset = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        labels=None,
        image_size=IMAGE_SIZE,
        shuffle=False,
        batch_size=BATCH_SIZE
    )
    return dataset

def encode(images_batch):
    """ Function to transform images """
    images_batch = tf.image.convert_image_dtype(images_batch, dtype=tf.float32)
    return images_batch


def get_tables(latitude, longitude):
    # download center tile +- EXTEND_TILES in all directions
    download_folder = download_tables(latitude, longitude)
    print(download_folder)

    # create tensorflow dataset
    dataset = get_dataset(download_folder)
    dataset_encode = dataset.map(lambda dataset: encode(dataset))

    # predict probabilities
    label_pred = MODEL.predict(dataset_encode)

    # get images with positive prediction
    keys_pos = np.where(label_pred > THRESHOLD)[0]
    if len(keys_pos) == 0:
        return [], []
    file_names = np.array(dataset.file_paths)

    positive_tiles = [os.path.basename(s.replace('.jpeg', '')) for s in file_names[keys_pos]]
    print(positive_tiles)
    positive_tiles = np.array([s.split('_') for s in positive_tiles])

    # extract x, y, z from filename and convert to lon, lat
    xs = positive_tiles[:, 0].astype(int)
    ys = positive_tiles[:, 1].astype(int)
    zooms = positive_tiles[:, 2].astype(int)
    z = zooms[0]
    longitudes = [Utils.tile2long(x, z) for x in xs]
    latitudes = [Utils.tile2lat(y, z) for y in ys]
    return longitudes, latitudes


def download_tables(latitude, longitude):
    center_x = Utils.long2tile(longitude, ZOOM)
    center_y = Utils.lat2tile(latitude, ZOOM)
    z = ZOOM

    tempDirectory = os.path.join("temp", f'tmp{Utils.randomString()}')
    os.makedirs(tempDirectory)
    for i in range(-EXTEND_TILES, EXTEND_TILES+1):
        for j in range(-EXTEND_TILES, EXTEND_TILES+1):
            # take n tiles left and right, up and down of center tile
            x = center_x + i
            y = center_y + j
            tempFile = f"{x}_{y}_{ZOOM}" + ".jpeg"
            tempFilePath = os.path.join(tempDirectory, tempFile)
            result = Utils.downloadFile(SOURCE, tempFilePath, x, y, z)
    return tempDirectory