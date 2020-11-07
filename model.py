import numpy as np
import tensorflow as tf
import math
import pandas as pd
from sklearn import model_selection
import glob
import os
import tqdm as tqdm
import datetime
import logging
import json

tf.get_logger().setLevel(logging.ERROR)
import warnings
warnings.filterwarnings("ignore")


os.environ["CUDA_VISIBLE_DEVICES"]="1"

gpus = tf.config.experimental.list_physical_devices('GPU')
num_gpus = len(gpus)
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(num_gpus, "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

    policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
    tf.keras.mixed_precision.experimental.set_policy(policy)
    print('Compute dtype: %s' % policy.compute_dtype)
    print('Variable dtype: %s' % policy.variable_dtype)

    
if num_gpus == 0:
    strategy = tf.distribute.OneDeviceStrategy(device='CPU')
    print("Setting strategy to OneDeviceStrategy(device='CPU')")
elif num_gpus == 1:
    strategy = tf.distribute.OneDeviceStrategy(device='GPU')
    print("Setting strategy to OneDeviceStrategy(device='GPU')")
else:
    strategy = tf.distribute.MirroredStrategy()
    print("Setting strategy to MirroredStrategy()")
    
config = {
    'learning_rate': 1e-3,
    'momentum': 0.9,
    'input_size': (224, 224, 3),
    'n_classes': 1049,
}

def create_model(input_shape,
                 n_classes,
                 dense_units=512,
                 dropout_rate=0.0,
                 scale=30):

    backbone = tf.keras.applications.ResNet50(
        include_top=False,
        input_shape=input_shape,
        weights=('imagenet')
    )

    pooling = tf.keras.layers.GlobalAveragePooling2D(name='head/pooling')
    dropout = tf.keras.layers.Dropout(dropout_rate, name='head/dropout')
    dense = tf.keras.layers.Dense(n_classes, name='head/dense')
    softmax = tf.keras.layers.Softmax(dtype='float32')

    image = tf.keras.layers.Input(input_shape, name='input/image')

    x = backbone(image)
    x = pooling(x)
    x = dropout(x)
    x = dense(x)
    x = softmax(x)
    model = tf.keras.Model(
        inputs=image, outputs=x)
    
    model.compile(optimizer = tf.keras.optimizers.SGD(config['learning_rate'], momentum=config['momentum']),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.NONE),
              metrics=['accuracy'])     
    return model

def read_image(image_path):
    image = tf.io.read_file(image_path)
    return tf.image.decode_jpeg(image, channels=3)

def preprocess_input(image, target_size, augment=False):
    
    image = tf.image.resize(
        image, target_size, method='bilinear')

    image = tf.cast(image, tf.uint8)
    if augment:
        image = _spatial_transform(image)
        image = _pixel_transform(image)
    image = tf.cast(image, tf.float32)
    image /= 255.
    return image

def pred_image(image_path):
    model = create_model(input_shape=config['input_size'],
        n_classes=config['n_classes'],
        dense_units=512,
        dropout_rate=0.0,
        scale=30)
    
    checkpoint_path = "training_1/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    print(latest)
    model.load_weights(latest)
    
    image = read_image(image_path)
    image = preprocess_input(image, config['input_size'][:2])
    image = np.reshape(image, (1, 224, 224, 3))
    
    print('start pred')
    pred = model.predict(image)
    probs_argsort = tf.argsort(pred, direction='DESCENDING')
    probs = pred[0][probs_argsort][:5]
    
    category = pd.read_csv('category.csv')
    mapping = 'mapping.json'
    with open(mapping) as f:
        json_data = json.load(f)
        
    probs = []
    classes = []
    for i in range(5):
        probs.append(pred[0][[probs_argsort[0][i]]])
        idx = probs_argsort[0][i]
        classes.append(category['landmark_name'][category['landmark_id'] == int(json_data['landmark_id'][idx])].values[0])        
        
    return classes, probs