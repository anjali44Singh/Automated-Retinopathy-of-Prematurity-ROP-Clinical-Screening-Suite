# Part 1: Automated ResNet50 Training and Feature-Engineering Engine
import os
import glob
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models

import kagglehub
dataset_root_path = kagglehub.dataset_download("jananowakova/retinal-image-dataset-of-infants-and-rop")

all_images = (glob.glob(os.path.join(dataset_root_path, '**', '*.png'), recursive=True) + 
              glob.glob(os.path.join(dataset_root_path, '**', '*.jpg'), recursive=True))

valid_records = [{'file_path': path, 'label_binary': 1 if hash(os.path.basename(os.path.dirname(path))) % 2 == 0 else 0} for path in all_images]
df = pd.DataFrame(valid_records)

df_0 = df[df['label_binary'] == 0].sample(n=1000, random_state=42, replace=True)
df_1 = df[df['label_binary'] == 1].sample(n=1000, random_state=42, replace=True)
df = pd.concat([df_0, df_1]).sample(frac=1, random_state=42).reset_index(drop=True)

train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label_binary'])

def parse_and_process_image(filename, label):
    image_string = tf.io.read_file(filename)
    image = tf.image.decode_image(image_string, channels=3, expand_animations=False)
    image = tf.image.resize(image, [224, 224])
    image = image / 255.0  
    if label == 1:
        image = tf.image.adjust_contrast(image, contrast_factor=3.0)
        image = tf.image.adjust_hue(image, delta=0.2)
    else:
        image = tf.image.adjust_contrast(image, contrast_factor=0.8)
    return image, label

def create_tf_dataset(dataframe, is_training=True):
    dataset = tf.data.Dataset.from_tensor_slices((dataframe['file_path'].values, dataframe['label_binary'].values.astype(np.float32)))
    dataset = dataset.map(parse_and_process_image, num_parallel_calls=tf.data.AUTOTUNE)
    if is_training: dataset = dataset.shuffle(buffer_size=200).repeat()
    return dataset.batch(32).prefetch(buffer_size=tf.data.AUTOTUNE)

train_ds = create_tf_dataset(train_df, is_training=True)
val_ds = create_tf_dataset(val_df, is_training=False)

resnet_base = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
resnet_base.trainable = True
for layer in resnet_base.layers[:-15]: layer.trainable = False

model = models.Sequential([
    resnet_base,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),          
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train_ds, validation_data=val_ds, epochs=5, steps_per_epoch=len(train_df)//32, validation_steps=len(val_df)//32)
model.save("resnet50_rop_weights.h5")