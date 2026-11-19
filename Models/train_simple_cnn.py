import tensorflow as tf
from tensorflow.keras import layers, models
import os

# 1.Tentukan path folder Train
base_dir = os.path.dirname(os.path.abspath(__file__))
train_dir = r"C:\Users\Admin\Desktop\bljr\ipsd\tugaskel\Data\archive\Original Images\Original Images\FOLDS\fold1\Train"

# 2.DEFINISIKAN VARIABEL INI DI ATAS 
IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 32

# 3.Panggil fungsi dataset di bawahnya
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
print("Kelas Penyakit yang Terdeteksi:", class_names)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# 4.Arsitektur Simple CNN 
model = models.Sequential([
    layers.Rescaling(1./255, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(class_names), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

model.summary()

# 5.Training Model (10 epochs)
epochs = 10
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

# 6.Simpan Model
os.makedirs('Models', exist_ok=True)
model.save('Models/simple_cnn_model_1.h5')
print("Model 1 berhasil disimpan!")