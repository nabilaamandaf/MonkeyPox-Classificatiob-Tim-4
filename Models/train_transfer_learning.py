import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
import os

# Menggunakan path dataset yang sama
train_dir = r"C:\Users\Admin\Desktop\bljr\ipsd\tugaskel\Data\archive\Original Images\Original Images\FOLDS\fold1\Train"

IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 32

print("Memuat dataset untuk Transfer Learning...")

# Memuat Dataset Training & Validation
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
print("Kelas Penyakit:", class_names)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

#3. ARSITEKTUR TRANSFER LEARNING (Menggunakan MobileNetV2)
# Memuat base model MobileNetV2 yang sudah terlatih (ImageNet), tanpa bagian top/classifier-nya
base_model = MobileNetV2(
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
    include_top=False,
    weights='imagenet'
)

# freeze base model agar bobot aslinya tidak berubah di awal pelatihan
base_model.trainable = False

# Menyusun model gabungan
model = models.Sequential([
    layers.Rescaling(1./127.5, offset=-1, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)), # Normalisasi khusus MobileNetV2 [-1, 1]
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.2),
    layers.Dense(len(class_names), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

model.summary()

# Training Model
epochs = 5
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

# Simpan Model Ketiga
os.makedirs('Models', exist_ok=True)
model.save('Models/transfer_learning_model_3.h5')
print("Model 3 (Transfer Learning) berhasil disimpan!")