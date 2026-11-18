import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import os

# Path ke folder validasi / fold data
val_dir = r"C:\Users\Admin\Desktop\bljr\ipsd\tugaskel\Data\archive\Original Images\Original Images\FOLDS\fold1\Train"

IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 32

# Memuat data validasi
val_ds = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

class_names = val_ds.class_names

# Mengambil label asli (True Labels)
y_true = []
for images, labels in val_ds:
    y_true.extend(labels.numpy())

# --- GANTI BAGIAN INI SECARA BERGANTIAN ---
# Pilihan model:
# 1. "Models/simple_cnn_model_1.h5"
# 2. "Models/modified_cnn_model_2.h5"
# 3. "Models/transfer_learning_model_3.h5"
model_path = "Models/transfer_learning_model_3.h5"

if os.path.exists(model_path):
    print(f"\n==============================================")
    print(f"MENGEVALUASI MODEL: {model_path}")
    print(f"==============================================")
    
    model = tf.keras.models.load_model(model_path)
    
    # Melakukan prediksi
    predictions = model.predict(val_ds)
    y_pred_classes = np.argmax(predictions, axis=1)
    
    # Menampilkan Classification Report (Precision, Recall, F1-Score, Accuracy)
    print("\n--- CLASSIFICATION REPORT ---")
    print(classification_report(y_true, y_pred_classes, target_names=class_names, zero_division=0))
    
    # Menampilkan Confusion Matrix
    print("\n--- CONFUSION MATRIX ---")
    print(confusion_matrix(y_true, y_pred_classes))
else:
    print(f"Model di {model_path} tidak ditemukan. Pastikan sudah melatihnya!")