# Klasifikasi Monkeypox Multi-Model (5-Fold Cross-Validation)

Repositori ini dibuat untuk memenuhi tugas Mata Kuliah Infrastruktur dan Platform Sains Data 
oleh Kelompok 4:
- Faatin Jamiilatul Hasanah (108102500037)
- Nabila Amanda Fijienisa   (108102500036)
- M. Albani Mufti Radja T   (108102530013)

berisi eksperimen klasifikasi citra Monkeypox menggunakan tiga pendekatan *deep learning*: 
1. **Simple CNN**
2. **Modified CNN**
3. **MobileNetV2** (*Transfer Learning*)
yang dievaluasi menggunakan skema **5-Fold Cross-Validation** (`fold1` - `fold5`) pada *Validation* dan *Test*.

---

## Ringkasan Model
1. **Simple CNN**: Arsitektur CNN dasar sebagai *baseline* yang melatih fitur citra dari awal.
2. **Modified CNN**: Pengembangan arsitektur CNN yang dimodifikasi untuk meningkatkan ekstraksi fitur dibanding *baseline*.
3. **MobileNetV2**: Model *transfer learning* berbasis bobot awal (*pre-trained*) untuk memanfaatkan fitur visual secara lebih efisien.

---

## Struktur Proyek
```text
MonkeyPox-Classification-Tim-4/
├── assets/               # Grafik evaluasi, confusion matrix, ROC, dan kurva 5-fold
│   ├── Matriks
│   ├── ![Confusion Matrix MobileNetV2](assets/cm_transfer_learning_model_3.png)
├── Data/                            # Dataset lokal 
├── Models/                          # Skrip pelatihan model & file bobot (.h5 tidak dimasukkan Git)
│   ├── train_simple_cnn.py
│   ├── train_modified_cnn.py
│   └── train_transfer_learning.py
├── Utils/                           # Skrip evaluasi & visualisasi tambahan
│   └── eval_mobilenetv2_5fold.py
├── hasil_5fold_Simple_CNN.csv       # Hasil evaluasi 5-fold Simple CNN
├── hasil_5fold_Modified_CNN.csv     # Hasil evaluasi 5-fold Modified CNN
├── hasil_5fold_MobileNetV2.csv      # Hasil evaluasi 5-fold MobileNetV2
├── evaluate_5fold.py
├── evaluate_models.py
├── train_5fold_curves.py
└── README.md
![Confusion Matrix MobileNetV2](assets/cm_transfer_learning_model_3.png)
