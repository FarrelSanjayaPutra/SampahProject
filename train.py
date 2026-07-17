import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# ==========================================
# 1. KONFIGURASI & PREPROCESSING DATA
# ==========================================
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10

# Path dataset (Sesuaikan dengan path Anda)
train_dir = 'dataset_sampah/train'
val_dir = 'dataset_sampah/validation'

# Augmentasi data untuk memperkaya variasi gambar
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

# Tentukan subset folder yang sah secara kaku
NAMA_KELAS = ['anorganik', 'limbah_b3', 'organik', 'residu']

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=NAMA_KELAS # <--- Tambahkan ini
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=NAMA_KELAS # <--- Tambahkan ini
)

# ==========================================
# 2. MEMBANGUN MODEL (TRANSFER LEARNING)
# ==========================================
# Menggunakan MobileNetV2 tanpa layer klasifikasi atasnya (include_top=False)
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False # Membekukan bobot asli MobileNetV2

# Menambahkan layer baru untuk 4 kelas kita
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5), # Mencegah overfitting
    layers.Dense(4, activation='softmax') # 4 output untuk 4 jenis sampah
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ==========================================
# 3. TRAINING MODEL
# ==========================================
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator
)

# ==========================================
# 4. MENYIMPAN MODEL
# ==========================================
model.save('model_klasifikasi_sampah.h5')
print("Model berhasil disimpan dengan nama 'model_klasifikasi_sampah.h5'")