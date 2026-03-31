import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# -------------------- PATHS --------------------
train_dir = "/content/drive/MyDrive/MedicalData"
print("✅ Classes found:", os.listdir(train_dir))

# -------------------- DATA GENERATORS --------------------
datagen = ImageDataGenerator(
    rescale=1.0/255,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.15,
    horizontal_flip=True
)

train_gen = datagen.flow_from_directory(
    train_dir,
    target_size=(128, 128),
    batch_size=64,
    class_mode="categorical",
    subset="training",
    classes=["chest_xray", "mri_file", "other"]
)

val_gen = datagen.flow_from_directory(
    train_dir,
    target_size=(128, 128),
    batch_size=64,
    class_mode="categorical",
    subset="validation",
    classes=["chest_xray", "mri_file", "other"]
)

print("✅ Class indices:", train_gen.class_indices)

# -------------------- MODEL --------------------
modality_model = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(128,128,3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation="relu"),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(256, activation="relu"),
    Dropout(0.5),
    Dense(3, activation="softmax")
])

# -------------------- COMPILE --------------------
modality_model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# -------------------- CALLBACKS --------------------
save_path = "/content/drive/MyDrive/MedicalData/modality_model1.h5"

checkpoint = ModelCheckpoint(
    filepath=save_path,
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=2,
    restore_best_weights=True,
    verbose=1
)

# -------------------- TRAIN --------------------
history = modality_model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=6n ,  # 👈 only 5 epochs (good for ~2 hours CPU)
    callbacks=[checkpoint, early_stop],
    verbose=1
)

# -------------------- EVALUATE --------------------
loss, acc = modality_model.evaluate(val_gen)
print(f"✅ Validation Accuracy: {acc*100:.2f}%")
print(f"💾 Best model saved at: {save_path}")