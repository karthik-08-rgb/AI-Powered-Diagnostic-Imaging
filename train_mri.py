#Train MRI Model (VGG16)
from tensorflow.keras.applications import VGG16
MRI_MODEL_PATH = "mri_model.h5"

train_gen_mri = datagen.flow_from_directory(MRI_DATA_DIR, target_size=IMAGE_SIZE,
                                            batch_size=BATCH_SIZE, class_mode="categorical",
                                            subset="training")
val_gen_mri = datagen.flow_from_directory(MRI_DATA_DIR, target_size=IMAGE_SIZE,
                                          batch_size=BATCH_SIZE, class_mode="categorical",
                                          subset="validation")

# Build VGG16 model
base_model = VGG16(weights="imagenet", include_top=False, input_shape=IMAGE_SIZE+(3,))
for layer in base_model.layers:
    layer.trainable = False

x = GlobalAveragePooling2D()(base_model.output)
x = Dense(512, activation="relu")(x)
x = Dropout(0.5)(x)
preds = Dense(len(train_gen_mri.class_indices), activation="softmax")(x)
model_mri = Model(inputs=base_model.input, outputs=preds)
model_mri.compile(optimizer=Adam(LR), loss="categorical_crossentropy", metrics=["accuracy"])

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint(MRI_MODEL_PATH, monitor="val_loss", save_best_only=True)
]

model_mri.fit(train_gen_mri, validation_data=val_gen_mri, epochs=EPOCHS, callbacks=callbacks)
print("✅ MRI model trained and saved:", MRI_MODEL_PATH)