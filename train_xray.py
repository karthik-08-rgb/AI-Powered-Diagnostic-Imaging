#Train X-ray Model (DenseNet121)
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 5
LR = 1e-4
XRAY_MODEL_PATH = "xray_model.h5"

# Data generators
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_gen = datagen.flow_from_directory(XRAY_DATA_DIR, target_size=IMAGE_SIZE,
                                        batch_size=BATCH_SIZE, class_mode="categorical",
                                        subset="training")
val_gen = datagen.flow_from_directory(XRAY_DATA_DIR, target_size=IMAGE_SIZE,
                                      batch_size=BATCH_SIZE, class_mode="categorical",
                                      subset="validation")

# Build DenseNet121 model
base_model = DenseNet121(weights="imagenet", include_top=False, input_shape=IMAGE_SIZE+(3,))
for layer in base_model.layers:
    layer.trainable = False

x = GlobalAveragePooling2D()(base_model.output)
x = Dense(256, activation="relu")(x)
x = Dropout(0.5)(x)
preds = Dense(len(train_gen.class_indices), activation="softmax")(x)
model_xray = Model(inputs=base_model.input, outputs=preds)
model_xray.compile(optimizer=Adam(LR), loss="categorical_crossentropy", metrics=["accuracy"])

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint(XRAY_MODEL_PATH, monitor="val_loss", save_best_only=True)
]

model_xray.fit(train_gen, validation_data=val_gen, epochs=EPOCHS, callbacks=callbacks)
print("✅ X-ray model trained and saved:", XRAY_MODEL_PATH)