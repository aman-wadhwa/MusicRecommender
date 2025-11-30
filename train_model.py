import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models, optimizers
import os

# --- CONFIGURATION ---
# Make sure these match the folder names you extracted
TRAIN_DIR = 'seg_train/seg_train'
VAL_DIR = 'seg_test/seg_test'
IMG_SIZE = (150, 150) # Slightly smaller to train faster on a laptop
BATCH_SIZE = 128

# 1. DATA GENERATORS (The Pipeline)
# We add data augmentation to the training set to make the model smarter
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

print("Loading Training Data...")
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

print("Loading Validation Data...")
validation_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# 2. LOAD PRE-TRAINED MODEL (Transfer Learning)
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
# base_model.trainable = False  # Freeze the base so we don't ruin the pre-trained weights

# --- THE FIX: FINE TUNING ---
# Instead of freezing the whole thing, we unfreeze the top layers
base_model.trainable = True

# We want to freeze the early layers (identifying lines/shapes) 
# and ONLY train the deep layers (identifying complex scenes)
# ResNet50 has ~175 layers. Let's freeze the first 140.
for layer in base_model.layers[:140]:
    layer.trainable = False

# 3. BUILD THE CUSTOM HEAD
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(6, activation='softmax') # 6 Neurons for the 6 Intel Classes
])

# 4. COMPILE
# model.compile(optimizer='adam',
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])


model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 5. TRAIN
# We use fewer epochs (5) because the dataset is large (~14k images)
print("\n--- STARTING TRAINING ---")
history = model.fit(
    train_generator,
    epochs=30, 
    validation_data=validation_generator
)

# 6. SAVE
model.save('intel_scene_model.h5')
print("\nModel saved as 'intel_scene_model.h5'")