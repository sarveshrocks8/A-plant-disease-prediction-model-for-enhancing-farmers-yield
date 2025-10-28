
# %%
import numpy as np
import pickle
import cv2
import os
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Activation
from keras.optimizers import Adam
from keras import backend as K

from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array

import kagglehub
import os

# Step 1: Download dataset
path = kagglehub.dataset_download("emmarex/plantdisease")
print("✅ Dataset downloaded at:", path)

# Step 2: Check what's inside
print("📁 Folders in dataset:", os.listdir(path))

# Step 3: Use this path in ImageDataGenerator or model pipeline
data_dir = os.path.join(path, "PlantVillage", "PlantVillage")
print("✅ Final data_dir:", data_dir)
print("📂 Class folders:", os.listdir(data_dir))
#data_dir = path + "/PlantVillage"  # adjust folder name if needed
# %%


img_height, img_width = 64, 64
batch_size = 128

# Train data generator
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2  # 20% data validation ke liye
)

train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# %%
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(64,64,3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(train_generator.num_classes, activation='softmax')
])


# Compile
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Summary
model.summary()

# %%
EPOCHS = 20

history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS
)

# %%
import matplotlib.pyplot as plt

# Accuracy plot
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.show()

# %%
model.save("plant_disease_cnn_model.h5")
print("✅ Model saved as 'plant_disease_cnn_model.h5'")

# %%
from tensorflow.keras.preprocessing import image 
import numpy as np

img_path = r"C:\Users\Dell\OneDrive\Pictures\Screenshots\potato_earlyblight_2.jpg"
img = image.load_img(img_path, target_size=(64, 64))
img_array = image.img_to_array(img)/255.0
img_array = np.expand_dims(img_array, axis=0)  # shape (1, height, width, 3)

pred = model.predict(img_array)
class_index = np.argmax(pred, axis=1)[0]
# Proper mapping: index → class
class_labels = {v: k for k, v in train_generator.class_indices.items()}

# Prediction
pred = model.predict(img_array)
class_index = np.argmax(pred, axis=1)[0]
class_label = class_labels[class_index]  # map index to class name

print("Predicted Class:", class_label)



# %%
