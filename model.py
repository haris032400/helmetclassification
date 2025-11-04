# 📦 Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import load_img, img_to_array
import os

# 🛠️ Config
IMG_SIZE = 128
BATCH_SIZE = 16
EPOCHS = 20
DATASET_PATH = r'C:\Users\PMLS\Downloads\Project3/dataset_root'  # ✅ Change this to your actual dataset path

# 📁 Data Loader (No validation split)
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

# 🧠 CNN Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(train_generator.num_classes, activation='softmax')  # 2 classes Wear Helmet, Don't Wear Helmet
])

# 🧪 Compile
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 🚀 Train
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    verbose=1
)

# 📊 Plot Accuracy & Loss
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()

plt.tight_layout()
plt.show()

# ✅ Final evaluation on training data
loss, acc = model.evaluate(train_generator)
print(f"\n✅ Final Training Accuracy: {acc:.2f}")

# 🔍 Prediction Function
def predict_image(img_path, model, img_size=128, train_generator=None):
    if not os.path.exists(img_path):
        print(f"❌ Image not found: {img_path}")
        return

    img = load_img(img_path, target_size=(img_size, img_size))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)

    if train_generator:
        class_labels = list(train_generator.class_indices.keys())
        predicted_class = class_labels[class_index]
    else:
        predicted_class = str(class_index)

    confidence = prediction[0][class_index]
    print(f"\n🧠 Predicted: {predicted_class} (Confidence: {confidence:.2f})")

# 🖼️ Predict a new image
predict_image(r'C:\Users\PMLS\Downloads\onlu.jpeg', model, img_size=IMG_SIZE, train_generator=train_generator)  # ✅ Change path


model.save("helmet_model.h5")
