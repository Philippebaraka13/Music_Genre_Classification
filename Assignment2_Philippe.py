import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Parameters
audio_dir = 'C:/Users/User/Documents/GenreDataSet/'
spectrogram_dir = 'spectrograms_small/'
os.makedirs(spectrogram_dir, exist_ok=True)

# Generate mel spectrograms and save as images
def generate_mel_spectrogram(audio_path, output_image_path):
    try:
        y, sr = librosa.load(audio_path, duration=30)
        mel = librosa.feature.melspectrogram(y=y, sr=sr)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        plt.figure(figsize=(2, 2))
        plt.axis('off')
        librosa.display.specshow(mel_db, sr=sr, x_axis='time', y_axis='mel')
        plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0)
        plt.close()
    except Exception as e:
        print(f"[ERROR] Skipping {audio_path} - {e}")

# Preprocess and save spectrograms
image_paths = []
labels = []
for genre in os.listdir(audio_dir):
    genre_folder = os.path.join(audio_dir, genre)
    if not os.path.isdir(genre_folder):
        continue
    for file in os.listdir(genre_folder):
        if file.endswith('.wav'):
            audio_path = os.path.join(genre_folder, file)
            image_path = os.path.join(spectrogram_dir, f"{genre}_{file.replace('.wav', '.png')}")
            generate_mel_spectrogram(audio_path, image_path)
            if os.path.exists(image_path):  # Only use successfully created images
                image_paths.append(image_path)
                labels.append(genre)

# Load and preprocess images
def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image: {path}")
    img = cv2.resize(img, (128, 128)) / 255.0
    return img

X = np.array([load_image(p) for p in image_paths]).reshape(-1, 128, 128, 1)
le = LabelEncoder()
y_encoded = le.fit_transform(labels)
y = to_categorical(y_encoded)

# Split dataset
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Build CNN model
def create_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Dropout(0.3),

        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.3),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = create_model((128, 128, 1), len(le.classes_))
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, batch_size=32)

# Evaluate model
y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)
y_true_labels = np.argmax(y_test, axis=1)

print(classification_report(y_true_labels, y_pred_labels, target_names=le.classes_))
cm = confusion_matrix(y_true_labels, y_pred_labels)
sns.heatmap(cm, annot=True, fmt='d', xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
