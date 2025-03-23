# 🎵 Music Genre Classification Using CNN 

![Screenshot 2025-03-22 204926](https://github.com/user-attachments/assets/28ae6a7b-12ee-48ee-b213-cfdb8d2277cc)

This project implements a Convolutional Neural Network (CNN) to classify music tracks into genres using spectrogram images derived from audio files. It was completed as part of the CSCI 426: Introduction to Artificial Intelligence course at North Dakota State University.

---

## 📁 Project Structure
```
Ass2/
├── Assignment2_Philippe.py           # Main Python script
├── README.md                         # Project overview and instructions
├── spectrograms_small/              # Auto-generated spectrogram images
├── GenreDataSet/                    # Folder with audio files organized by genre
```

---

## 🧠 Objective
The goal is to use deep learning (CNNs) to automatically classify `.wav` music files into predefined genres by analyzing their visual audio features (mel-spectrograms).

---

## 🔧 Requirements

Install the required Python libraries:
```bash
pip install librosa matplotlib numpy opencv-python seaborn scikit-learn tensorflow ffmpeg-python
```

Make sure you have [**FFmpeg**](https://ffmpeg.org/download.html) installed and added to your system's PATH for proper audio decoding.

---

## 🚀 How to Run

1. Ensure your dataset folder (`GenreDataSet/`) contains subfolders like `rock/`, `pop/`, `jazz/`, etc.
2. Update this line in the script if needed:
```python
audio_dir = 'C:/Path/To/Your/GenreDataSet/'
```
3. Run the script:
```bash
python Assignment2_Philippe.py
```
4. The script will:
   - Convert `.wav` files to mel-spectrogram images
   - Preprocess and normalize the data
   - Train a CNN model
   - Print a classification report
   - Show a confusion matrix

---

## 🧪 Output

- Accuracy and loss across 20 training epochs
- Classification report (precision, recall, F1-score)
- Confusion matrix (genre vs. genre prediction)

---

## 📊 Model Architecture

- `Conv2D` (32 filters, 3x3) + ReLU + MaxPooling + Dropout
- `Conv2D` (64 filters, 3x3) + ReLU + MaxPooling + Dropout
- `Flatten` → `Dense(128)` → Dropout → `Dense(10)` + Softmax

**Loss Function:** Categorical Crossentropy  
**Optimizer:** Adam  
**Input Size:** 128x128 grayscale spectrograms  
**Epochs:** 20  
**Batch Size:** 32

---

## ⚠️ Notes

- Some audio files may fail to load and will be skipped with a warning.
- You can test your setup on a smaller subset first.
- For best results, ensure balanced and clean data per genre.

---

## 👨‍💻 Author

**Philippe Baraka**  

---

