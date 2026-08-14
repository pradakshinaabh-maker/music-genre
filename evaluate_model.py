import os
import pickle
import librosa
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import matplotlib.pyplot as plt


# ==========================================
# SETTINGS
# ==========================================

DATASET_PATH = "dataset/genres_original"

genres = [
    "blues",
    "classical",
    "country",
    "disco",
    "hiphop",
    "jazz",
    "metal",
    "pop",
    "reggae",
    "rock"
]


# ==========================================
# FEATURE EXTRACTION
# ==========================================

def extract_features(file_path):

    try:

        y, sr = librosa.load(
            file_path,
            duration=30
        )

        mfcc = librosa.feature.mfcc(
            y=y,
            sr=sr,
            n_mfcc=13
        )

        mfcc_mean = np.mean(
            mfcc,
            axis=1
        )

        return mfcc_mean

    except Exception as e:

        print("Error:", file_path)
        return None


# ==========================================
# LOAD DATASET
# ==========================================

print()
print("======================================")
print("🎵 MUSIC GENRE MODEL EVALUATION")
print("======================================")
print()

X = []
y = []


for genre in genres:

    genre_path = os.path.join(
        DATASET_PATH,
        genre
    )

    print("Processing:", genre)

    if not os.path.exists(genre_path):

        print("❌ Folder not found:", genre)
        continue

    files = os.listdir(genre_path)

    wav_files = [
        file
        for file in files
        if file.lower().endswith(".wav")
    ]

    for file in wav_files:

        file_path = os.path.join(
            genre_path,
            file
        )

        features = extract_features(
            file_path
        )

        if features is not None:

            X.append(features)
            y.append(genre)


# ==========================================
# NUMPY ARRAYS
# ==========================================

X = np.array(X)
y = np.array(y)


print()
print("Total samples:", len(X))
print("Feature shape:", X.shape)


# ==========================================
# SPLIT DATA
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print()
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==========================================
# SCALE FEATURES
# ==========================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)


# ==========================================
# TRAIN MODEL
# ==========================================

print()
print("🌲 Training evaluation model...")
print()

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(
    X_train_scaled,
    y_train
)


# ==========================================
# PREDICTION
# ==========================================

y_pred = model.predict(
    X_test_scaled
)


# ==========================================
# ACCURACY
# ==========================================

accuracy = accuracy_score(
    y_test,
    y_pred
)


print()
print("======================================")
print("🎯 MODEL ACCURACY")
print("======================================")

print(
    f"Accuracy: {accuracy * 100:.2f}%"
)


# ==========================================
# CLASSIFICATION REPORT
# ==========================================

print()
print("======================================")
print("📊 CLASSIFICATION REPORT")
print("======================================")

print(
    classification_report(
        y_test,
        y_pred
    )
)


# ==========================================
# CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=genres
)


print()
print("======================================")
print("🔢 CONFUSION MATRIX")
print("======================================")

print(cm)


# ==========================================
# DISPLAY CONFUSION MATRIX
# ==========================================

plt.figure(figsize=(10, 8))

plt.imshow(cm)

plt.title(
    "Music Genre Classification - Confusion Matrix"
)

plt.xlabel(
    "Predicted Genre"
)

plt.ylabel(
    "Actual Genre"
)

plt.xticks(
    range(len(genres)),
    genres,
    rotation=45
)

plt.yticks(
    range(len(genres)),
    genres
)

plt.colorbar()

plt.tight_layout()

plt.show()


# ==========================================
# FINISHED
# ==========================================

print()
print("======================================")
print("✅ EVALUATION COMPLETED")
print("======================================")