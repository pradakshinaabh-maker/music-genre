import os
import librosa
import numpy as np
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


# ==========================================
# SETTINGS
# ==========================================

DATASET_PATH = "dataset/genres_original"

MODEL_FILE = "music_genre_model.pkl"
SCALER_FILE = "scaler.pkl"


# ==========================================
# EXTRACT MFCC FEATURES
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

        print("Error processing:", file_path)
        print(e)

        return None


# ==========================================
# START
# ==========================================

print()
print("======================================")
print("🎵 MUSIC GENRE CLASSIFIER")
print("======================================")
print()

print("Checking dataset...")
print("Dataset path:", DATASET_PATH)
print()


# ==========================================
# CHECK DATASET
# ==========================================

if not os.path.exists(DATASET_PATH):

    print("❌ Dataset folder not found!")
    print()
    print("Expected:")
    print("dataset/genres_original")
    exit()


# ==========================================
# GENRES
# ==========================================

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


X = []
y = []


# ==========================================
# PROCESS DATASET
# ==========================================

for genre in genres:

    genre_path = os.path.join(
        DATASET_PATH,
        genre
    )

    print()
    print("🎧 Processing:", genre)

    if not os.path.exists(genre_path):

        print("⚠️ Folder not found")
        continue

    files = os.listdir(genre_path)

    wav_files = [
        file
        for file in files
        if file.lower().endswith(".wav")
    ]

    print("Files found:", len(wav_files))


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
# CONVERT TO NUMPY
# ==========================================

X = np.array(X)
y = np.array(y)


print()
print("======================================")
print("FEATURE EXTRACTION COMPLETED")
print("======================================")

print("Total samples:", len(X))
print("Feature shape:", X.shape)


# ==========================================
# CHECK DATA
# ==========================================

if len(X) == 0:

    print()
    print("❌ No audio files were processed.")
    print("Please check the dataset.")
    exit()


# ==========================================
# SCALE FEATURES
# ==========================================

print()
print("Scaling features...")

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# ==========================================
# TRAIN RANDOM FOREST
# ==========================================

print()
print("🌲 Training Random Forest model...")
print("Please wait...")


model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(
    X_scaled,
    y
)


# ==========================================
# SAVE MODEL
# ==========================================

with open(
    MODEL_FILE,
    "wb"
) as file:

    pickle.dump(
        model,
        file
    )


# ==========================================
# SAVE SCALER
# ==========================================

with open(
    SCALER_FILE,
    "wb"
) as file:

    pickle.dump(
        scaler,
        file
    )


# ==========================================
# DONE
# ==========================================

print()
print("======================================")
print("✅ TRAINING COMPLETED!")
print("======================================")

print()
print("Model saved as:")
print(MODEL_FILE)

print()
print("Scaler saved as:")
print(SCALER_FILE)

print()
print("🎵 Genre classifier is ready!")