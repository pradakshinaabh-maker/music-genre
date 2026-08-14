import os
import librosa
import numpy as np
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ==========================================
# SETTINGS
# ==========================================

DATASET_PATH = "dataset/genres_original"

MODEL_FILE = "music_genre_model_v2.pkl"
SCALER_FILE = "scaler_v2.pkl"
EVALUATION_FILE = "evaluation_results_v2.pkl"


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
# SAME 61 FEATURES USED FOR V2
# ==========================================

def extract_features(file_path):

    try:

        y, sr = librosa.load(
            file_path,
            duration=30,
            mono=True
        )

        features = []

        # ----------------------------------
        # 1. MFCC
        # ----------------------------------

        mfcc = librosa.feature.mfcc(
            y=y,
            sr=sr,
            n_mfcc=13
        )

        features.extend(
            np.mean(
                mfcc,
                axis=1
            )
        )

        features.extend(
            np.std(
                mfcc,
                axis=1
            )
        )


        # ----------------------------------
        # 2. CHROMA
        # ----------------------------------

        chroma = librosa.feature.chroma_stft(
            y=y,
            sr=sr
        )

        features.extend(
            np.mean(
                chroma,
                axis=1
            )
        )

        features.extend(
            np.std(
                chroma,
                axis=1
            )
        )


        # ----------------------------------
        # 3. SPECTRAL CENTROID
        # ----------------------------------

        spectral_centroid = (
            librosa.feature.spectral_centroid(
                y=y,
                sr=sr
            )
        )

        features.append(
            np.mean(
                spectral_centroid
            )
        )

        features.append(
            np.std(
                spectral_centroid
            )
        )


        # ----------------------------------
        # 4. SPECTRAL BANDWIDTH
        # ----------------------------------

        spectral_bandwidth = (
            librosa.feature.spectral_bandwidth(
                y=y,
                sr=sr
            )
        )

        features.append(
            np.mean(
                spectral_bandwidth
            )
        )

        features.append(
            np.std(
                spectral_bandwidth
            )
        )


        # ----------------------------------
        # 5. SPECTRAL ROLLOFF
        # ----------------------------------

        spectral_rolloff = (
            librosa.feature.spectral_rolloff(
                y=y,
                sr=sr
            )
        )

        features.append(
            np.mean(
                spectral_rolloff
            )
        )

        features.append(
            np.std(
                spectral_rolloff
            )
        )


        # ----------------------------------
        # 6. ZERO CROSSING RATE
        # ----------------------------------

        zcr = librosa.feature.zero_crossing_rate(
            y
        )

        features.append(
            np.mean(
                zcr
            )
        )

        features.append(
            np.std(
                zcr
            )
        )


        # ----------------------------------
        # 7. RMS ENERGY
        # ----------------------------------

        rms = librosa.feature.rms(
            y=y
        )

        features.append(
            np.mean(
                rms
            )
        )

        features.append(
            np.std(
                rms
            )
        )


        # ----------------------------------
        # 8. TEMPO
        # ----------------------------------

        tempo, _ = librosa.beat.beat_track(
            y=y,
            sr=sr
        )

        tempo_value = float(
            np.asarray(
                tempo
            ).reshape(-1)[0]
        )

        features.append(
            tempo_value
        )


        return np.array(
            features
        )


    except Exception as e:

        print(
            "Error processing:",
            file_path
        )

        print(e)

        return None


# ==========================================
# START
# ==========================================

print()
print("======================================")
print("📊 V2 MODEL EVALUATION")
print("======================================")
print()

print(
    "Preparing GTZAN evaluation dataset..."
)

print()


# ==========================================
# DATASET CHECK
# ==========================================

if not os.path.exists(
    DATASET_PATH
):

    print(
        "❌ Dataset folder not found!"
    )

    exit()


# ==========================================
# CREATE DATA
# ==========================================

X = []
y = []


for genre in genres:

    genre_path = os.path.join(
        DATASET_PATH,
        genre
    )

    print(
        "🎧 Processing:",
        genre
    )


    if not os.path.exists(
        genre_path
    ):

        print(
            "⚠️ Folder not found"
        )

        continue


    files = os.listdir(
        genre_path
    )


    wav_files = [
        file
        for file in files
        if file.lower().endswith(
            ".wav"
        )
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

            X.append(
                features
            )

            y.append(
                genre
            )


# ==========================================
# NUMPY
# ==========================================

X = np.array(
    X
)

y = np.array(
    y
)


print()
print(
    "Total samples:",
    len(X)
)

print(
    "Feature shape:",
    X.shape
)


if len(X) == 0:

    print(
        "❌ No data available."
    )

    exit()


# ==========================================
# TRAIN / TEST SPLIT
# ==========================================

print()
print(
    "Creating 80/20 stratified train-test split..."
)


X_train, X_test, y_train, y_test = (
    train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )
)


print(
    "Training samples:",
    len(X_train)
)

print(
    "Testing samples:",
    len(X_test)
)


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
# TRAIN EVALUATION MODEL
# SAME V2 RANDOM FOREST SETTINGS
# ==========================================

print()
print(
    "🌲 Training evaluation Random Forest..."
)


evaluation_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)


evaluation_model.fit(
    X_train_scaled,
    y_train
)


# ==========================================
# PREDICTIONS
# ==========================================

print()
print(
    "🔍 Evaluating test set..."
)


y_pred = evaluation_model.predict(
    X_test_scaled
)


# ==========================================
# ACCURACY
# ==========================================

accuracy = accuracy_score(
    y_test,
    y_pred
)


# ==========================================
# CLASSIFICATION REPORT
# ==========================================

report = classification_report(
    y_test,
    y_pred,
    labels=genres,
    output_dict=True,
    zero_division=0
)


# ==========================================
# CONFUSION MATRIX
# ==========================================

matrix = confusion_matrix(
    y_test,
    y_pred,
    labels=genres
)


# ==========================================
# PRINT RESULTS
# ==========================================

print()
print("======================================")
print("📊 EVALUATION RESULTS")
print("======================================")
print()


print(
    f"Accuracy: {accuracy * 100:.2f}%"
)


print()
print(
    "Per-Genre Performance:"
)

print(
    "-" * 70
)


for genre in genres:

    if genre in report:

        precision = (
            report[genre]["precision"]
        )

        recall = (
            report[genre]["recall"]
        )

        f1 = (
            report[genre]["f1-score"]
        )

        support = (
            report[genre]["support"]
        )

        print(
            f"{genre.title():12}"
            f" Precision: {precision:.3f}"
            f"  Recall: {recall:.3f}"
            f"  F1: {f1:.3f}"
            f"  N: {support}"
        )


print()
print(
    "Macro F1:",
    round(
        report["macro avg"]["f1-score"],
        3
    )
)

print(
    "Weighted F1:",
    round(
        report["weighted avg"]["f1-score"],
        3
    )
)


print()
print(
    "Confusion Matrix:"
)

print(
    matrix
)


# ==========================================
# SAVE RESULTS
# ==========================================

evaluation_results = {

    "accuracy":
        float(accuracy),

    "classification_report":
        report,

    "confusion_matrix":
        matrix,

    "genres":
        genres,

    "test_samples":
        int(len(X_test)),

    "train_samples":
        int(len(X_train))
}


with open(
    EVALUATION_FILE,
    "wb"
) as file:

    pickle.dump(
        evaluation_results,
        file
    )


# ==========================================
# FINISHED
# ==========================================

print()
print("======================================")
print("✅ EVALUATION COMPLETED")
print("======================================")
print()

print(
    "Evaluation results saved as:"
)

print(
    EVALUATION_FILE
)

print()
print(
    "🎵 Model evaluation is ready for the dashboard!"
)