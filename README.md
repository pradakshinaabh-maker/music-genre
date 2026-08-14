AI-Based Music Intelligence, Genre Analysis and Smart Playlist System
Project Overview
The AI-Based Music Intelligence, Genre Analysis and Smart Playlist System is a machine-learning application designed to perform intelligent analysis of uploaded audio files.

Unlike a conventional music genre classifier that simply maps a song to a single genre, this system performs multiple stages of analysis, including audio quality assessment, noise-level estimation, acoustic feature extraction, genre classification, cross-genre analysis, prediction reliability estimation, audio profiling, and prediction explanation.

The system uses the GTZAN Genre Collection as its primary training dataset and extracts Mel-Frequency Cepstral Coefficients (MFCCs) along with additional acoustic features. A Random Forest Classifier is then trained to identify musical genres.

The application is implemented using Python and Streamlit, providing an interactive interface for uploading and analyzing audio files.

Key Features
1. Audio Quality Analysis
The system evaluates the quality of an uploaded audio file before performing genre classification.

The analysis considers:

RMS energy
Clipping
Dynamic range
Estimated noise level
Overall audio quality
A quality score from 0 to 100 is generated.

The system categorizes the input into three general conditions:

Good quality — suitable for analysis
Moderate quality — analysis can continue with a warning
Low quality — classification is stopped because the prediction may be unreliable
This prevents the system from blindly generating predictions from severely degraded audio.

2. Noise Detection and Preprocessing
When an audio file contains a significant amount of noise but remains usable, the application provides an option to perform basic preprocessing.

The current preprocessing pipeline applies a high-pass filter to reduce unwanted low-frequency components.

Uploaded Audio | v Noise and Quality Analysis | v High-Pass Filtering | v Normalized Audio | v Feature Extraction

The purpose of this process is to improve the input signal sufficiently for subsequent analysis.

3. Acoustic Feature Extraction
The system converts audio signals into numerical representations suitable for machine-learning algorithms.

Primary Features
MFCC — Mel-Frequency Cepstral Coefficients
Spectral Centroid
Spectral Bandwidth
Zero-Crossing Rate
Spectral Rolloff
Tempo
RMS Energy
The current implementation extracts 20 MFCC coefficients and additional spectral and rhythmic features.

Audio Signal | v Feature Extraction | +-- MFCC (20) +-- Spectral Centroid +-- Spectral Bandwidth +-- Zero-Crossing Rate +-- Spectral Rolloff +-- Tempo +-- RMS Energy | v Numerical Feature Vector

4. Machine Learning-Based Genre Classification
The extracted features are provided to a Random Forest Classifier.

The model is trained using the GTZAN dataset, which contains ten commonly used music genres:

Blues
Classical
Country
Disco
Hip-Hop
Jazz
Metal
Pop
Reggae
Rock
The model produces a probability distribution across the known genres.

Example:

Rock 67% Metal 21% Blues 7% Jazz 5%

The highest-probability genre is presented as the primary genre prediction.

5. Genre Probability Analysis
Instead of displaying only the predicted genre, the application presents the probability distribution for all recognized genres.

This provides additional information about the model's decision.

Example:

Rock 67.20% Metal 20.85% Blues 6.91% Jazz 3.14%

The probability map allows users to observe whether the model strongly favors one genre or produces a more distributed prediction.

6. Cross-Genre Analysis
Music can contain characteristics associated with multiple genres.

Therefore, the system does not display only the highest-probability genre. It also identifies the next strongest genre characteristics.

For example:

Primary Genre: Rock 46% Secondary: Metal 43%

The system can interpret a small difference between the highest and second-highest probabilities as an indication of possible cross-genre characteristics.

Example:

Rock — 46% Metal — 43%

Possible Rock-Metal Cross-Genre Track

This provides a more informative result than a strict single-label classification.

7. Prediction Reliability
The application categorizes predictions based on the probability of the highest-ranked genre.

High Confidence
Rock — 78%

Reliability: HIGH

Mixed or Uncertain
Rock — 46% Metal — 43%

Reliability: MIXED / UNCERTAIN

No Strong Match
Rock — 23% Jazz — 21% Blues — 19%

No strong genre match

This approach helps communicate model uncertainty instead of presenting every prediction as equally reliable.

8. Audio DNA Profile
The application generates an Audio DNA profile describing important acoustic characteristics of the uploaded track.

The current profile includes:

Tempo
Spectral Centroid
Spectral Bandwidth
RMS Energy
Zero-Crossing Rate
These characteristics are also visualized using a chart to provide a compact representation of the audio signal.

Example:

Audio DNA

Tempo : 124 BPM Spectral Energy : 0.08 Zero Crossing Rate : 0.09

The Audio DNA provides additional information beyond the predicted genre.

9. Explainable AI
The system provides a basic explanation of the model's prediction using Random Forest feature importance.

The extracted features are grouped into two categories:

MFCC characteristics
Other acoustic characteristics
The application calculates their relative contribution to the Random Forest feature importance.

Example:

MFCC Characteristics : 68.40% Other Acoustic Features : 31.60%

This provides a basic explanation of which feature groups influenced the classification.

10. Smart Playlist Categorization
Based on the analysis, the application determines a playlist category.

Examples include:

Rock Playlist Jazz Playlist Cross-Genre Discovery Unknown / Experimental

Songs exhibiting similar characteristics can therefore be conceptually organized according to their classification and cross-genre characteristics.

#Complete System Workflow

                AUDIO UPLOAD
                     |
                     v
           AUDIO QUALITY ANALYSIS
                     |
         +-----------+-----------+
         |           |           |
         v           v           v
       GOOD     MODERATE       LOW
         |           |           |
         |           v           v
         |     OPTIONAL       STOP
         |    PREPROCESSING     |
         |           |           |
         +-----------+           |
                     |           |
                     +-----------+
                          |
                          v
                FEATURE EXTRACTION
                          |
                          v
               MFCC + ACOUSTIC FEATURES
                          |
                          v
                RANDOM FOREST MODEL
                          |
                          v
                GENRE PROBABILITIES
                          |
           +--------------+--------------+
           |              |              |
           v              v              v
      PRIMARY GENRE   CROSS-GENRE    RELIABILITY
           |              |              |
           +--------------+--------------+
                          |
                          v
                   AUDIO DNA PROFILE
                          |
                          v
                 PREDICTION EXPLANATION
                          |
                          v
                PLAYLIST CATEGORIZATION
What Makes the System Different?
A conventional genre classification system generally follows:

Audio → Genre

This project follows a broader intelligence pipeline:

Audio ↓ Quality Assessment ↓ Noise Analysis ↓ Optional Preprocessing ↓ Feature Extraction ↓ Machine Learning Classification ↓ Genre Probability Analysis ↓ Primary Genre ↓ Cross-Genre Analysis ↓ Prediction Reliability ↓ Audio DNA ↓ Explainable Prediction ↓ Playlist Categorization

The main objective is therefore not simply to predict a genre, but to provide a more informative analysis of the uploaded music.

Problem Statement
Traditional music genre classification systems often assume that:

The input audio has sufficient quality.
Every song belongs clearly to one known genre.
A prediction should always be produced.
Genre overlap does not need to be represented.
Users only require a single genre label.
Real-world audio can be noisy, distorted, genre-blended, or difficult to classify confidently.

This project addresses these challenges by combining:

Audio Quality Analysis + Noise-Aware Processing + Acoustic Feature Extraction + Machine Learning Classification + Cross-Genre Analysis + Reliability Estimation + Audio Profiling + Explainable AI

Dataset
The project uses the GTZAN Genre Collection for model training.

The dataset contains ten music genres:

Genre
Blues
Classical
Country
Disco
Hip-Hop
Jazz
Metal
Pop
Reggae
Rock
The dataset should be organized as:

dataset/ └── genres_original/ ├── blues/ ├── classical/ ├── country/ ├── disco/ ├── hiphop/ ├── jazz/ ├── metal/ ├── pop/ ├── reggae/ └── rock/

The dataset is used only during the model-training stage.

Machine Learning Pipeline
The training process consists of the following stages:

GTZAN Dataset ↓ Audio Loading ↓ Feature Extraction ↓ Feature Matrix Creation ↓ Train/Test Split ↓ Random Forest Training ↓ Model Evaluation ↓ Trained Model ↓ genre_model.pkl

The dataset is divided into training and testing subsets using a stratified train-test split.

The current Random Forest configuration uses:

300 decision trees
Balanced class weighting
Fixed random state for reproducibility
Technology Stack
Component	Technology
Programming Language	Python
Machine Learning	Scikit-learn
Classification Algorithm	Random Forest
Audio Processing	Librosa
Numerical Processing	NumPy
Model Serialization	Joblib
Signal Processing	SciPy
Visualization	Matplotlib
User Interface	Streamlit
Training Dataset	GTZAN Genre Collection
Project Structure
AI-Music-Intelligence/ │ ├── dataset/ │ └── genres_original/ │ ├── blues/ │ ├── classical/ │ ├── country/ │ ├── disco/ │ ├── hiphop/ │ ├── jazz/ │ ├── metal/ │ ├── pop/ │ ├── reggae/ │ └── rock/ │ ├── models/ │ └── genre_model.pkl │ ├── train.py ├── app.py ├── requirements.txt └── README.md

The genre_model.pkl file is generated automatically after executing the training script.

Installation
Clone the repository and install the required Python packages.

pip install -r requirements.txt

The required dependencies are:

streamlit numpy librosa joblib scikit-learn scipy matplotlib

Model Training
Place the GTZAN dataset in:

dataset/genres_original/

Then execute:

python train.py

The training script will:

Load the audio files.
Extract MFCC and acoustic features.
Create the training and testing datasets.
Train the Random Forest classifier.
Evaluate the model.
Save the trained model.
The resulting model will be saved as:

models/genre_model.pkl

Running the Application
After the model has been generated, start the Streamlit application:

streamlit run app.py

The application allows the user to:

Upload an audio file.
Analyze audio quality.
Estimate noise level.
Perform optional preprocessing.
Extract acoustic features.
Predict the primary music genre.
View genre probabilities.
Analyze cross-genre characteristics.
View prediction reliability.
Inspect the Audio DNA profile.
View a basic explanation of the model prediction.
Determine a playlist category.
Supported Audio Formats
The application currently supports:

WAV
MP3
OGG
FLAC
For analysis, the uploaded audio is loaded at a sampling rate of 22,050 Hz and the first 30 seconds are analyzed.

Limitations
The current implementation has several limitations.

Dataset Limitations
The classifier is trained using the GTZAN dataset and therefore its predictions are limited to patterns represented in that dataset.

Genre Limitations
The model recognizes the genres represented in the training dataset. It does not provide unrestricted classification of every possible music genre.

Novelty Detection
The current "unknown" classification is threshold-based rather than a dedicated open-set or anomaly-detection model.

Noise Reduction
The current preprocessing uses a basic high-pass filter. It is not a full neural noise-removal system.

Explainability
The current explanation is based on Random Forest feature importance and does not provide a complete per-prediction SHAP explanation.

Audio Duration
Only the first 30 seconds of an uploaded audio file are analyzed.

Future Enhancements
Potential future improvements include:

Advanced neural noise reduction
CNN-based audio classification
Transformer-based music representation
Dedicated open-set/novelty detection
SHAP-based per-prediction explanations
More comprehensive audio features
Larger and more diverse training datasets
Real-time audio analysis
Persistent playlist storage
User-specific music recommendations
Spotify or other music-service integration where permitted by the service API
Improved cross-genre classification
Model comparison using Random Forest, XGBoost and neural networks
Project Objective
The objective of this project is to develop an intelligent music-analysis system that goes beyond conventional genre classification.

The system aims to answer several questions about an uploaded track:

Is the audio suitable for analysis?

Does it contain significant noise or degradation?

What genre does it most closely resemble?

Does it contain characteristics of multiple genres?

How reliable is the prediction?

What acoustic characteristics define the track?

Which feature groups influenced the model's decision?

How can the track be categorized for music discovery?

By combining audio processing, machine learning, uncertainty awareness, acoustic profiling and explainability, the project provides a broader approach to automated music intelligence.

License
This project is intended for educational and research purposes.

The GTZAN dataset is subject to its own licensing and usage conditions. Users should review the dataset's terms before redistribution or commercial use.
