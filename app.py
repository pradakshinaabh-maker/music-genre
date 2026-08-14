import streamlit as st
import librosa
import numpy as np
import pickle
import tempfile
import os
import json
import soundfile as sf
import matplotlib.pyplot as plt

from datetime import datetime
from sklearn.decomposition import PCA


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="RHYTHM IQ | Music Intelligence",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# FILES
# =========================================================

MODEL_FILE = "music_genre_model_v2.pkl"
SCALER_FILE = "scaler_v2.pkl"
NOVELTY_FILE = "novelty_reference_v2.pkl"
SIMILARITY_FILE = "similarity_reference_v2.pkl"
EVALUATION_FILE = "evaluation_results_v2.pkl"
PLAYLIST_FILE = "playlists.json"


# =========================================================
# GENRES
# =========================================================

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


# =========================================================
# SIMPLE CSS
# ONLY CSS - NO HTML CONTENT
# =========================================================

st.markdown(
    """
    <style>

    .stApp {
        background-color: #0b1020;
    }

    [data-testid="stSidebar"] {
        background-color: #080d19;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# LOAD MODEL + REFERENCES
# =========================================================

try:

    with open(
        MODEL_FILE,
        "rb"
    ) as file:

        model = pickle.load(file)


    with open(
        SCALER_FILE,
        "rb"
    ) as file:

        scaler = pickle.load(file)


    with open(
        NOVELTY_FILE,
        "rb"
    ) as file:

        novelty_reference = pickle.load(file)


    with open(
        SIMILARITY_FILE,
        "rb"
    ) as file:

        similarity_reference = pickle.load(file)


    with open(
        EVALUATION_FILE,
        "rb"
    ) as file:

        evaluation_results = pickle.load(file)


    model_loaded = True


except Exception as e:

    model_loaded = False
    model_error = str(e)


# =========================================================
# PLAYLIST FUNCTIONS
# =========================================================

def load_playlists():

    if not os.path.exists(
        PLAYLIST_FILE
    ):

        return []


    try:

        with open(
            PLAYLIST_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(file)


        if isinstance(
            data,
            list
        ):

            return data


        return []


    except Exception:

        return []


def save_playlists(
    playlists
):

    with open(
        PLAYLIST_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            playlists,
            file,
            indent=4,
            ensure_ascii=False
        )


def add_playlist(
    playlist
):

    playlists = load_playlists()

    playlists.append(
        playlist
    )

    save_playlists(
        playlists
    )


def delete_playlist(
    index
):

    playlists = load_playlists()

    if (
        0 <= index <
        len(playlists)
    ):

        playlists.pop(
            index
        )

        save_playlists(
            playlists
        )


# =========================================================
# SESSION STATE
# =========================================================

if (
    "loaded_playlist"
    not in st.session_state
):

    st.session_state.loaded_playlist = None


# =========================================================
# FEATURE EXTRACTION
# SAME 61 FEATURES
# =========================================================

def extract_features(
    file_path
):

    y, sr = librosa.load(
        file_path,
        duration=30,
        mono=True
    )

    features = []


    # -----------------------------------------------------
    # MFCC
    # -----------------------------------------------------

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


    # -----------------------------------------------------
    # CHROMA
    # -----------------------------------------------------

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


    # -----------------------------------------------------
    # SPECTRAL CENTROID
    # -----------------------------------------------------

    centroid = librosa.feature.spectral_centroid(
        y=y,
        sr=sr
    )

    features.append(
        np.mean(
            centroid
        )
    )

    features.append(
        np.std(
            centroid
        )
    )


    # -----------------------------------------------------
    # SPECTRAL BANDWIDTH
    # -----------------------------------------------------

    bandwidth = librosa.feature.spectral_bandwidth(
        y=y,
        sr=sr
    )

    features.append(
        np.mean(
            bandwidth
        )
    )

    features.append(
        np.std(
            bandwidth
        )
    )


    # -----------------------------------------------------
    # SPECTRAL ROLLOFF
    # -----------------------------------------------------

    rolloff = librosa.feature.spectral_rolloff(
        y=y,
        sr=sr
    )

    features.append(
        np.mean(
            rolloff
        )
    )

    features.append(
        np.std(
            rolloff
        )
    )


    # -----------------------------------------------------
    # ZERO CROSSING RATE
    # -----------------------------------------------------

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


    # -----------------------------------------------------
    # RMS ENERGY
    # -----------------------------------------------------

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


    # -----------------------------------------------------
    # TEMPO
    # -----------------------------------------------------

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


# =========================================================
# ADAPTIVE NOISE REDUCTION
# =========================================================

def reduce_noise(
    y,
    sr,
    strength=0.3
):

    stft = librosa.stft(
        y
    )

    magnitude = np.abs(
        stft
    )

    phase = np.angle(
        stft
    )


    frame_energy = np.mean(
        magnitude ** 2,
        axis=0
    )


    noise_threshold = np.percentile(
        frame_energy,
        15
    )


    noise_frames = magnitude[
        :,
        frame_energy <= noise_threshold
    ]


    if noise_frames.shape[1] == 0:

        noise_profile = np.mean(
            magnitude,
            axis=1,
            keepdims=True
        )

    else:

        noise_profile = np.mean(
            noise_frames,
            axis=1,
            keepdims=True
        )


    cleaned_magnitude = (
        magnitude -
        strength * noise_profile
    )


    cleaned_magnitude = np.maximum(
        cleaned_magnitude,
        0
    )


    cleaned_stft = (
        cleaned_magnitude *
        np.exp(
            1j * phase
        )
    )


    cleaned_audio = librosa.istft(
        cleaned_stft,
        length=len(y)
    )


    max_value = np.max(
        np.abs(
            cleaned_audio
        )
    )


    if max_value > 1:

        cleaned_audio = (
            cleaned_audio /
            max_value
        )


    return cleaned_audio


# =========================================================
# NOVELTY DETECTION
# =========================================================

def calculate_novelty(
    scaled_features
):

    training_features = (
        novelty_reference[
            "training_features_scaled"
        ]
    )


    threshold = float(
        novelty_reference[
            "novelty_threshold"
        ]
    )


    differences = (
        training_features -
        scaled_features
    )


    distances = np.sqrt(
        np.sum(
            differences ** 2,
            axis=1
        )
    )


    nearest_distance = float(
        np.min(
            distances
        )
    )


    similarity = (
        100 *
        np.exp(
            -nearest_distance /
            max(
                threshold,
                1e-8
            )
        )
    )


    similarity = float(
        max(
            0,
            min(
                100,
                similarity
            )
        )
    )


    is_novel = (
        nearest_distance >
        threshold
    )


    return (
        nearest_distance,
        threshold,
        similarity,
        is_novel
    )


# =========================================================
# GENRE SIMILARITY MAP
# =========================================================

def create_similarity_map(
    training_features,
    training_labels,
    song_features
):

    pca = PCA(
        n_components=2,
        random_state=42
    )


    training_2d = pca.fit_transform(
        training_features
    )


    song_2d = pca.transform(
        song_features.reshape(
            1,
            -1
        )
    )[0]


    fig, ax = plt.subplots(
        figsize=(11, 7)
    )


    unique_genres = np.unique(
        training_labels
    )


    for genre in unique_genres:

        mask = (
            training_labels ==
            genre
        )


        ax.scatter(
            training_2d[mask, 0],
            training_2d[mask, 1],
            label=genre.title(),
            alpha=0.22,
            s=25
        )


    ax.scatter(
        song_2d[0],
        song_2d[1],
        marker="*",
        s=420,
        label="Your Song"
    )


    ax.set_title(
        "Genre Similarity Map"
    )


    ax.set_xlabel(
        "Acoustic Dimension 1"
    )


    ax.set_ylabel(
        "Acoustic Dimension 2"
    )


    ax.grid(
        alpha=0.18
    )


    ax.legend(
        bbox_to_anchor=(
            1.02,
            1
        ),
        loc="upper left"
    )


    fig.tight_layout()


    return fig


# =========================================================
# TITLE
# =========================================================

st.title(
    "🎵 RHYTHM IQ"
)

st.subheader(
    "AI-powered music intelligence beyond genre classification."
)


header1, header2, header3 = st.columns(3)


with header1:

    st.info(
        "🤖 V2 Genre Engine"
    )


with header2:

    st.info(
        "🧬 61 Acoustic Features"
    )


with header3:

    st.info(
        "🧠 Explainable AI"
    )


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header(
        "🧠 Intelligence Pipeline"
    )


    pipeline_items = [
        "🎧 Audio Upload",
        "🎚 Quality Analysis",
        "🎼 61 Acoustic Features",
        "🤖 V2 Genre Classification",
        "📊 Confidence Analysis",
        "🔍 Novelty Detection",
        "🗺️ Genre Similarity",
        "🧹 Noise Analysis",
        "🧬 Audio DNA",
        "🧠 Explainable AI",
        "🎧 Smart Playlist",
        "📊 Model Evaluation"
    ]


    for item in pipeline_items:

        st.write(
            item
        )


    st.divider()


    # =====================================================
    # MY PLAYLISTS
    # =====================================================

    st.header(
        "🎧 My Playlists"
    )


    playlists = load_playlists()


    if not playlists:

        st.caption(
            "No saved playlists yet."
        )

    else:

        playlist_names = [
            playlist["name"]
            for playlist in playlists
        ]


        selected_playlist = st.selectbox(
            "Select playlist",
            playlist_names,
            key="playlist_selector"
        )


        selected_index = (
            playlist_names.index(
                selected_playlist
            )
        )


        load_col, delete_col = st.columns(2)


        with load_col:

            load_clicked = st.button(
                "📂 Load",
                key="load_playlist",
                use_container_width=True
            )


        with delete_col:

            delete_clicked = st.button(
                "🗑️ Delete",
                key="delete_playlist",
                use_container_width=True
            )


        if load_clicked:

            st.session_state.loaded_playlist = (
                playlists[
                    selected_index
                ]
            )

            st.rerun()


        if delete_clicked:

            delete_playlist(
                selected_index
            )

            st.session_state.loaded_playlist = None

            st.success(
                "Playlist deleted."
            )

            st.rerun()


    st.divider()


    if model_loaded:

        st.success(
            "🤖 V2 Model Ready"
        )

        st.success(
            "🔍 Novelty Ready"
        )

        st.success(
            "🗺️ Similarity Ready"
        )

        st.success(
            "📊 Evaluation Ready"
        )


# =========================================================
# MODEL CHECK
# =========================================================

if not model_loaded:

    st.error(
        "❌ Required files could not be loaded."
    )

    st.code(
        model_error
    )

    st.stop()


# =========================================================
# LOADED PLAYLIST
# =========================================================

if (
    st.session_state.loaded_playlist
    is not None
):

    loaded_playlist = (
        st.session_state.loaded_playlist
    )


    st.header(
        "📂 Loaded Playlist"
    )


    st.success(
        loaded_playlist["name"]
    )


    lp1, lp2, lp3 = st.columns(3)


    with lp1:

        st.metric(
            "🎵 Genre",
            loaded_playlist["genre"].title()
        )


    with lp2:

        st.metric(
            "🎯 Confidence",
            f"{loaded_playlist['confidence']:.2f}%"
        )


    with lp3:

        st.metric(
            "🥁 Tempo",
            f"{loaded_playlist['tempo']:.1f} BPM"
        )


    st.write(
        f"**Source Song:** "
        f"{loaded_playlist['source_song']}"
    )


    st.write(
        f"**Created:** "
        f"{loaded_playlist['created_at']}"
    )


    st.write(
        f"**Profile:** "
        f"{loaded_playlist['profile']}"
    )


    st.subheader(
        "🎧 Recommended Styles"
    )


    for item in loaded_playlist[
        "recommendations"
    ]:

        st.write(
            f"• {item}"
        )


    st.divider()


# =========================================================
# UPLOAD
# =========================================================

st.header(
    "🎧 Analyze a Song"
)


st.write(
    "Upload a WAV, MP3, OGG or FLAC file to generate "
    "a complete music intelligence profile."
)


uploaded_file = st.file_uploader(
    "Choose your audio file",
    type=[
        "wav",
        "mp3",
        "ogg",
        "flac"
    ]
)


# =========================================================
# PROCESS
# =========================================================

if uploaded_file is not None:

    st.success(
        "✅ Audio uploaded successfully!"
    )


    st.audio(
        uploaded_file
    )


    # =====================================================
    # FILE INFO
    # =====================================================

    st.header(
        "📁 File Information"
    )


    file_size_kb = (
        uploaded_file.size /
        1024
    )


    info1, info2 = st.columns(2)


    with info1:

        st.metric(
            "File",
            uploaded_file.name
        )


    with info2:

        st.metric(
            "Size",
            f"{file_size_kb:.2f} KB"
        )


    # =====================================================
    # TEMP FILE
    # =====================================================

    extension = os.path.splitext(
        uploaded_file.name
    )[1]


    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=extension
    ) as temp_file:

        temp_file.write(
            uploaded_file.getbuffer()
        )

        temp_audio_path = (
            temp_file.name
        )


    try:

        # =================================================
        # LOAD AUDIO
        # =================================================

        y, sr = librosa.load(
            temp_audio_path,
            sr=None,
            mono=True
        )


        # =================================================
        # BASIC AUDIO DATA
        # =================================================

        duration = librosa.get_duration(
            y=y,
            sr=sr
        )


        rms_signal = np.sqrt(
            np.mean(
                y ** 2
            )
        )


        peak = np.max(
            np.abs(
                y
            )
        )


        rms_db = (
            20 *
            np.log10(
                rms_signal +
                1e-10
            )
        )


        zcr_basic = float(
            np.mean(
                librosa.feature.zero_crossing_rate(
                    y
                )
            )
        )


        # =================================================
        # QUALITY SCORE
        # =================================================

        quality_score = 100


        if rms_signal < 0.01:

            quality_score -= 40

        elif rms_signal < 0.03:

            quality_score -= 20


        if zcr_basic > 0.25:

            quality_score -= 15


        if sr < 16000:

            quality_score -= 20


        quality_score = max(
            0,
            min(
                100,
                quality_score
            )
        )


        # =================================================
        # AUDIO QUALITY
        # =================================================

        st.header(
            "🎚 Audio Quality"
        )


        q1, q2, q3, q4 = st.columns(4)


        with q1:

            st.metric(
                "⏱ Duration",
                f"{duration:.2f}s"
            )


        with q2:

            st.metric(
                "🎚 Sample Rate",
                f"{sr} Hz"
            )


        with q3:

            st.metric(
                "🔊 Signal",
                f"{rms_db:.2f} dB"
            )


        with q4:

            st.metric(
                "💯 Quality",
                f"{quality_score}/100"
            )


        st.progress(
            quality_score / 100
        )


        if quality_score >= 70:

            st.success(
                "🟢 Good Audio — suitable for analysis."
            )

        elif quality_score >= 40:

            st.warning(
                "🟡 Audio needs attention."
            )

        else:

            st.error(
                "🔴 Low audio quality."
            )


        # =================================================
        # FEATURE EXTRACTION
        # =================================================

        st.header(
            "🎼 Acoustic Features"
        )


        features = extract_features(
            temp_audio_path
        )


        st.success(
            "✅ 61 acoustic features extracted"
        )


        scaled_features = scaler.transform(
            features.reshape(
                1,
                -1
            )
        )


        # =================================================
        # GENRE PREDICTION
        # =================================================

        probabilities = model.predict_proba(
            scaled_features
        )[0]


        probability_map = dict(
            zip(
                model.classes_,
                probabilities
            )
        )


        sorted_genres = sorted(
            probability_map.items(),
            key=lambda x: x[1],
            reverse=True
        )


        primary_genre = (
            sorted_genres[0][0]
        )


        primary_confidence = (
            sorted_genres[0][1] *
            100
        )


        # =================================================
        # MAIN GENRE RESULT
        # =================================================

        st.header(
            "🤖 AI Genre Prediction"
        )


        result_left, result_right = st.columns(
            [1, 2]
        )


        with result_left:

            st.subheader(
                "🎵 Primary Genre"
            )


            st.success(
                primary_genre.upper()
            )


            st.metric(
                "🎯 Model Probability",
                f"{primary_confidence:.2f}%"
            )


        with result_right:

            st.subheader(
                "📊 Genre Probability"
            )


            for genre, probability in sorted_genres:

                st.write(
                    f"**{genre.title()}** — "
                    f"{probability * 100:.2f}%"
                )


                st.progress(
                    float(probability)
                )


        # =================================================
        # RELIABILITY
        # =================================================

        st.subheader(
            "🧠 Prediction Reliability"
        )


        if primary_confidence >= 70:

            reliability_label = (
                "🟢 High Confidence"
            )


            st.success(
                "The model has a strong genre match."
            )


        elif primary_confidence >= 45:

            reliability_label = (
                "🟡 Mixed / Uncertain"
            )


            st.warning(
                "The song contains mixed genre characteristics."
            )


        else:

            reliability_label = (
                "🔴 Low Confidence"
            )


            st.error(
                "The model does not have a strong match."
            )


        # =================================================
        # NOVELTY
        # =================================================

        st.header(
            "🔍 Feature-Space Novelty Detection"
        )


        (
            nearest_distance,
            novelty_threshold,
            feature_similarity,
            is_novel
        ) = calculate_novelty(
            scaled_features[0]
        )


        if is_novel:

            st.error(
                "⚠️ Potentially Novel Audio"
            )


            st.write(
                "The acoustic representation is relatively "
                "far from the learned training feature space."
            )


        else:

            st.success(
                "✅ Familiar Training-Space Audio"
            )


            st.write(
                "The acoustic representation lies within "
                "the learned training feature space."
            )


        n1, n2, n3 = st.columns(3)


        with n1:

            st.metric(
                "🔬 Nearest Distance",
                f"{nearest_distance:.3f}"
            )


        with n2:

            st.metric(
                "📏 Reference Threshold",
                f"{novelty_threshold:.3f}"
            )


        with n3:

            st.metric(
                "🎯 Feature Similarity",
                f"{feature_similarity:.1f}%"
            )


        st.caption(
            "Novelty is estimated using nearest-neighbour "
            "distance in the standardized 61-feature training space."
        )


        # =================================================
        # SIMILARITY MAP
        # =================================================

        st.header(
            "🗺️ Genre Similarity Map"
        )


        st.write(
            "The map projects the 61-dimensional acoustic "
            "feature space into two dimensions for visualization."
        )


        training_features = (
            similarity_reference[
                "training_features_scaled"
            ]
        )


        training_labels = (
            similarity_reference[
                "training_labels"
            ]
        )


        similarity_figure = create_similarity_map(
            training_features,
            training_labels,
            scaled_features[0]
        )


        st.pyplot(
            similarity_figure
        )


        st.info(
            "⭐ The star represents your song. "
            "The map is a 2D PCA visualization of the "
            "61-dimensional acoustic feature space."
        )


        # =================================================
        # CROSS GENRE
        # =================================================

        st.header(
            "🔀 Cross-Genre Analysis"
        )


        second_genre = (
            sorted_genres[1][0]
        )


        second_confidence = (
            sorted_genres[1][1] *
            100
        )


        difference = (
            primary_confidence -
            second_confidence
        )


        if difference <= 10:

            st.warning(
                f"Possible "
                f"{primary_genre.title()}–"
                f"{second_genre.title()} "
                f"cross-genre track."
            )


        else:

            st.info(
                f"Primary characteristics strongly favor "
                f"{primary_genre.title()}."
            )


        # =================================================
        # TOP GENRES
        # =================================================

        st.header(
            "🎼 Top Genre Characteristics"
        )


        top_cols = st.columns(3)


        for index, (
            genre,
            probability
        ) in enumerate(
            sorted_genres[:3]
        ):


            with top_cols[index]:

                st.metric(
                    f"#{index + 1}",
                    genre.title(),
                    f"{probability * 100:.2f}%"
                )


        # =================================================
        # NOISE DETECTION
        # =================================================

        st.header(
            "🧹 Noise Detection"
        )


        spectral_flatness = (
            librosa.feature.spectral_flatness(
                y=y
            )
        )


        noise_indicator = float(
            np.mean(
                spectral_flatness
            )
        )


        if noise_indicator < 0.015:

            noise_level = (
                "🟢 Low Noise"
            )


            noise_score = min(
                100,
                (
                    noise_indicator /
                    0.015
                ) * 30
            )


            reduction_strength = 0.0


            noise_message = (
                "The audio is relatively clean. "
                "The original audio will be preserved."
            )


        elif noise_indicator < 0.05:

            noise_level = (
                "🟡 Moderate Noise"
            )


            noise_score = min(
                100,
                30 + (
                    (
                        noise_indicator -
                        0.015
                    ) /
                    0.035
                ) * 40
            )


            reduction_strength = 0.30


            noise_message = (
                "Moderate noise detected. "
                "Gentle reduction will be used."
            )


        else:

            noise_level = (
                "🔴 High Noise"
            )


            noise_score = min(
                100,
                70 + (
                    (
                        noise_indicator -
                        0.05
                    ) /
                    0.10
                ) * 30
            )


            reduction_strength = 0.50


            noise_message = (
                "Higher noise detected. "
                "Stronger reduction may help."
            )


        noise1, noise2, noise3 = st.columns(3)


        with noise1:

            st.metric(
                "🧹 Noise Level",
                noise_level
            )


        with noise2:

            st.metric(
                "📊 Noise Indicator",
                f"{noise_indicator:.4f}"
            )


        with noise3:

            st.metric(
                "⚠️ Noise Score",
                f"{noise_score:.1f}/100"
            )


        st.info(
            noise_message
        )


        # =================================================
        # NOISE REDUCTION
        # =================================================

        st.header(
            "🧹 Adaptive Noise Reduction"
        )


        clean_button = st.button(
            "🧹 Clean & Continue",
            use_container_width=True
        )


        if clean_button:

            with st.spinner(
                "Processing audio..."
            ):


                if reduction_strength == 0:

                    cleaned_audio = (
                        y.copy()
                    )


                    cleaning_action = (
                        "Low noise detected — "
                        "original audio preserved."
                    )


                else:

                    cleaned_audio = reduce_noise(
                        y,
                        sr,
                        strength=reduction_strength
                    )


                    cleaning_action = (
                        "Adaptive noise reduction applied."
                    )


            st.success(
                "✅ Audio processing completed!"
            )


            st.info(
                cleaning_action
            )


            cleaned_temp = (
                tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".wav"
                )
            )


            cleaned_audio_path = (
                cleaned_temp.name
            )


            cleaned_temp.close()


            sf.write(
                cleaned_audio_path,
                cleaned_audio,
                sr
            )


            with open(
                cleaned_audio_path,
                "rb"
            ) as audio_file:

                cleaned_audio_data = (
                    audio_file.read()
                )


            st.subheader(
                "🎧 Cleaned Audio Preview"
            )


            st.audio(
                cleaned_audio_data,
                format="audio/wav"
            )


            st.download_button(
                "⬇️ Download Cleaned Audio",
                data=cleaned_audio_data,
                file_name="cleaned_audio.wav",
                mime="audio/wav",
                use_container_width=True
            )


            cleaned_features = extract_features(
                cleaned_audio_path
            )


            cleaned_scaled_features = (
                scaler.transform(
                    cleaned_features.reshape(
                        1,
                        -1
                    )
                )
            )


            cleaned_probabilities = (
                model.predict_proba(
                    cleaned_scaled_features
                )[0]
            )


            cleaned_probability_map = dict(
                zip(
                    model.classes_,
                    cleaned_probabilities
                )
            )


            cleaned_sorted_genres = sorted(
                cleaned_probability_map.items(),
                key=lambda x: x[1],
                reverse=True
            )


            cleaned_primary_genre = (
                cleaned_sorted_genres[0][0]
            )


            cleaned_confidence = (
                cleaned_sorted_genres[0][1] *
                100
            )


            st.subheader(
                "🤖 Cleaned Audio Prediction"
            )


            c1, c2 = st.columns(2)


            with c1:

                st.metric(
                    "🎵 Genre",
                    cleaned_primary_genre.title()
                )


            with c2:

                st.metric(
                    "🎯 Confidence",
                    f"{cleaned_confidence:.2f}%"
                )


            st.subheader(
                "🔄 Original vs Cleaned"
            )


            original_col, cleaned_col = (
                st.columns(2)
            )


            with original_col:

                st.metric(
                    "Original",
                    primary_genre.title(),
                    f"{primary_confidence:.2f}%"
                )


            with cleaned_col:

                st.metric(
                    "Cleaned",
                    cleaned_primary_genre.title(),
                    f"{cleaned_confidence:.2f}%"
                )


            confidence_change = (
                cleaned_confidence -
                primary_confidence
            )


            if (
                primary_genre ==
                cleaned_primary_genre
            ):

                if abs(
                    confidence_change
                ) <= 5:

                    st.success(
                        "✅ Genre and confidence remained stable."
                    )

                else:

                    st.info(
                        "ℹ️ Genre remained consistent, "
                        "but confidence changed."
                    )


            else:

                st.warning(
                    "⚠️ Cleaning changed the predicted genre."
                )


            if os.path.exists(
                cleaned_audio_path
            ):

                os.remove(
                    cleaned_audio_path
                )


        # =================================================
        # AUDIO DNA
        # =================================================

        st.header(
            "🧬 Audio DNA"
        )


        spectral_centroid = (
            librosa.feature.spectral_centroid(
                y=y,
                sr=sr
            )
        )


        spectral_bandwidth = (
            librosa.feature.spectral_bandwidth(
                y=y,
                sr=sr
            )
        )


        rms_energy = librosa.feature.rms(
            y=y
        )


        tempo, _ = librosa.beat.beat_track(
            y=y,
            sr=sr
        )


        tempo_value = float(
            np.asarray(
                tempo
            ).reshape(-1)[0]
        )


        centroid_value = float(
            np.mean(
                spectral_centroid
            )
        )


        bandwidth_value = float(
            np.mean(
                spectral_bandwidth
            )
        )


        energy_value = float(
            np.mean(
                rms_energy
            )
        )


        zcr_value = float(
            np.mean(
                librosa.feature.zero_crossing_rate(
                    y
                )
            )
        )


        energy_score = min(
            100,
            (
                energy_value /
                0.2
            ) * 100
        )


        brightness_score = min(
            100,
            (
                centroid_value /
                5000
            ) * 100
        )


        rhythm_score = min(
            100,
            (
                zcr_value /
                0.2
            ) * 100
        )


        dna1, dna2, dna3, dna4 = (
            st.columns(4)
        )


        with dna1:

            st.metric(
                "🥁 Tempo",
                f"{tempo_value:.1f} BPM"
            )


        with dna2:

            st.metric(
                "⚡ Energy",
                f"{energy_score:.1f}/100"
            )


        with dna3:

            st.metric(
                "✨ Brightness",
                f"{brightness_score:.1f}/100"
            )


        with dna4:

            st.metric(
                "🥁 Rhythm",
                f"{rhythm_score:.1f}/100"
            )


        st.subheader(
            "🔬 Acoustic Measurements"
        )


        ac1, ac2 = st.columns(2)


        with ac1:

            st.write(
                f"**Spectral Centroid:** "
                f"{centroid_value:.2f} Hz"
            )


            st.write(
                f"**Spectral Bandwidth:** "
                f"{bandwidth_value:.2f} Hz"
            )


        with ac2:

            st.write(
                f"**RMS Energy:** "
                f"{energy_value:.4f}"
            )


            st.write(
                f"**Zero Crossing Rate:** "
                f"{zcr_value:.4f}"
            )


        # =================================================
        # AUDIO DNA INTERPRETATION
        # =================================================

        st.subheader(
            "🎼 Audio DNA Interpretation"
        )


        if tempo_value < 80:

            tempo_label = (
                "Low / Slow"
            )

        elif tempo_value < 130:

            tempo_label = (
                "Medium"
            )

        else:

            tempo_label = (
                "High / Fast"
            )


        if energy_score < 35:

            energy_label = "Low"

        elif energy_score < 70:

            energy_label = "Medium"

        else:

            energy_label = "High"


        if brightness_score < 35:

            brightness_label = "Low"

        elif brightness_score < 70:

            brightness_label = "Medium"

        else:

            brightness_label = "High"


        if rhythm_score < 35:

            rhythm_label = "Low"

        elif rhythm_score < 70:

            rhythm_label = "Medium"

        else:

            rhythm_label = "High"


        st.write(
            f"🎵 **Primary Genre:** "
            f"{primary_genre.title()}"
        )


        st.write(
            f"🥁 **Tempo:** {tempo_label}"
        )


        st.write(
            f"⚡ **Energy:** {energy_label}"
        )


        st.write(
            f"✨ **Brightness:** {brightness_label}"
        )


        st.write(
            f"🎶 **Rhythmic Activity:** "
            f"{rhythm_label}"
        )


        # =================================================
        # EXPLAINABLE AI
        # =================================================

        st.header(
            "🧠 Explainable AI"
        )


        st.subheader(
            f"🔎 Why did the AI predict "
            f"{primary_genre.title()}?"
        )


        st.write(
            f"• 🥁 Tempo: "
            f"{tempo_value:.1f} BPM "
            f"({tempo_label})"
        )


        st.write(
            f"• ⚡ Energy: "
            f"{energy_score:.1f}/100 "
            f"({energy_label})"
        )


        st.write(
            f"• ✨ Brightness: "
            f"{brightness_score:.1f}/100 "
            f"({brightness_label})"
        )


        st.write(
            f"• 🎶 Rhythm: "
            f"{rhythm_score:.1f}/100 "
            f"({rhythm_label})"
        )


        st.write(
            f"• 🔬 Spectral Centroid: "
            f"{centroid_value:.2f} Hz"
        )


        st.write(
            f"• 🔬 Spectral Bandwidth: "
            f"{bandwidth_value:.2f} Hz"
        )


        st.info(
            f"The V2 Random Forest considers "
            f"**{primary_genre.title()}** "
            f"the strongest match with "
            f"{primary_confidence:.2f}% probability."
        )


        st.caption(
            "This is a feature-based explanation, "
            "not a formal SHAP attribution analysis."
        )


        # =================================================
        # SMART PLAYLIST
        # =================================================

        st.header(
            "🎧 Smart Playlist"
        )


        genre_playlist = {

            "blues":
                "🎸 Blues Classics & Soulful Grooves",

            "classical":
                "🎻 Classical Focus & Instrumental Calm",

            "country":
                "🤠 Country Roads & Acoustic Stories",

            "disco":
                "🪩 Retro Dance & Disco Nights",

            "hiphop":
                "🎤 Hip-Hop Beats & Urban Flow",

            "jazz":
                "🎷 Smooth Jazz & Improvisation",

            "metal":
                "🤘 Heavy Metal & High-Energy Riffs",

            "pop":
                "🎤 Pop Hits & Catchy Melodies",

            "reggae":
                "🌴 Reggae Vibes & Island Grooves",

            "rock":
                "🎸 Rock Anthems & Guitar Energy"
        }


        playlist_1 = genre_playlist.get(
            primary_genre,
            "🎵 Genre-Matched Collection"
        )


        if energy_score >= 70:

            playlist_2 = (
                "🔥 High-Energy Workout & Party Mix"
            )

        elif energy_score >= 35:

            playlist_2 = (
                "🌟 Balanced Everyday Listening Mix"
            )

        else:

            playlist_2 = (
                "🌙 Calm & Relaxing Listening Mix"
            )


        if tempo_value >= 130:

            playlist_3 = (
                "⚡ Fast Tempo & Dance Momentum"
            )

        elif tempo_value >= 80:

            playlist_3 = (
                "🚗 Mid-Tempo Drive & Lifestyle Mix"
            )

        else:

            playlist_3 = (
                "🧘 Slow Tempo & Relaxation Mix"
            )


        if brightness_score >= 70:

            playlist_4 = (
                "✨ Bright & Crisp Soundscape"
            )

        elif brightness_score >= 35:

            playlist_4 = (
                "🎶 Balanced Tonal Soundscape"
            )

        else:

            playlist_4 = (
                "🌌 Warm & Dark Tonal Soundscape"
            )


        recommendations = [

            playlist_1,
            playlist_2,
            playlist_3,
            playlist_4

        ]


        for i, playlist in enumerate(
            recommendations,
            start=1
        ):

            st.write(
                f"**{i}. {playlist}**"
            )


        recommended_profile = (
            f"{primary_genre.title()} · "
            f"{tempo_label} Tempo · "
            f"{energy_label} Energy · "
            f"{brightness_label} Brightness"
        )


        st.success(
            f"🎧 Recommended profile: "
            f"{recommended_profile}"
        )


        # =================================================
        # SAVE PLAYLIST
        # =================================================

        st.subheader(
            "💾 Save This Playlist"
        )


        playlist_name = st.text_input(
            "Playlist name",
            value=(
                f"{primary_genre.title()} "
                "Intelligence Mix"
            ),
            key="playlist_name"
        )


        save_playlist_button = st.button(
            "💾 Save Playlist",
            use_container_width=True
        )


        if save_playlist_button:

            if playlist_name.strip() == "":

                st.warning(
                    "Please enter a playlist name."
                )

            else:

                new_playlist = {

                    "name":
                        playlist_name.strip(),

                    "source_song":
                        uploaded_file.name,

                    "genre":
                        primary_genre,

                    "confidence":
                        round(
                            primary_confidence,
                            2
                        ),

                    "tempo":
                        round(
                            tempo_value,
                            2
                        ),

                    "energy":
                        round(
                            energy_score,
                            2
                        ),

                    "brightness":
                        round(
                            brightness_score,
                            2
                        ),

                    "rhythm":
                        round(
                            rhythm_score,
                            2
                        ),

                    "profile":
                        recommended_profile,

                    "recommendations":
                        recommendations,

                    "created_at":
                        datetime.now().strftime(
                            "%Y-%m-%d %H:%M:%S"
                        )
                }


                add_playlist(
                    new_playlist
                )


                st.success(
                    f"✅ '{playlist_name.strip()}' "
                    "saved successfully!"
                )


                st.info(
                    "Open the sidebar → My Playlists "
                    "to load or delete it."
                )


        # =================================================
        # MODEL EVALUATION
        # =================================================

        st.header(
            "📊 Model Evaluation"
        )


        evaluation_accuracy = (
            evaluation_results[
                "accuracy"
            ] * 100
        )


        evaluation_report = (
            evaluation_results[
                "classification_report"
            ]
        )


        macro_f1 = (
            evaluation_report[
                "macro avg"
            ]["f1-score"] * 100
        )


        weighted_f1 = (
            evaluation_report[
                "weighted avg"
            ]["f1-score"] * 100
        )


        test_samples = (
            evaluation_results[
                "test_samples"
            ]
        )


        train_samples = (
            evaluation_results[
                "train_samples"
            ]
        )


        e1, e2, e3, e4 = (
            st.columns(4)
        )


        with e1:

            st.metric(
                "🎯 Accuracy",
                f"{evaluation_accuracy:.2f}%"
            )


        with e2:

            st.metric(
                "📊 Macro F1",
                f"{macro_f1:.2f}%"
            )


        with e3:

            st.metric(
                "📈 Weighted F1",
                f"{weighted_f1:.2f}%"
            )


        with e4:

            st.metric(
                "🧪 Test Samples",
                test_samples
            )


        st.write(
            f"**Training samples:** "
            f"{train_samples}"
        )


        st.progress(
            evaluation_accuracy / 100
        )


        with st.expander(
            "🎼 View Per-Genre Performance"
        ):

            genres_for_evaluation = (
                evaluation_results[
                    "genres"
                ]
            )


            for genre in genres_for_evaluation:

                if genre not in evaluation_report:

                    continue


                precision = (
                    evaluation_report[
                        genre
                    ]["precision"] *
                    100
                )


                recall = (
                    evaluation_report[
                        genre
                    ]["recall"] *
                    100
                )


                f1 = (
                    evaluation_report[
                        genre
                    ]["f1-score"] *
                    100
                )


                st.write(
                    f"**{genre.title()}**"
                )


                p1, p2, p3 = (
                    st.columns(3)
                )


                with p1:

                    st.metric(
                        "Precision",
                        f"{precision:.1f}%"
                    )


                with p2:

                    st.metric(
                        "Recall",
                        f"{recall:.1f}%"
                    )


                with p3:

                    st.metric(
                        "F1",
                        f"{f1:.1f}%"
                    )


                st.progress(
                    f1 / 100
                )


        with st.expander(
            "🔥 View Confusion Matrix"
        ):

            confusion_matrix_data = np.array(
                evaluation_results[
                    "confusion_matrix"
                ]
            )


            cm_genres = (
                evaluation_results[
                    "genres"
                ]
            )


            fig_cm, ax_cm = plt.subplots(
                figsize=(10, 7)
            )


            image = ax_cm.imshow(
                confusion_matrix_data
            )


            ax_cm.set_title(
                "V2 Genre Classification Confusion Matrix"
            )


            ax_cm.set_xlabel(
                "Predicted Genre"
            )


            ax_cm.set_ylabel(
                "Actual Genre"
            )


            ax_cm.set_xticks(
                range(
                    len(cm_genres)
                )
            )


            ax_cm.set_yticks(
                range(
                    len(cm_genres)
                )
            )


            ax_cm.set_xticklabels(
                [
                    genre.title()
                    for genre in cm_genres
                ],
                rotation=45,
                ha="right"
            )


            ax_cm.set_yticklabels(
                [
                    genre.title()
                    for genre in cm_genres
                ]
            )


            for row in range(
                confusion_matrix_data.shape[0]
            ):

                for col in range(
                    confusion_matrix_data.shape[1]
                ):

                    ax_cm.text(
                        col,
                        row,
                        str(
                            confusion_matrix_data[
                                row,
                                col
                            ]
                        ),
                        ha="center",
                        va="center"
                    )


            fig_cm.colorbar(
                image,
                ax=ax_cm
            )


            fig_cm.tight_layout()


            st.pyplot(
                fig_cm
            )


        st.caption(
            "Evaluation uses a stratified 80/20 train-test split."
        )


        # =================================================
        # FINAL SUMMARY
        # =================================================

        st.header(
            "📋 Intelligence Summary"
        )


        s1, s2, s3, s4 = (
            st.columns(4)
        )


        with s1:

            st.metric(
                "🎵 Genre",
                primary_genre.title()
            )


        with s2:

            st.metric(
                "🎯 Confidence",
                f"{primary_confidence:.1f}%"
            )


        with s3:

            st.metric(
                "💯 Quality",
                f"{quality_score}/100"
            )


        with s4:

            st.metric(
                "🧹 Noise",
                noise_level
            )


        st.write(
            f"🥁 **Tempo:** "
            f"{tempo_value:.1f} BPM"
        )


        st.write(
            f"⚡ **Energy:** "
            f"{energy_score:.1f}/100"
        )


        st.write(
            f"✨ **Brightness:** "
            f"{brightness_score:.1f}/100"
        )


        st.write(
            f"🎶 **Rhythm:** "
            f"{rhythm_score:.1f}/100"
        )


        st.write(
            f"🔍 **Feature Similarity:** "
            f"{feature_similarity:.1f}%"
        )


        st.write(
            f"📊 **Evaluation Accuracy:** "
            f"{evaluation_accuracy:.2f}%"
        )


        st.write(
            f"📈 **Macro F1:** "
            f"{macro_f1:.2f}%"
        )


        st.write(
            f"🧠 **Reliability:** "
            f"{reliability_label}"
        )


        st.success(
            "✅ Full audio intelligence analysis completed."
        )


        # =================================================
        # FOOTER
        # =================================================

        st.divider()


        st.caption(
            "Soniq AI • AI Music Intelligence System • "
            "V2 Acoustic Genre Intelligence Engine"
        )


    except Exception as e:

        st.error(
            f"❌ Could not analyze this audio: {e}"
        )


    finally:

        if os.path.exists(
            temp_audio_path
        ):

            os.remove(
                temp_audio_path
            )