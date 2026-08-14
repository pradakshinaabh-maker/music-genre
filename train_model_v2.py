import streamlit as st
import librosa
import numpy as np
import pickle
import tempfile
import os
import soundfile as sf
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI Music Intelligence System",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 800;
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 17px;
        opacity: 0.75;
        margin-bottom: 30px;
    }

    .section-title {
        font-size: 28px;
        font-weight: 700;
        margin-top: 15px;
        margin-bottom: 10px;
    }

    .genre-card {
        padding: 24px;
        border-radius: 18px;
        border: 1px solid rgba(128, 128, 128, 0.25);
        text-align: center;
    }

    .genre-name {
        font-size: 32px;
        font-weight: 800;
    }

    .footer {
        text-align: center;
        opacity: 0.6;
        padding: 30px 0;
        font-size: 13px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# FILES
# =========================================================

MODEL_FILE = "music_genre_model_v2.pkl"
SCALER_FILE = "scaler_v2.pkl"
NOVELTY_FILE = "novelty_reference_v2.pkl"
SIMILARITY_FILE = "similarity_reference_v2.pkl"


# =========================================================
# LOAD MODEL FILES
# =========================================================

try:

    with open(
        MODEL_FILE,
        "rb"
    ) as file:

        model = pickle.load(
            file
        )


    with open(
        SCALER_FILE,
        "rb"
    ) as file:

        scaler = pickle.load(
            file
        )


    with open(
        NOVELTY_FILE,
        "rb"
    ) as file:

        novelty_reference = pickle.load(
            file
        )


    with open(
        SIMILARITY_FILE,
        "rb"
    ) as file:

        similarity_reference = pickle.load(
            file
        )


    model_loaded = True

except Exception as e:

    model_loaded = False
    model_error = str(e)


# =========================================================
# FEATURE EXTRACTION
# =========================================================

def extract_features(file_path):

    y, sr = librosa.load(
        file_path,
        duration=30,
        mono=True
    )

    features = []

    # MFCC

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


    # Chroma

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


    # Spectral Centroid

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


    # Spectral Bandwidth

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


    # Spectral Rolloff

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


    # Zero Crossing Rate

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


    # RMS

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


    # Tempo

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
        strength *
        noise_profile
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

    # PCA is fitted on training data only.
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
        figsize=(10, 7)
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
            alpha=0.25,
            s=25
        )

    ax.scatter(
        song_2d[0],
        song_2d[1],
        marker="*",
        s=350,
        label="Your Song"
    )

    ax.set_title(
        "🗺️ Genre Similarity Map"
    )

    ax.set_xlabel(
        "Acoustic Dimension 1"
    )

    ax.set_ylabel(
        "Acoustic Dimension 2"
    )

    ax.legend(
        bbox_to_anchor=(
            1.02,
            1
        ),
        loc="upper left"
    )

    ax.grid(
        alpha=0.2
    )

    return fig


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">'
    '🎵 AI Music Intelligence System'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Analyze • Classify • Explain • Understand Your Music'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header(
        "🧠 AI Intelligence Pipeline"
    )

    st.write(
        "🎧 Audio Upload"
    )

    st.write(
        "🎚 Quality Analysis"
    )

    st.write(
        "🎼 61 Acoustic Features"
    )

    st.write(
        "🤖 V2 Genre Classification"
    )

    st.write(
        "📊 Confidence Analysis"
    )

    st.write(
        "🔍 Feature-Space Novelty Detection"
    )

    st.write(
        "🗺️ Genre Similarity Map"
    )

    st.write(
        "🧹 Noise Analysis"
    )

    st.write(
        "🧬 Audio DNA"
    )

    st.write(
        "🧠 Explainable AI"
    )

    st.write(
        "🎧 Smart Playlist"
    )

    st.divider()

    st.success(
        "🤖 V2 Model Ready"
    )

    st.success(
        "🗺️ Similarity Map Ready"
    )


# =========================================================
# CHECK FILES
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
# UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "🎧 Upload your audio file",
    type=[
        "wav",
        "mp3",
        "ogg",
        "flac"
    ]
)


if uploaded_file is not None:

    st.success(
        "✅ Audio uploaded successfully!"
    )

    st.audio(
        uploaded_file
    )


    # =====================================================
    # FILE INFORMATION
    # =====================================================

    st.markdown(
        '<div class="section-title">'
        '📁 File Information'
        '</div>',
        unsafe_allow_html=True
    )

    file_size_kb = (
        uploaded_file.size /
        1024
    )

    c1, c2 = st.columns(2)

    with c1:

        st.write(
            f"**File:** {uploaded_file.name}"
        )

    with c2:

        st.write(
            f"**Size:** {file_size_kb:.2f} KB"
        )


    # =====================================================
    # TEMPORARY FILE
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
        # LOAD
        # =================================================

        y, sr = librosa.load(
            temp_audio_path,
            sr=None,
            mono=True
        )


        # =================================================
        # QUALITY
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


        st.markdown(
            '<div class="section-title">'
            '🎚 Audio Quality Analysis'
            '</div>',
            unsafe_allow_html=True
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
                "🟢 Good Audio — "
                "Audio appears suitable for analysis."
            )

        elif quality_score >= 40:

            st.warning(
                "🟡 Audio Needs Attention."
            )

        else:

            st.error(
                "🔴 Low Audio Quality."
            )


        # =================================================
        # FEATURES
        # =================================================

        st.markdown(
            '<div class="section-title">'
            '🎼 Acoustic Feature Extraction'
            '</div>',
            unsafe_allow_html=True
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
        # PREDICTION
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
        # GENRE
        # =================================================

        st.markdown(
            '<div class="section-title">'
            '🤖 AI Genre Prediction'
            '</div>',
            unsafe_allow_html=True
        )


        left, right = st.columns(
            [1, 2]
        )


        with left:

            st.markdown(
                '<div class="genre-card">',
                unsafe_allow_html=True
            )

            st.write(
                "🎵 Primary Genre"
            )

            st.markdown(
                f'<div class="genre-name">'
                f'{primary_genre.upper()}'
                f'</div>',
                unsafe_allow_html=True
            )

            st.write(
                f"{primary_confidence:.2f}% confidence"
            )

            st.markdown(
                '</div>',
                unsafe_allow_html=True
            )


        with right:

            st.subheader(
                "📊 Genre Probability Map"
            )

            for genre, probability in sorted_genres:

                st.write(
                    f"**{genre.title()}** — "
                    f"{probability * 100:.2f}%"
                )

                st.progress(
                    float(
                        probability
                    )
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
                "The song may contain characteristics "
                "of multiple genres."
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

        st.markdown(
            '<div class="section-title">'
            '🔍 Feature-Space Novelty Detection'
            '</div>',
            unsafe_allow_html=True
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
            "Novelty uses nearest-neighbour distance "
            "in the standardized 61-feature training space."
        )


        # =================================================
        # GENRE SIMILARITY MAP
        # =================================================

        st.markdown(
            '<div class="section-title">'
            '🗺️ Genre Similarity Map'
            '</div>',
            unsafe_allow_html=True
        )


        st.write(
            "This projection shows where the uploaded "
            "song sits relative to the acoustic feature "
            "distribution of the ten training genres."
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
            "🗺️ The star represents your song. "
            "Nearby genre clusters indicate greater "
            "acoustic similarity in the 61-dimensional "
            "feature space."
        )


        # =================================================
        # CROSS GENRE
        # =================================================

        st.subheader(
            "🔀 Cross-Genre Analysis"
        )


        second_genre = sorted_genres[1][0]


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

        st.subheader(
            "🎼 Top Genre Characteristics"
        )


        tc1, tc2, tc3 = st.columns(3)


        for index, (
            genre,
            probability
        ) in enumerate(
            sorted_genres[:3]
        ):

            with [
                tc1,
                tc2,
                tc3
            ][index]:

                st.metric(
                    f"#{index + 1}",
                    genre.title(),
                    f"{probability * 100:.2f}%"
                )


        # =================================================
        # NOISE
        # =================================================

        st.markdown(
            '<div class="section-title">'
            '🧹 Noise Detection'
            '</div>',
            unsafe_allow_html=True
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


        nn1, nn2, nn3 = st.columns(3)


        with nn1:

            st.metric(
                "🧹 Noise Level",
                noise_level
            )


        with nn2:

            st.metric(
                "📊 Noise Indicator",
                f"{noise_indicator:.4f}"
            )


        with nn3:

            st.metric(
                "⚠️ Noise Score",
                f"{noise_score:.1f}/100"
            )


        st.info(
            noise_message
        )


        # =================================================
        # CLEANING
        # =================================================

        st.subheader(
            "🧹 Adaptive Noise Reduction"
        )


        clean_button = st.button(
            "🧹 Clean & Continue",
            use_container_width=True
        )


        if clean_button:

            with st.spinner(
                "🧹 Processing audio..."
            ):

                if reduction_strength == 0:

                    cleaned_audio = y.copy()

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


            cleaned_temp = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".wav"
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


            cleaned_scaled_features = scaler.transform(
                cleaned_features.reshape(
                    1,
                    -1
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


            st.divider()


            st.header(
                "🤖 Cleaned Audio Prediction"
            )


            cc1, cc2 = st.columns(2)


            with cc1:

                st.metric(
                    "🎵 Genre",
                    cleaned_primary_genre.title()
                )


            with cc2:

                st.metric(
                    "🎯 Confidence",
                    f"{cleaned_confidence:.2f}%"
                )


            st.subheader(
                "🔄 Original vs Cleaned"
            )


            oc1, oc2 = st.columns(2)


            with oc1:

                st.metric(
                    "Original",
                    primary_genre.title(),
                    f"{primary_confidence:.2f}%"
                )


            with oc2:

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

        st.markdown(
            '<div class="section-title">'
            '🧬 Audio DNA'
            '</div>',
            unsafe_allow_html=True
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


        dna1, dna2, dna3, dna4 = st.columns(4)


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


        aa1, aa2 = st.columns(2)


        with aa1:

            st.write(
                f"**Spectral Centroid:** "
                f"{centroid_value:.2f} Hz"
            )


            st.write(
                f"**Spectral Bandwidth:** "
                f"{bandwidth_value:.2f} Hz"
            )


        with aa2:

            st.write(
                f"**RMS Energy:** "
                f"{energy_value:.4f}"
            )


            st.write(
                f"**Zero Crossing Rate:** "
                f"{zcr_value:.4f}"
            )


        # =================================================
        # INTERPRETATION
        # =================================================

        st.subheader(
            "🎼 Audio DNA Interpretation"
        )


        if tempo_value < 80:

            tempo_label = "Low / Slow"

        elif tempo_value < 130:

            tempo_label = "Medium"

        else:

            tempo_label = "High / Fast"


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

        st.markdown(
            '<div class="section-title">'
            '🧠 Explainable AI'
            '</div>',
            unsafe_allow_html=True
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

        st.markdown(
            '<div class="section-title">'
            '🎧 Smart Playlist Suggestions'
            '</div>',
            unsafe_allow_html=True
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


        for i, playlist in enumerate(
            [
                playlist_1,
                playlist_2,
                playlist_3,
                playlist_4
            ],
            start=1
        ):

            st.write(
                f"**{i}. {playlist}**"
            )


        st.success(
            f"🎧 Recommended profile: "
            f"{primary_genre.title()} · "
            f"{tempo_label} Tempo · "
            f"{energy_label} Energy · "
            f"{brightness_label} Brightness"
        )


        # =================================================
        # FINAL SUMMARY
        # =================================================

        st.markdown(
            '<div class="section-title">'
            '📋 AI Music Intelligence Summary'
            '</div>',
            unsafe_allow_html=True
        )


        s1, s2, s3, s4 = st.columns(4)


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
            f"🧠 **Reliability:** "
            f"{reliability_label}"
        )


        st.success(
            "✅ Full audio intelligence analysis completed."
        )


        st.markdown(
            '<div class="footer">'
            'AI Music Intelligence System • '
            'V2 Acoustic Genre Intelligence'
            '</div>',
            unsafe_allow_html=True
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