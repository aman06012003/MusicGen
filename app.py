from audiocraft.models import MusicGen
import streamlit as st
import torch
import torchaudio
import os
import numpy as np
import base64
import random
import audiocraft

@st.cache_resource
def load_model():
    model = MusicGen.get_pretrained('facebook/musicgen-melody')
    return model

def create_placeholder_audio(sample_rate=32000, duration=5):
    """Creates a silent waveform as a placeholder audio."""
    num_samples = int(sample_rate * duration)
    placeholder_waveform = torch.zeros(1, num_samples)
    return placeholder_waveform, sample_rate

def generate_music_tensors(description, duration: int):
    em_map = {
        'happy': ["Ashtami title rough track_instrumental.wav", "Group Song Band Vadya FINAL--2_instrumental.wav", "Maimeda Baalelu Title Song Master---1_instrumental.wav"],
        'sad': ["Perumala Ballal Master_instrumental.wav", "Sanvi song To listen--1_instruemntal.wav"],
        'romantic': ["Love song asthami for Lyrics_instrumental.wav", "Sanvi song To listen--1_instrumental.wav"],
        'angry': "Godduba Chend Master_instrumental.wav"
    }

    description = description.lower()
    melody_waveform, sr = None, None

    if "happy" in description:
        file_path = random.choice(em_map["happy"])
    elif "sad" in description:
        file_path = random.choice(em_map["sad"])
    elif "romantic" in description:
        file_path = random.choice(em_map["romantic"])
    elif "angry" in description:
        file_path = em_map["angry"]
    else:
        file_path = random.choice(em_map["happy"])

    if os.path.exists(file_path):
        melody_waveform, sr = torchaudio.load(file_path)
        melody_waveform = melody_waveform.unsqueeze(0).repeat(1, 1, 1)
    else:
        st.warning(f"File {file_path} does not exist. Using placeholder audio.")
        melody_waveform, sr = create_placeholder_audio(duration=5)

    model = load_model()
    
    model.set_generation_params(
        use_sampling=True,
        top_k=250,
        duration=duration
    )

    output = model.generate_with_chroma(
        descriptions=[str(description) + f" similar to {melody_waveform}"],
        progress=True,
        return_tokens=True,
        melody_wavs=melody_waveform,
        melody_sample_rate=sr
    )

    return output[0]

def save_audio(samples: torch.Tensor):
    """Saves audio samples to a file."""
    sample_rate = 32000
    assert samples.dim() == 2 or samples.dim() == 3

    samples = samples.detach().cpu()
    if samples.dim() == 2:
        samples = samples[None, ...]

    for idx, audio in enumerate(samples):
        audio_path = f"audio_{idx}.wav"
        torchaudio.save(audio_path, audio, sample_rate)

def get_binary_file_downloader_html(bin_file, file_label='File'):
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Download {file_label}</a>'
    return href

st.set_page_config(
    page_icon="musical_note",
    page_title="Music Gen"
)

def main():
    st.title("Text to Music Generator🎵")

    with st.expander("See explanation"):
        st.write("Music Generator app built using Meta's Audiocraft library. We are using Music Gen Melody model.")

    text_area = st.text_area("Enter your description.......")
    time_slider = st.slider("Select time duration (In Seconds)", 0, 20, 10)

    if text_area and time_slider:
        st.subheader("Generated Music")
        music_tensors = generate_music_tensors(text_area, time_slider)
        save_audio(music_tensors)
        audio_filepath = f'audio_0.wav'
        audio_file = open(audio_filepath, 'rb')
        audio_bytes = audio_file.read()
        st.audio(audio_bytes)
        st.markdown(get_binary_file_downloader_html(audio_filepath, 'Audio'), unsafe_allow_html=True)

if __name__ == "__main__":
    main()
