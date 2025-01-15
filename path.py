from audiocraft.models import MusicGen
import streamlit as st
import torch
import torchaudio
import os
import numpy as np
import base64
import random
import audiocraft

def load_model():
    model = MusicGen.get_pretrained('facebook/musicgen-melody')
    
    # Attempt to retrieve the model directory if it exists
    try:
        # `MusicGen` might not expose `model_dir`, so this is a generic fallback
        model_path = model.__dict__.get('_path', 'Unknown location')
        print(f"Model is loaded from: {model_path}")
        
        # Print where the model weights are likely stored
        checkpoint_path = os.path.join(model_path, "checkpoints") if model_path != 'Unknown location' else "Checkpoints directory unknown"
        print(f"Checkpoints are located at: {checkpoint_path}")
    except AttributeError:
        print("Unable to retrieve model directory or checkpoints path. Model might handle this differently.")
    
    return model


load_model()