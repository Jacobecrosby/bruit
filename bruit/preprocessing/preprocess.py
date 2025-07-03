# let's resample!
import librosa 
import numpy as np
import os
import logging
import yaml
from bruit.modules.config_loader import load_config, resolve_path
from pathlib import Path



def resample_audio(input_path, output_path, config=None):

    resample_rate = config.get("preprocessing", {}).get("resample_rate", 16000)




def run_preprocessing(input_dir, output_path, config=None, quiet=False):
    path_config = load_config("configs/paths.yaml")
    input_path = Path(input_dir)

    # Load the original YAML
    with open("configs/parameters.yaml", "r") as infile:
        full_config = yaml.safe_load(infile)

    resample_audio(input_dir, output_path, config=config, quiet=quiet)