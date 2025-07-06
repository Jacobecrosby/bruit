import librosa
import numpy as np
import soundfile as sf
from pathlib import Path
import os
import logging
from tqdm import tqdm

logger = logging.getLogger("bruit")

def pad_or_truncate(feature, target_shape):
    padded = np.zeros(target_shape, dtype=feature.dtype)
    f_dim, t_dim = feature.shape
    f_target, t_target = target_shape

    f = min(f_dim, f_target)
    t = min(t_dim, t_target)

    padded[:f, :t] = feature[:f, :t]
    return padded

def get_max_shape(features, axis=1):
    return max(f.shape[axis] for f in features if isinstance(f, np.ndarray))
    
def extract_features(y, sr, segment_id, label):
    n_fft = min(2048, len(y))
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=n_fft)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64, n_fft=n_fft)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    rms = librosa.feature.rms(y=y)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]

    return {
        "segment_id": segment_id,
        "label": label,
        "mfcc": mfcc,
        "mel": mel_db,
        "rms": rms,
        "zcr": zcr
    }

def run_feature_extraction(segment_dirs, save_path, config, sr=16000, quiet=False):
    all_features = []

    for segment_root in tqdm(segment_dirs, desc="Extracting Features", disable=quiet):
        segment_root = Path(segment_root)

        segment_files = sorted(segment_root.glob("*.wav"))
        label = segment_root.parents[1].name.lower() if len(segment_root.parents) >= 2 else "unknown"

        for segment_path in tqdm(segment_files, desc=f"Extracting [{label}]", leave=False, disable=quiet):
            y, _ = sf.read(segment_path)
            y = librosa.to_mono(y) if y.ndim > 1 else y

            # Get label from two levels above
            try:
                label = segment_path.parents[2].name.lower()
            except IndexError:
                label = "unknown"

            segment_id = segment_path.stem
            feats = extract_features(y, sr, segment_id, label)
            all_features.append(feats)

            if not quiet:
                logger.info(f"[{label}] Extracted: {segment_id}")
    
    # Collect all features
    mfcc_raw = [f["mfcc"] for f in all_features]
    mel_raw  = [f["mel"] for f in all_features]
    rms_raw  = [f["rms"] for f in all_features]
    zcr_raw  = [f["zcr"] for f in all_features]

    # Auto-infer time dimensions. Get max shape for each feature type
    mfcc_t = get_max_shape(mfcc_raw, axis=1)
    mel_t  = get_max_shape(mel_raw, axis=1)
    rms_t  = get_max_shape(rms_raw, axis=0)
    zcr_t  = get_max_shape(zcr_raw, axis=0)

    # Define padding targets
    mfcc_target = (13, mfcc_t)
    mel_target  = (64, mel_t)
    rms_target  = (rms_t,)
    zcr_target  = (zcr_t,)

    padded_mfcc = [pad_or_truncate(f, mfcc_target) for f in mfcc_raw]
    padded_mel  = [pad_or_truncate(f, mel_target) for f in mel_raw]
    padded_rms   = [pad_or_truncate(f.reshape(1, -1), (1, rms_t))[0] for f in rms_raw]
    padded_zcr   = [pad_or_truncate(f.reshape(1, -1), (1, zcr_t))[0] for f in zcr_raw]


    np.savez_compressed(
        save_path,
        segment_ids=np.array([f["segment_id"] for f in all_features]),
        labels=np.array([f["label"] for f in all_features]),
        mfcc=np.stack(padded_mfcc),  
        mel=np.stack(padded_mel),    
        rms=np.stack(padded_rms),     
        zcr=np.stack(padded_zcr),     
    )

    if not quiet:
        print(f"Saved all extracted features to {save_path}")
        logger.info(f"Saved all extracted features to {save_path}")