import librosa
import numpy as np
import soundfile as sf
from pathlib import Path
import os
import logging
from tqdm import tqdm

logger = logging.getLogger("bruit")

def extract_features(y, sr, segment_id, label):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
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

def run_feature_extraction(segment_dirs, save_path, sr=16000, quiet=False):
    all_features = []

    for segment_root in tqdm(segment_dirs, desc="Extracting Features", disable=quiet):
        segment_root = Path(segment_root)

        segment_files = sorted(segment_root.glob("*.wav"))
        label = segment_root.parents[1].name.lower() if len(segment_root.parents) >= 2 else "unknown"
        print(f"Processing segments for label: {label}")

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
    
    for i, f in enumerate(all_features):
        mfcc = f["mfcc"]
        if not isinstance(mfcc, np.ndarray):
            logger.warning(f"⚠️ MFCC {i} is not a NumPy array: type={type(mfcc)}")
        elif mfcc.ndim != 2:
            logger.warning(f"⚠️ MFCC {i} is not 2D: shape={mfcc.shape}")

    try:
        mfccs = np.asarray([f["mfcc"] for f in all_features], dtype=object)
    except Exception as e:
        logger.error(f"🔥 Error while creating MFCC array: {e}")
        for i, f in enumerate(all_features):
            logger.warning(f"MFCC shape {i}: {f['mfcc'].shape}")
        raise e
    
    # Convert to structured arrays
    segment_ids = np.array([f["segment_id"] for f in all_features])
    labels = np.asarray([f["label"] for f in all_features])
    mfccs = np.asarray([f["mfcc"] for f in all_features], dtype=object)
    mels = np.asarray([f["mel"] for f in all_features], dtype=object)
    rms = np.asarray([f["rms"] for f in all_features], dtype=object)
    zcr = np.asarray([f["zcr"] for f in all_features], dtype=object)

    np.savez_compressed(
        save_path,
        segment_ids=segment_ids,
        labels=labels,
        mfcc=mfccs,
        mel=mels,
        rms=rms,
        zcr=zcr
    )

    if not quiet:
        print(f"\n✅ Saved features to {save_path}")
        logger.info(f"\n✅ Saved all extracted features to {save_path}")