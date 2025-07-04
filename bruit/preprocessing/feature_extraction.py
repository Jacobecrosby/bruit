import librosa
import numpy as np
import soundfile as sf
from pathlib import Path
import os

def save_feature_npz(features, save_path):
    np.savez_compressed(save_path, **features)

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

def run_feature_extraction(segment_dir, label_map, save_path, sr=16000, quiet=False):
    segment_dir = Path(segment_dir)
    all_features = []

    for segment_path in sorted(segment_dir.glob("*.wav")):
        y, _ = sf.read(segment_path)
        y = librosa.to_mono(y) if y.ndim > 1 else y

        # Derive label from parent directory or use filename logic
        parent = segment_path.parent.name.lower()
        label = label_map.get(parent, "unknown")

        segment_id = segment_path.stem
        feats = extract_features(y, sr, segment_id, label)
        all_features.append(feats)

        if not quiet:
            print(f"Extracted features for {segment_id}")

    # Convert to structured arrays
    segment_ids = np.array([f["segment_id"] for f in all_features])
    labels = np.array([f["label"] for f in all_features])
    mfccs = np.array([f["mfcc"] for f in all_features], dtype=object)
    mels = np.array([f["mel"] for f in all_features], dtype=object)
    rms = np.array([f["rms"] for f in all_features], dtype=object)
    zcr = np.array([f["zcr"] for f in all_features], dtype=object)

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
        print(f"\n✅ Saved all features to: {save_path}")