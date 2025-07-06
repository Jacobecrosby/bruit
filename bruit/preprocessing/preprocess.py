import librosa 
import soundfile as sf
from scipy.signal import butter, lfilter, filtfilt, find_peaks
import numpy as np
from tqdm import tqdm
import os
import logging
import yaml
from bruit.modules.config_loader import load_config, resolve_path
from bruit.modules.plotting import plot_waveform, plot_spectrogram, plot_segment_counts
from pathlib import Path
import matplotlib.pyplot as plt

logger = logging.getLogger("bruit")


def adaptive_lowcut_filter(y, sr, highcut=1200, order=4, lowcut_range=(20, 250), step=10, peak_prominence=0.05, quiet=False, plot_debug=False, debug_save_path=None):
    best_score = float('-inf')
    best_filtered = None
    best_lowcut = None
    score_log = []

    lowcut_values = range(lowcut_range[0], lowcut_range[1] + 1, step)

    for lowcut in lowcut_values:
        # Bandpass filter
        b, a = butter(order, [lowcut / (0.5 * sr), highcut / (0.5 * sr)], btype='band')
        y_filt = filtfilt(b, a, y)


        # Peak detection
        abs_y = np.abs(y_filt)
        peaks, _ = find_peaks(abs_y, prominence=peak_prominence)
        if len(peaks) < 3:
            score_log.append((lowcut, 0))
            continue

        # Remove outlier peaks using MAD
        peak_heights = abs_y[peaks]
        median = np.median(peak_heights)
        mad = np.median(np.abs(peak_heights - median))
        if mad == 0:
            score_log.append((lowcut, 0))
            continue
        valid_mask = np.abs(peak_heights - median) < 3 * mad
        valid_peaks = peaks[valid_mask]

        if len(valid_peaks) < 3:
            score_log.append((lowcut, 0))
            continue

        # Scoring
        spacing = np.diff(valid_peaks)
        spacing_var = np.var(spacing)
        amp_mad = np.median(np.abs(peak_heights[valid_mask] - np.median(peak_heights[valid_mask])))

        score = len(valid_peaks) / (1 + spacing_var + amp_mad)
        score_log.append((lowcut, score))

        if score > best_score:
            best_score = score
            best_filtered = y_filt
            best_lowcut = lowcut

    if best_filtered is None:
        if not quiet:
            logger.info("No valid filtering result. Returning original signal.")
        return y, None

    # Optional debug plot
    if plot_debug:
        lowcuts, scores = zip(*score_log)
        plt.figure(figsize=(8, 3))
        plt.plot(lowcuts, scores, marker='o')
        plt.title("Adaptive Lowcut Scoring")
        plt.xlabel("Lowcut Frequency (Hz)")
        plt.ylabel("Score")
        plt.grid(True)
        plt.tight_layout()
        if debug_save_path:
            plt.savefig(debug_save_path, dpi=300)
            logger.info(f"Debug score plot saved to {debug_save_path}")
        else:
            plt.show()
        plt.close()

    return best_filtered, best_lowcut


def bandpass_filter(data, sr, config):
    filter_cfg = config.get("preprocessing", {}).get("filter", {})

    order = filter_cfg.get("order", 5)
    lowcut = filter_cfg.get("lowcut", 20)
    highcut = filter_cfg.get("highcut", 600)

    b, a = butter(order, [lowcut / (sr/2), highcut / (sr/2)], btype='band')
    
    return lfilter(b, a, data)

def preprocess_audio(input_file, paths, config, sample_rate, quiet=False):

    mono = config.get("preprocessing", {}).get("mono", True)
    normalize = config.get("preprocessing", {}).get("normalize", True)    
    
    if not quiet:
        logger.info(f"Resampling: {input_file} → {paths['output']} @ {sample_rate} Hz")
    
    y, _ = librosa.load(input_file, sr=sample_rate, mono=mono)

    if config.get("plotting", {}).get("save_spectogram_plots", False):
            plot_spectrogram(y, sample_rate, save_path=paths['spectrograms']['raw'] / f"{input_file.stem}_spectrogram.png")

    # Check if the audio is empty
    if y.size == 0:
        logger.warning(f"Empty audio file: {input_file}. Skipping.")
        return None
    # apply bandpass filter if configured
    if config.get("preprocessing", {}).get("filter", {}).get("enabled", False):
        # Applies an adaptive lowcut filter if configured, otherwise applies a bandpass filter
        if config.get("preprocessing", {}).get("filter", {}).get("adaptive_lowcut",{}).get("enabled", False):
            if not quiet:
                logger.info(f"Applying adaptive lowcut filter to {input_file.name}")
            y, lowcut = adaptive_lowcut_filter(y, sample_rate, 
                                               highcut=config.get("preprocessing", {}).get("filter", {}).get("adaptive_lowcut",{}).get("highcut", 1200),
                                               order=config.get("preprocessing", {}).get("filter", {}).get("adaptive_lowcut",{}).get("order", 5),
                                               lowcut_range=config.get("preprocessing", {}).get("filter", {}).get("adaptive_lowcut",{}).get("adaptive_lowcut_range", (20, 250)),
                                               step= config.get("preprocessing", {}).get("filter", {}).get("adaptive_lowcut",{}).get("steps", 10),
                                               peak_prominence= config.get("preprocessing", {}).get("filter", {}).get("adaptive_lowcut",{}).get("peak_prominence", 0.05),
                                               quiet=quiet,
                                               plot_debug=config.get("preprocessing", {}).get("filter", {}).get("adaptive_lowcut",{}).get("plot_debug", False),
                                               debug_save_path=paths['debug'] / f"{input_file.stem}_adaptive_lowcut_debug.png")
            logger.info(f"Adaptive lowcut filter applied with lowcut frequency: {lowcut} Hz")
        else:
            if not quiet:
                logger.info(f"Applying bandpass filter to {input_file.name}")
            y = bandpass_filter(y, sample_rate, config)

    if config.get("plotting", {}).get("save_spectogram_plots", False):
            plot_spectrogram(y, sample_rate, save_path=paths['spectrograms']['filtered'] / f"{input_file.stem}_spectrogram.png")
    
    # Normalize the audio
    if normalize:
        y = y / np.max(np.abs(y))

    #sf.write(output_file, y, sample_rate)

    return y 

def split_audio_by_heartbeats(y, sr, config,  retry_aggressive_band=True, band_lowcut=100, band_highcut=1000,band_peak_prominence=0.02, retry_with_rms=True, quiet=False):
    """
    Splits audio into chunks containing one S1 + one S2 beat using peak intervals.

    Parameters:
        y (np.ndarray): Filtered audio signal
        sr (int): Sample rate
        peak_prominence (float): Minimum prominence for peak detection
        min_spacing (float): Minimum S1–S2 spacing in seconds
        max_spacing (float): Maximum S1–S2 spacing in seconds
        context (float): Padding before S1 and after S2 in seconds

    Returns:
        List of audio segments (np.ndarray) containing one heartbeat each
    """
    min_segments = config.get("preprocessing", {}).get("splitting", {}).get("min_segments", 4)
    peak_prominence = config.get("preprocessing", {}).get("splitting",{}).get("peak_prominence", 0.05)
    min_spacing = config.get("preprocessing", {}).get("splitting",{}).get("min_spacing", 0.3)
    max_spacing = config.get("preprocessing", {}).get("splitting",{}).get("max_spacing", 0.6)
    context = config.get("preprocessing", {}).get("splitting",{}).get("context", 0.1)
    mad_threshold = config.get("preprocessing", {}).get("splitting",{}).get("mad_multiplier", 3.0)
    band_lowcut = config.get("preprocessing", {}).get("splitting",{}).get("band_lowcut", 100)
    band_highcut = config.get("preprocessing", {}).get("splitting",{}).get("band_highcut", 1000)
    band_peak_prominence = config.get("preprocessing", {}).get("splitting",{}).get("band_peak_prominence", 0.02)
    rms_window = config.get("preprocessing", {}).get("splitting",{}).get("rms_window", 0.1)
    rms_peak_prominence = config.get("preprocessing", {}).get("splitting",{}).get("rms_peak_prominence", 0.005)
    segment_duration = config.get("preprocessing", {}).get("splitting", {}).get("uniform_duration", 1.2)
    stride = config.get("preprocessing", {}).get("splitting", {}).get("uniform_stride", segment_duration)
    min_rms_threshold = config.get("preprocessing", {}).get("splitting", {}).get("min_rms_energy", 0.005)

    stage = 1

    def extract_segments_from_peaks(peaks, abs_y):
        segments = []
        for i in range(len(peaks) - 1):
            t1, t2 = peaks[i], peaks[i + 1]
            dt = (t2 - t1) / sr
            if min_spacing <= dt <= max_spacing and abs_y[t1] > abs_y[t2]:
                start = max(0, t1 - int(context * sr))
                end = min(len(y), t2 + int(context * sr))
                segments.append(y[start:end])
        return segments
    
    def bandpass(y, sr, lowcut, highcut, order=4):
        nyq = 0.5 * sr
        b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
        return filtfilt(b, a, y)
    
    # Stage 1: Peak detection on single band filtering
    abs_y = np.abs(y)
    peaks, _ = find_peaks(abs_y, prominence=peak_prominence)

    peak_vals = abs_y[peaks]
    if len(peak_vals) > 0:
        median = np.median(peak_vals)
        mad = np.median(np.abs(peak_vals - median))
        valid_mask = np.abs(peak_vals - median) < mad_threshold * mad
        clean_peaks = peaks[valid_mask]
    else:
        clean_peaks = []

    segments = extract_segments_from_peaks(clean_peaks, abs_y)

    # Stage 2: Aggressive bandpass + peak detection
    if retry_aggressive_band and len(segments) <= min_segments:
        if not quiet:
            logger.info("No segments from Stage 1. Retrying with aggressive bandpass...")
        stage = 2

        y_band = bandpass(y, sr, band_lowcut, band_highcut)
        abs_y_band = np.abs(y_band)

        peaks_band, _ = find_peaks(abs_y_band, prominence=band_peak_prominence)
        peak_vals = abs_y_band[peaks_band]
        if len(peak_vals) > 0:
            median = np.median(peak_vals)
            mad = np.median(np.abs(peak_vals - median))
            valid_mask = np.abs(peak_vals - median) < 3 * mad
            clean_peaks = peaks_band[valid_mask]
        else:
            clean_peaks = []   
        
        segments = extract_segments_from_peaks(clean_peaks, abs_y_band)

    # Stage 3: RMS envelope
    if retry_with_rms and len(segments) <= min_segments:
        if not quiet:
            logger.info("No segments from Stage 2. Retrying using RMS envelope...")
        stage = 3

        hop_length = int(rms_window * sr)
        frame_length = 2 * hop_length
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        rms_time = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
        rms_peaks, _ = find_peaks(rms, prominence=rms_peak_prominence)
        rms_peak_times = rms_time[rms_peaks]
        rms_peak_samples = (rms_peak_times * sr).astype(int)

        segments = extract_segments_from_peaks(rms_peak_samples, np.abs(y))

    if len(segments) <= min_segments:
        if not quiet:
            logger.info("No segments from Stage 3. Retrying with uniform fixed-length chopping...")
        stage = 4

        segment_samples = int(segment_duration * sr)
        stride_samples = int(stride * sr)
        segments = []

        for start in range(0, len(y) - segment_samples + 1, stride_samples):
            end = start + segment_samples
            segment = y[start:end]
            rms = np.sqrt(np.mean(segment**2))

            if rms < min_rms_threshold:
                continue

            # Trim to first two prominent peaks (if available)
            abs_seg = np.abs(segment)
            peaks, _ = find_peaks(abs_seg, prominence=peak_prominence)

            if len(peaks) >= 2:
                t1, t2 = peaks[0], peaks[1]
                pad = int(context * sr)
                seg_start = max(0, t1 - pad)
                seg_end = min(len(segment), t2 + pad)
                trimmed_segment = segment[seg_start:seg_end]
                segments.append(trimmed_segment)
                logger.debug(f"Stage 4 trimmed segment length: {len(trimmed_segment)} samples")
            else:
                # Fallback: keep full segment if only one peak
                segments.append(segment)

        if not quiet:
            logger.info(f"Stage 4 extracted {len(segments)} cleaned segments with first peak pair")

    return segments, stage


def run_preprocessing(input_dir, output_path, config=None, quiet=False):

    sample_rate = config.get("preprocessing", {}).get("sample_rate", 16000)
    input_path = Path(input_dir)

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    audio_full_path = output_path / "audio" / "full"
    audio_full_path.mkdir(parents=True, exist_ok=True)

    audio_segment_path = output_path / "audio" / "segments"
    audio_segment_path.mkdir(parents=True, exist_ok=True)

    plot_path = output_path / "plots"
    plot_type_paths = [plot_path / "spectrograms" / "raw", plot_path / "spectrograms" / "filtered", plot_path / "raw", plot_path / "preprocessed" , plot_path / "debug", plot_path / "segments"]
 
    for d in plot_type_paths:
        d.mkdir(parents=True, exist_ok=True)

    paths = {
        "output": output_path,
        "audio_full": audio_full_path,
        "audio_segment": audio_segment_path,
        "spectrograms": {
            "raw": plot_type_paths[0],
            "filtered": plot_type_paths[1]
        },
        "raw": plot_type_paths[2],
        "preprocessed_plots": plot_type_paths[3],
        "debug": plot_type_paths[4],
        "segments": plot_type_paths[5]
    }

    total_segments = []
    total_stages = []
    file_names = []
    for file in tqdm(list(input_path.glob("*.wav")), desc=f"Preprocessing {input_dir}", disable=quiet):
        input_file = file.resolve()
        file_names.append(file.name)
        # Resample the audio
        # Plot raw full waveform
        if config.get("plotting", {}).get("save_preprocessed_plots", False):
            logger.info(f"Plotting raw waveform for {file.name}")
            y, _ = librosa.load(input_file, sr=sample_rate, mono=True)
            plot_waveform(y, sample_rate, paths['raw'] / f"{file.stem}_waveform.png")


        y_filtered = preprocess_audio(input_file, paths, config, sample_rate, quiet=quiet)

        if config.get('preprocessing', {}).get('save_full_filtered_audio', False):
            # Save the full filtered audio
            output_file = paths['audio_full'] / f"{file.stem}_filtered.wav"
            sf.write(output_file, y_filtered, sample_rate)
            logger.info(f"Saved full filtered audio to {output_file}")

        # Plot full waveform
        if config.get("plotting", {}).get("save_preprocessed_plots", False):
            logger.info(f"Plotting waveform for {file.name}")
            plot_waveform(y_filtered, sample_rate, paths['preprocessed_plots'] / f"{file.stem}_waveform.png")

        # Segment into heartbeats
        segments, stages = split_audio_by_heartbeats(y_filtered, sample_rate, config)
        total_segments.append(int(len(segments)))
        total_stages.append(stages)

        if not quiet:
            logger.info(f"Processing {file.name} with {len(segments)} segments")

        # Plot segments
        if config.get("plotting", {}).get("save_preprocessed_plots", False):
            for i, segment in enumerate(segments):
                logger.info(f"Plotting segment {i} waveform for {file.name}")
                plot_waveform(segment, sample_rate, paths['segments'] / f"{file.stem}_{i}_waveform.png")

        if config.get('preprocessing', {}).get('save_segment_waveform', False):
            for i, s in enumerate(segments):
                save_segment_path = paths['audio_segment'] / f"{file.stem}_{i}.wav"
                sf.write(save_segment_path, s, sample_rate)

    if config.get("plotting", {}).get("save_preprocessed_plots", False):
        plot_segment_counts(file_names, total_segments, total_stages,input_path.name, save_path=plot_path / "segment_counts.png")

    logger.info(f"Preprocessing complete. Processed {len(total_segments)} files with a total of {sum(total_segments)} segments.")