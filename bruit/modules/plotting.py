import librosa
import numpy as np
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger("bruit")

def plot_waveform(y, sr, save_path=None, title='Preprocessed Audio'):

    t = np.linspace(0, len(y) / sr, num=len(y))
    plt.figure(figsize=(10, 3))
    plt.plot(t, y, linewidth=0.7)
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        logger.info(f"Waveform saved to {save_path}")
    else:
        plt.show()
    
    plt.close()

def plot_spectrogram(y, sr, save_path=None, title="Spectrogram"):
    """
    Plots a dB-scaled spectrogram of the input audio signal.

    Parameters:
        y (np.ndarray): Audio signal (1D).
        sr (int): Sample rate.
        title (str): Plot title.
        save_path (str or Path, optional): If given, saves the plot instead of displaying it.
    """
    # Compute short-time Fourier transform
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    # Plot
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar(format="%+2.0f dB")
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        logger.info(f"Spectrogram saved to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_segment_counts(file_names, segment_counts, save_path=None, title="Heartbeat Segments per File"):
    """
    Plots the number of segments found per audio file.

    Parameters:
        file_names (list of str): Names of the audio files.
        segment_counts (list of int): Number of segments found in each file.
        save_path (Path or str, optional): If provided, saves the plot instead of showing.
        title (str): Title of the plot.
    """
    total_segments = sum(segment_counts)

    plt.figure(figsize=(12, 4))
    plt.bar(range(len(file_names)), segment_counts, color='skyblue',label=f"Total segments: {total_segments}")
    plt.xticks(range(len(file_names)), file_names, rotation=90, fontsize=8)
    plt.ylabel("Segments Found")
    plt.title(title)
    plt.legend(loc='upper right')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        logger.info(f"Segment count plot saved to {save_path}")
    else:
        plt.show()

    plt.close()