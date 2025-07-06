import librosa
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.cm as cm
import logging
import json

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


def plot_segment_counts(file_names, segment_counts, stages, label, save_path=None, title="Segments per File"):
    assert len(file_names) == len(segment_counts) == len(stages), "Mismatched input lengths"

    title = f"{title} - {label}"
    total_segments = sum(segment_counts)

    # Generate distinct colors for each stage using colormap
    unique_stages = sorted(set(stages))
    cmap = cm.get_cmap('tab10', len(unique_stages))  # or 'viridis', 'Set1'
    stage_to_color = {stage: cmap(i) for i, stage in enumerate(unique_stages)}
    bar_colors = [stage_to_color[s] for s in stages]

    # Plot
    plt.figure(figsize=(12, 4))
    bars = plt.bar(range(len(file_names)), segment_counts, color=bar_colors)

    plt.xticks(range(len(file_names)), file_names, rotation=90, fontsize=8)
    plt.ylabel("Segments Found")
    plt.title(title)
    plt.legend(
        handles=[plt.Rectangle((0, 0), 1, 1, color=stage_to_color[s]) for s in unique_stages],
        labels=[f"Stage {s}" for s in unique_stages],
        title=f"Total segments: {total_segments}",
        loc='upper right'
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        logger.info(f"Segment count plot saved to {save_path}")
    else:
        plt.show()

    plt.close()

def plot_training_history(history, save_path=None, title="Training History"):
    plt.figure(figsize=(10, 4))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss")
    plt.grid(True)
    plt.legend()

    # Plot Accuracy (if available)
    if "accuracy" in history.history:
        plt.subplot(1, 2, 2)
        plt.plot(history.history["accuracy"], label="Train Acc")
        plt.plot(history.history["val_accuracy"], label="Val Acc")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Accuracy")
        plt.grid(True)
        plt.legend()

    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        logger.info(f"Saved training plot to {save_path}")
    else:
        plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None, normalize=False):
    cm = confusion_matrix(y_true, y_pred, normalize='true' if normalize else None)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Normalized Confusion Matrix" if normalize else "Confusion Matrix")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        logger.info(f"Saved confusion matrix to {save_path}")
    else:
        plt.show()
    plt.close()

def report_metrics(model, X_val, y_val, class_names, save_confusion_path=None,report_path=None):
    y_pred_probs = model.predict(X_val)
    y_pred = np.argmax(y_pred_probs, axis=1)

    #Generate the classification report as a dictionary
    report_dict = classification_report(y_val, y_pred, target_names=class_names, output_dict=True)

    # Print to console
    print("Classification Report:")
    print(classification_report(y_val, y_pred, target_names=class_names))

    # Save report to JSON
    if report_path:
        with open(report_path, "w") as f:
            json.dump(report_dict, f, indent=4)
        logger.info(f"Saved classification report to {report_path}")


    plot_confusion_matrix(y_val, y_pred, class_names, save_path=save_confusion_path, normalize=True)