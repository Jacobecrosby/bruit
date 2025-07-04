Read me please!

.wav --> resample --> trim --> normalize --> pad/crop --> extract features (MFCC or Mel) --> classifier


A healthy heart rate is typically at a recording frequency range of 20 Hz - 200 Hz
Heart murmurs typically have a recording frequency of 130 Hz - 410 Hz 

1. Use median absolute deviation (MAD) to filter peak outliers
Better than simple percentile thresholds

Identify and ignore extreme-amplitude peaks before scoring

python
Copy
Edit
# After peak detection
peak_heights = clipped_y[peaks]
median = np.median(peak_heights)
mad = np.median(np.abs(peak_heights - median))
valid_mask = np.abs(peak_heights - median) < 3 * mad
valid_peaks = peaks[valid_mask]

2. New Scoring Function
We combine:

Peak count (want at least N peaks)

Spacing consistency (low variance in np.diff)

Peak regularity (MAD of peak amplitudes)

python
Copy
Edit
spacing = np.diff(valid_peaks)
spacing_var = np.var(spacing)
amplitude_mad = np.median(np.abs(peak_heights[valid_mask] - np.median(peak_heights[valid_mask])))

# You want many peaks, consistent spacing, and stable amplitude
score = len(valid_peaks) / (1 + spacing_var + amplitude_mad)

# 3. Reject filters with too few valid peaks (<3–5)
python
Copy
Edit
if len(valid_peaks) < 3:
    continue\
4. Optional: Visual debug plots for scoring curves
To help tune it later, log or plot scores per lowcut.