import os

import numpy as np
import matplotlib
from matplotlib import font_manager

# Avoid the missing LastResort font in some Python distributions.
_dejavu_font = os.path.join(
    matplotlib.get_data_path(), "fonts", "ttf", "DejaVuSans.ttf"
)
if os.path.isfile(_dejavu_font):
    font_manager.fontManager.addfont(_dejavu_font)
    matplotlib.rcParams["font.family"] = [
        font_manager.FontProperties(fname=_dejavu_font).get_name()
    ]
matplotlib.rcParams["font.enable_last_resort"] = False

import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view


class IMUDataProcessor:
    """Preprocess and analyze multichannel IMU measurements."""

    def __init__(self, raw_data):
        data = np.asarray(raw_data, dtype=float)
        if data.ndim != 2:
            raise ValueError("raw_data must be a two-dimensional array")
        if data.shape[1] < 3:
            raise ValueError("raw_data must contain at least three channels")
        self.raw_data = data
        self.data = data.copy()

    def preprocess(self):
        """Remove each channel's mean and scale it to unit variance."""
        mean = self.data.mean(axis=0)
        standard_deviation = self.data.std(axis=0)
        if np.any(standard_deviation == 0):
            raise ValueError("cannot normalize a channel with zero variance")
        self.data = (self.data - mean) / standard_deviation
        return self.data

    def detect_anomalies(self, threshold=3.0):
        """Return sample indices whose acceleration magnitude exceeds threshold."""
        if threshold < 0:
            raise ValueError("threshold must be non-negative")
        acceleration = self.data[:, :3]
        magnitude = np.linalg.norm(acceleration, axis=1)
        return np.flatnonzero(magnitude > threshold)

    def analyze_windows(self, window_size=100):
        """Return per-channel means and standard deviations for sliding windows."""
        if not isinstance(window_size, (int, np.integer)) or window_size <= 0:
            raise ValueError("window_size must be a positive integer")
        if window_size > len(self.data):
            raise ValueError("window_size cannot exceed the number of samples")

        windows = sliding_window_view(self.data, window_shape=window_size, axis=0)
        return windows.mean(axis=2), windows.std(axis=2)


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    time = np.arange(1000) / 100.0
    true_signal = 2 * np.sin(2 * np.pi * 2 * time)
    noise = rng.normal(0, 0.3, 1000)

    acceleration_x = true_signal + noise
    acceleration_y = true_signal + rng.normal(0, 0.3, 1000)
    acceleration_z = true_signal + rng.normal(0, 0.3, 1000)
    gyroscope_z = 0.5 + rng.normal(0, 0.1, 1000)
    data = np.column_stack(
        [acceleration_x, acceleration_y, acceleration_z, gyroscope_z]
    )

    processor = IMUDataProcessor(data)
    anomaly_indices = processor.detect_anomalies()
    acceleration_magnitude = np.linalg.norm(processor.data[:, :3], axis=1)
    processor.preprocess()
    window_means, window_standard_deviations = processor.analyze_windows()

    print(f"Processed data shape: {processor.data.shape}")
    print(f"Number of anomalies: {len(anomaly_indices)}")
    print(f"First anomaly indices: {anomaly_indices[:5]}")
    print(f"Window means shape: {window_means.shape}")
    print(f"Window standard deviations shape: {window_standard_deviations.shape}")

    plt.figure(figsize=(12, 5))
    plt.plot(time, acceleration_magnitude, label="Acceleration magnitude")
    plt.scatter(
        time[anomaly_indices],
        acceleration_magnitude[anomaly_indices],
        color="red",
        label="Anomalies",
        zorder=3,
    )
    plt.title("IMU Acceleration Magnitude & Anomaly Detection")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Magnitude")
    plt.legend()
    plt.tight_layout()
    plt.show()
