# IMU Anomaly Detector

A Python-based tool for preprocessing multichannel Inertial Measurement Unit (IMU) data and detecting physical anomalies (such as shocks or falls) using sliding window analysis.

## Features
- **Object-Oriented Design:** Clean, reusable `IMUDataProcessor` class.
- **Statistical Preprocessing:** Mean removal and unit variance scaling.
- **Anomaly Detection:** Identifies spikes in acceleration magnitude.
- **Sliding Window Analysis:** Calculates rolling means and standard deviations for time-series segmentation.
- **Visual Dashboard:** Generates a Matplotlib graph highlighting detected anomalies.

## Requirements
- Python 3.x
- `numpy`
- `matplotlib`

Install dependencies via:
```bash
pip install numpy matplotlib
