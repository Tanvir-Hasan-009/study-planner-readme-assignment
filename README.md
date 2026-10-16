# FallSafe — Elderly Safety System
> A fall detection project designed to help older adults living alone and support their caregivers.

## Problem & Users

When an older adult falls while alone, help may not arrive quickly. FallSafe aims to detect possible falls using wearable sensor data.

The project is intended for older adults, their families, and caregivers.

## Features / Solution

- Detect falls using accelerometer and gyroscope data.
- Distinguish falls from normal daily activities and movements that look like falls.
- Compare models trained individually with models trained through federated learning.
- Study how adding a new user affects the shared model.

Future work includes caregiver alerts and a separate AI guidance feature.

## Architecture

```text
Wearable Sensor Data
        ↓
Data Preprocessing
        ↓
Fall Detection Model
        ↓
Normal Activity / Fall-like Activity / Fall
        ↓
Caregiver Alert (planned)
```

Federated learning allows simulated home devices to train locally and share model updates with a central server.

## Prerequisites

- Python
- Jupyter Notebook
- NumPy, pandas, Matplotlib, and PyTorch
- SciPy, cryptography, and dp-accounting for the additional experiments
- SisFall dataset

## Installation

Install the required packages:

```bash
pip install notebook numpy pandas matplotlib torch scipy cryptography dp-accounting
```

Download the project notebook and place `SisFall.zip` in the same folder.

## Quick Start / Usage

1. Open `FallSafe_Part1_Part2_Combined.ipynb` in Jupyter Notebook.
2. Check the dataset path in the setup section.
3. Run the cells in order.
4. Review the results and charts saved in the `artifacts` folder.

## Example Output

The model classifies sensor windows into three categories:

```text
normal_adl     → Normal daily activity
fall_like_adl  → Movement that resembles a fall
fall           → Fall detected
```

These are example labels, not results from a live monitoring system.

## Data / Model (for AI projects)

The project uses the **SisFall dataset**, which contains sensor recordings of daily activities and performed falls.

A one-dimensional convolutional neural network (CNN) classifies the motion data. The project compares local training, centralized training, and two federated learning methods: **FedAvg** and **FedProx**.

## Limitations

- The project is a research prototype.
- It has been evaluated using recorded data, not everyday home monitoring.
- It may miss falls or produce false alarms.
- Federated learning is simulated on one computer.
- Caregiver alerts and AI guidance are future work.
- It is not a clinically validated medical device.

## Contributing · License · Contact

**Contributing:** Suggestions and improvements are welcome through GitHub Issues and pull requests.

**License:** A project license has not been selected yet.

**Contact:** [Tanvir Hasan on GitHub](https://github.com/Tanvir-Hasan-009)

Prepared for the assignment given by **Dr. Jia Uddin**.
