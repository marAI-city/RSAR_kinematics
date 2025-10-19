# RSAR_kinematics
Predict camera 6-DOF pose from 2-DOF pan/tilt input using a multi-task LSTM-Attention model

# 🎯 Multi-Task Attention LSTM for Camera Pose Estimation
This project implements a **multi-task learning model** based on **Attention-enhanced LSTM**  
to predict **camera position** and **orientation** (rotation) from input parameters.  
It combines sequential modeling (LSTM) with Transformer-like attention to achieve  
high-precision camera parameter estimation.

---

## 📂 Project Structure

```
├── train_multitask_attention_LSTM.py   # Training script
├── eval_multitask_attention_LSTM.py    # Evaluation script
├── exportResultExcel.py                # Exports AI & Classical model results to Excel
├── dataset.txt                         # Input dataset (camera pose records)
├── scalers/                            # Saved normalization scalers
│   ├── scaler_x.pkl
│   └── scaler_pos.pkl
└── best_model_sequential.pth           # Trained model weights
```

---

## 🧠 Model Overview

The model performs **multi-task regression** with shared representation learning:

| Task | Output | Loss Function |
|------|---------|---------------|
| Camera Position | (X, Y, Z) | Mean Squared Error (MSELoss) |
| Camera Rotation | Quaternion (x, y, z, w) | Geodesic Loss (Angular Distance) |

**Architecture Highlights:**
- Bidirectional **LSTM Encoder**
- **3 Residual Blocks**
- **Multi-Head Self-Attention Layer**
- Two separate output heads for **Position** and **Rotation**
- Combined total loss:  
  `Loss = MSE(Position) + λ × Geodesic(Rotation)`

---

## 🧩 Dataset Format
`dataset.txt` is a plain text file with the following columns:
| id | p_deg | t_deg | camPosX | camPosY | camPosZ | camEulerX_deg | camEulerY_deg | camEulerZ_deg |
- `p_deg`, `t_deg`: input angular parameters (encoded as sin/cos pairs)
- `camPos*`: ground-truth camera position
- `camEuler*`: camera rotation in degrees (converted to quaternion internally)

---

## 🚀 How to Run
### 1️⃣ Train the Model
```bash
python train_multitask_attention_LSTM.py
```
- Loads `dataset.txt` and prepares sequential data.
- Saves feature and position scalers under `scalers/`.
- The model checkpoints automatically when validation rotation MAE improves.

---

### 2️⃣ Evaluate Performance
```bash
python eval_multitask_attention_LSTM.py
```
- Loads `best_model_sequential.pth` and evaluates on the test split.
- Prints detailed metrics:
  - **Mean Euclidean Distance Error** (Position)
  - **Mean Angular Distance Error** (Rotation)
  - **Per-Axis Euler MAE (Roll, Pitch, Yaw)**

Example Output:
```
Position Mean Euclidean Distance Error: 0.0784
Rotation Mean Angular Distance Error: 2.015°
MAE Euler X (Roll): 1.22°
MAE Euler Y (Pitch): 1.54°
MAE Euler Z (Yaw): 1.03°
```

---

### 3️⃣ Export Excel Results
```bash
python exportResultExcel.py
```
- Runs both **Classical ML models** (SVR, Random Forest, Polynomial Regression)  
  and the **AI model** for comparison.
- Generates:
  - `predictions.xlsx` → predicted values for all models  
  - `errors.xlsx` → per-axis prediction errors

---

## 🧮 Environment (conda)
Below is a summary of **non-default dependencies** required to run the project.

| Package | Version | Source |
|----------|----------|---------|
| torch | 2.5.1 | pytorch |
| torchvision | 0.20.1 | pypi |
| torchaudio | 2.5.1 | pypi |
| onnx | 1.19.1 | pypi |
| onnxruntime | 1.23.1 | pypi |
| joblib | 1.5.2 | conda-forge |
| scikit-learn | 1.7.2 | conda-forge |
| scipy | 1.15.2 | conda-forge |
| pandas | 2.3.3 | conda-forge |
| numpy | 2.0.1 | conda |
| tqdm | 4.67.1 | conda-forge |
| matplotlib | 3.10.6 | conda |
| seaborn | 0.13.2 | conda |
| coloredlogs | 15.0.1 | pypi |

> 💡 The project was developed under **CUDA 12.1**  
> (`pytorch-cuda=12.1` is required for GPU acceleration).

---

## 📜 License

```
MIT License

Copyright (c) 2025 Ahyun Lee

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the “Software”), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
```

---
