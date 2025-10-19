# RSAR_kinematics
Predict camera 6-DOF pose from 2-DOF pan/tilt input using a multi-task LSTM-Attention model
아래는 당신이 제공한 **3개의 주요 Python 코드 (`train_multitask_attention_LSTM.py`, `eval_multitask_attention_LSTM.py`, `exportResultExcel.py`)**,
`dataset.txt`, 그리고 **conda 환경 정보**를 기반으로 정리한 GitHub용 `README.md` 초안입니다.

이 파일은 바로 GitHub에 업로드하거나 복사해 사용할 수 있습니다.

---

```markdown
# 🎯 Multi-Task Attention LSTM for Camera Pose Estimation

이 프로젝트는 **다중 작업 기반의 Attention LSTM 모델**을 사용하여  
입력된 **시점 정보(p, t)**로부터 **카메라 위치(Position)**와 **회전(Rotation)**을 예측하는 모델입니다.  
PyTorch 기반의 Transformer-like 구조를 이용해 **정확한 카메라 파라미터 추정**을 수행합니다.

---

## 📂 Repository Structure

```

├── train_multitask_attention_LSTM.py   # 모델 학습 (훈련용)
├── eval_multitask_attention_LSTM.py    # 모델 성능 평가 및 결과 출력
├── exportResultExcel.py                # AI 및 Classical 모델 결과 Excel 내보내기
├── dataset.txt                         # 입력 데이터 (id, p_deg, t_deg, camPos, camEuler)
├── scalers/                            # 학습 중 저장되는 표준화 스케일러
│   ├── scaler_x.pkl
│   └── scaler_pos.pkl
└── best_model_sequential.pth           # 훈련 완료된 모델 가중치 (자동 저장)

````

---

## 🧠 Model Overview

본 모델은 **다중 작업 학습(Multi-task learning)** 구조로 설계되어 있습니다.

| Task | Output | Loss Function |
|------|---------|----------------|
| Camera Position | (X, Y, Z) | MSELoss |
| Camera Rotation | Quaternion (x, y, z, w) | Geodesic Loss (각도 거리) |

모델 구조:
- **LSTM 기반 시계열 인코더** (Bidirectional)
- **Residual Blocks ×3**
- **Multi-Head Self-Attention Block**
- **Position/Rotation Dual Heads**

학습 시 Position과 Rotation을 동시에 예측하며,  
손실 함수는 `loss = MSE(Position) + λ * Geodesic(Rotation)` 형태로 결합됩니다.

---

## 🧩 Data Format

`dataset.txt`는 아래 구조로 되어 있습니다.

| id | p_deg | t_deg | camPosX | camPosY | camPosZ | camEulerX_deg | camEulerY_deg | camEulerZ_deg |
|----|--------|--------|----------|----------|----------|----------------|----------------|----------------|
| 0 | 15.3 | 45.1 | 0.1 | 0.5 | 2.3 | 180 | -10 | 5 |
| ... | ... | ... | ... | ... | ... | ... | ... | ... |

- **p_deg, t_deg**: 입력 파라미터 (cyclical encoding 사용)
- **camPos\***: 카메라 위치
- **camEuler\***: 카메라 회전 (deg 단위, quaternion으로 변환됨)

---

## ⚙️ Training Configuration

| 항목 | 값 |
|------|-----|
| Sequence Length | 10 |
| Batch Size | 256 |
| Learning Rate | 1e-4 |
| Optimizer | AdamW |
| Scheduler | CosineAnnealingWarmRestarts |
| Epochs | 500 |
| Early Stopping | 50 |
| Lambda (Rotation Loss Weight) | 20.0 |
| Device | CUDA / CPU 자동 선택 |

훈련 완료 시 `best_model_sequential.pth` 가 자동 저장됩니다.

---

## 🚀 How to Run

### 1️⃣ Training
```bash
python train_multitask_attention_LSTM.py
````

* `dataset.txt` 로부터 데이터를 로드하고, 스케일링 및 시퀀스 변환을 수행합니다.
* 학습 후 `scalers/` 폴더에 표준화 정보 저장.
* 성능이 향상된 모델은 자동으로 `best_model_sequential.pth`로 저장됩니다.

### 2️⃣ Evaluation

```bash
python eval_multitask_attention_LSTM.py
```

* 저장된 모델(`best_model_sequential.pth`)을 불러와 테스트 데이터셋으로 성능 측정.
* 출력:

  * **Position Mean Euclidean Error**
  * **Rotation Mean Angular Error (deg)**
  * **Per-axis Euler MAE (Roll, Pitch, Yaw)**

### 3️⃣ Export Results

```bash
python exportResultExcel.py
```

* AI 모델과 Classical 모델(SVR, RF, Polynomial Regression)의 결과를 비교.
* 결과 파일:

  * `predictions.xlsx` — 예측 결과
  * `errors.xlsx` — 오차 분석 결과

---

## 🧮 Environment (conda)

아래 표는 **기본 설치 외 추가된 주요 패키지**입니다.

| Package      | Version | Source      |
| ------------ | ------- | ----------- |
| torch        | 2.5.1   | pytorch     |
| torchvision  | 0.20.1  | pypi        |
| torchaudio   | 2.5.1   | pypi        |
| onnx         | 1.19.1  | pypi        |
| onnxruntime  | 1.23.1  | pypi        |
| joblib       | 1.5.2   | conda-forge |
| scikit-learn | 1.7.2   | conda-forge |
| scipy        | 1.15.2  | conda-forge |
| pandas       | 2.3.3   | conda-forge |
| numpy        | 2.0.1   | conda       |
| tqdm         | 4.67.1  | conda-forge |
| matplotlib   | 3.10.6  | conda       |
| seaborn      | 0.13.2  | conda       |
| coloredlogs  | 15.0.1  | pypi        |

> 💡 CUDA 12.1 환경 기반으로 `pytorch-cuda=12.1` 설치됨.

---

## 📊 Output Example

**Training Log**

```
Epoch 41/500 | Val Pos Error: 0.0812 | Val Rot MAE: 1.9345° | LR: 0.000087
   -> Best model saved. Val Rot MAE: 1.9345°
```

**Evaluation Report**

```
Position Mean Euclidean Distance Error: 0.0784
Rotation Mean Angular Distance Error: 2.015°
MAE Euler X (Roll): 1.22°
MAE Euler Y (Pitch): 1.54°
MAE Euler Z (Yaw): 1.03°
```

---

## 📜 License

This project is licensed under the **MIT License**.
See the [LICENSE](LICENSE) file for details.

---

## ✨ Author

**이아현 (Ahyun Lee)**
📧 [ahyun.sch@gmail.com](mailto:ahyun.sch@gmail.com)
Meta&Game.SCH

```

---

원하신다면 다음 버전으로도 만들어드릴 수 있습니다:
1. 🇰🇷 **완전 한글 버전** (연구 보고서 스타일)
2. 🇺🇸 **영문 GitHub용 버전** (논문/공개 프로젝트 스타일)
3. 🧾 **README + LICENSE 자동 생성 버전 (MIT 포함)**

어떤 형식으로 최종 생성할까요?
```
