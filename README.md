# RSAR_kinematics
Predict camera 6-DOF pose from 2-DOF pan/tilt input using a multi-task LSTM-Attention model

## 📜 Overview

This project trains and evaluates a deep learning model (**LSTM with Attention and Residual Blocks**) to predict **camera pose** (position and orientation) based on **pan and tilt angles**. It takes pan (`p_deg`) and tilt (`t_deg`) angles as input and predicts the camera's 3D position (`camPosX`, `camPosY`, `camPosZ`) and 3D orientation (represented as Euler angles `camEulerX_deg`, `camEulerY_deg`, `camEulerZ_deg`, but predicted internally using quaternions). The project also includes scripts for evaluating the trained model and exporting prediction results alongside comparisons with classical machine learning models (Polynomial Regression, SVR, Random Forest).

* **Core Function**: Predict camera 6-DOF pose from 2-DOF pan/tilt input using a multi-task LSTM-Attention model.
* **Evaluation**: Calculate position (Euclidean distance) and rotation (geodesic distance/angular error) metrics.
* **Comparison**: Export results and errors to Excel, including comparisons with baseline ML models.

---

## 🛠️ Installation

Follow these steps to set up the environment for running the project.

### 1. Clone Repository

```bash
git clone [https://github.com/](https://github.com/)[Your-Username]/[Your-Repo-Name].git
cd [Your-Repo-Name]

💾 Dataset (dataset.txt)
The dataset used for this project is provided in dataset.txt.

Source: Simulation (Specify further if known)

Format: Text file. The first line is a header comment. Subsequent lines contain 9 comma-separated (or space-separated after code processing) values per record:

id: Record identifier.

p_deg: Pan angle in degrees.

t_deg: Tilt angle in degrees.

camPosX, camPosY, camPosZ: Camera position coordinates.

camEulerX_deg, camEulerY_deg, camEulerZ_deg: Camera orientation as Euler angles (XYZ order) in degrees.

Preparation:

Place the dataset.txt file in the root directory of the project.

No further preparation steps are needed; the scripts handle loading directly.
