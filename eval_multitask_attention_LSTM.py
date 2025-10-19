import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from scipy.spatial.transform import Rotation as R
import os

# --- Utilities, Data Loaders, Model Architecture (Keep identical to training code) ---
def eulers_to_quaternions(eulers_deg):
    rotations = R.from_euler('xyz', eulers_deg, degrees=True)
    return rotations.as_quat()

def create_sequences(X, y_pos, y_rot, y_eulers, sequence_length):
    X_seq, y_pos_seq, y_rot_seq, y_eulers_seq = [], [], [], []
    for i in range(len(X) - sequence_length):
        X_seq.append(X[i:i + sequence_length])
        y_pos_seq.append(y_pos[i + sequence_length - 1])
        y_rot_seq.append(y_rot[i + sequence_length - 1])
        y_eulers_seq.append(y_eulers[i + sequence_length - 1]) # Also slice Euler angles according to sequence
    return np.array(X_seq), np.array(y_pos_seq), np.array(y_rot_seq), np.array(y_eulers_seq)

def load_and_preprocess_data_for_testing(file_path, sequence_length):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()[1:]
    full_content = ' '.join(lines).replace(',', ' ')
    numbers = [float(s) for s in full_content.split()]
    num_records = len(numbers) // 9
    data_array = np.array(numbers).reshape(num_records, 9)
    df = pd.DataFrame(data_array, columns=['id', 'p_deg', 't_deg', 'camPosX', 'camPosY', 'camPosZ', 'camEulerX_deg', 'camEulerY_deg', 'camEulerZ_deg'])

    input_cyclical = ['p_deg', 't_deg']
    for feat in input_cyclical:
        rad = np.deg2rad(df[feat]); df[f'{feat}_sin'] = np.sin(rad); df[f'{feat}_cos'] = np.cos(rad)
    
    features_sincos = [f'{feat}_{suf}' for feat in input_cyclical for suf in ['sin', 'cos']]
    X_base = df[features_sincos].values

    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X_base)
    
    # Assuming a scaler fitted on the entire training data and only performing transformation
    # In practice, it's most accurate to save and load the scaler_x used during training
    scaler_x = StandardScaler().fit(X_poly)
    X_scaled = scaler_x.transform(X_poly)
    
    y_pos = df[['camPosX', 'camPosY', 'camPosZ']].values
    y_orig_rot_eulers = df[['camEulerX_deg', 'camEulerY_deg', 'camEulerZ_deg']].values
    y_quats = eulers_to_quaternions(y_orig_rot_eulers)
    y_quats[y_quats[:, 3] < 0] *= -1

    # Convert the entire dataset into sequences
    X_seq, y_pos_seq, y_quat_seq, y_eulers_seq = create_sequences(X_scaled, y_pos, y_quats, y_orig_rot_eulers, sequence_length)

    # Split the test set using indices identically to train_test_split (assuming same random_state=42)
    test_size = int(len(X_seq) * 0.2)
    X_test = X_seq[-test_size:]
    y_test_orig_pos = y_pos_seq[-test_size:]
    y_test_orig_quat = y_quat_seq[-test_size:]
    y_test_orig_rot_eulers = y_eulers_seq[-test_size:]

    scaler_pos = StandardScaler().fit(y_pos) # Fit position scaler on the entire data as well
    
    test_data = {
        'X_test': X_test, 
        'y_test_orig_pos': y_test_orig_pos, 
        'y_test_orig_rot_eulers': y_test_orig_rot_eulers, 
        'y_test_orig_quat': y_test_orig_quat
    }
    return test_data, scaler_pos

class ResidualBlock(nn.Module):
    def __init__(self, size, dropout_p=0.4):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(nn.Linear(size, size), nn.BatchNorm1d(size), nn.ReLU(), nn.Dropout(dropout_p), nn.Linear(size, size), nn.BatchNorm1d(size))
        self.relu = nn.ReLU()
    def forward(self, x): return self.relu(x + self.block(x))

class AttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(AttentionBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(nn.Linear(embed_dim, embed_dim * 4), nn.ReLU(), nn.Linear(embed_dim * 4, embed_dim))
    def forward(self, x):
        x_seq = x.unsqueeze(1)
        attn_output, _ = self.attention(x_seq, x_seq, x_seq)
        x = self.norm1(x + attn_output.squeeze(1))
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        return x

class SequentialPredictor(nn.Module):
    def __init__(self, input_size, pos_output_size=3, rot_output_size=4, lstm_hidden_size=256, hidden_size=512, num_blocks=3, num_heads=8):
        super(SequentialPredictor, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=lstm_hidden_size, num_layers=2, batch_first=True, bidirectional=True, dropout=0.2)
        self.backbone_input = nn.Sequential(nn.Linear(lstm_hidden_size * 2, hidden_size), nn.BatchNorm1d(hidden_size), nn.ReLU())
        self.backbone_res = nn.Sequential(*[ResidualBlock(hidden_size) for _ in range(num_blocks)])
        self.attention_block = AttentionBlock(hidden_size, num_heads)
        self.pos_head = nn.Linear(hidden_size, pos_output_size)
        self.rot_head = nn.Linear(hidden_size, rot_output_size)
    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        last_hidden_state = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)
        features = self.backbone_res(self.backbone_input(last_hidden_state))
        attended_features = self.attention_block(features)
        pos_output = self.pos_head(attended_features)
        rot_output = self.rot_head(attended_features)
        return pos_output, rot_output

def test_performance(model, X_test, y_test_orig_pos, y_test_orig_rot_eulers, y_test_orig_quat, scaler_pos, device):
    print("\n--- 🚀 Starting Final Performance Test ---")
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    model.eval()
    with torch.no_grad():
        pos_pred_scaled, rot_pred_raw = model(X_test_tensor)
        pos_pred_scaled = pos_pred_scaled.cpu().numpy()
        rot_pred_raw = rot_pred_raw.cpu().numpy()

    predicted_positions = scaler_pos.inverse_transform(pos_pred_scaled)
    pred_norms = np.linalg.norm(rot_pred_raw, axis=1, keepdims=True)
    predicted_quats_normalized = rot_pred_raw / pred_norms

    print("\n--- ✅ Final Performance Report ---")
    pos_error = np.mean(np.sqrt(np.sum((predicted_positions - y_test_orig_pos)**2, axis=1)))
    print(f"Position Mean Euclidean Distance Error: {pos_error:.4f}")
    
    dot_products = np.abs(np.sum(predicted_quats_normalized * y_test_orig_quat, axis=1))
    dot_products = np.clip(dot_products, -1.0, 1.0)
    angular_distances_rad = 2 * np.arccos(dot_products)
    angular_distances_deg = np.degrees(angular_distances_rad)
    print(f"Rotation Mean Angular Distance Error: {np.mean(angular_distances_deg):.4f}°")

    print("\n--- Per-Axis Euler Angle MAE (for reference) ---")
    pred_rotations = R.from_quat(predicted_quats_normalized)
    pred_eulers_deg = pred_rotations.as_euler('xyz', degrees=True)
    error_x = np.abs(pred_eulers_deg[:, 0] - y_test_orig_rot_eulers[:, 0]); mae_ex = np.mean(np.minimum(error_x, 360 - error_x))
    error_y = np.abs(pred_eulers_deg[:, 1] - y_test_orig_rot_eulers[:, 1]); mae_ey = np.mean(np.minimum(error_y, 360 - error_y))
    error_z = np.abs(pred_eulers_deg[:, 2] - y_test_orig_rot_eulers[:, 2]); mae_ez = np.mean(np.minimum(error_z, 360 - error_z))
    print(f"  - MAE Euler X (Roll):  {mae_ex:.4f}°")
    print(f"  - MAE Euler Y (Pitch): {mae_ey:.4f}°")
    print(f"  - MAE Euler Z (Yaw):   {mae_ez:.4f}°")

if __name__ == "__main__":
    SEQUENCE_LENGTH = 10
    FILE_PATH = 'dataset.txt'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); print(f'Using device: {device}')
    
    test_data, scaler_pos = load_and_preprocess_data_for_testing(FILE_PATH, SEQUENCE_LENGTH)
    
    input_features = test_data['X_test'].shape[2]
    model = SequentialPredictor(input_size=input_features).to(device)

    try:
        print("Loading trained model weights...")
        model.load_state_dict(torch.load('best_model_sequential.pth', map_location=device))
        print("Model loaded successfully.")
    except FileNotFoundError as e:
        print(f"Error: {e}. Please run the training script first.")
        exit()

    test_performance(
        model, test_data['X_test'], test_data['y_test_orig_pos'], 
        test_data['y_test_orig_rot_eulers'], test_data['y_test_orig_quat'], 
        scaler_pos, device
    )