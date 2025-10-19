import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from scipy.spatial.transform import Rotation as R
import os
import joblib

# --- Utility Functions ---
def eulers_to_quaternions(eulers_deg):
    """Converts Euler angles (in degrees) to quaternions."""
    rotations = R.from_euler('xyz', eulers_deg, degrees=True)
    return rotations.as_quat()

class GeodesicLoss(nn.Module):
    """Loss function to calculate the geodesic distance (angle) between quaternions."""
    def __init__(self, eps=1e-7):
        super(GeodesicLoss, self).__init__()
        self.eps = eps

    def forward(self, y_pred, y_true):
        dot_product = torch.sum(y_pred * y_true, dim=1)
        dot_product = torch.clamp(torch.abs(dot_product), -1.0 + self.eps, 1.0 - self.eps)
        angle = 2 * torch.acos(dot_product)
        return torch.mean(angle)

def create_sequences(X, y_pos, y_rot, sequence_length):
    """Generates time-series data using a sliding window approach."""
    X_seq, y_pos_seq, y_rot_seq = [], [], []
    for i in range(len(X) - sequence_length):
        X_seq.append(X[i:i + sequence_length])
        y_pos_seq.append(y_pos[i + sequence_length - 1])
        y_rot_seq.append(y_rot[i + sequence_length - 1])
    return np.array(X_seq), np.array(y_pos_seq), np.array(y_rot_seq)

# -- Data Loading and Preprocessing --
def load_and_preprocess_data(file_path, sequence_length):
    """Loads the data and performs all preprocessing, including time-series transformation."""
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
    
    print(f"Original feature count: {X_base.shape[1]}")
    print(f"Polynomial feature count: {X_poly.shape[1]}")

    y_pos = df[['camPosX', 'camPosY', 'camPosZ']].values
    euler_angles = df[['camEulerX_deg', 'camEulerY_deg', 'camEulerZ_deg']].values
    y_quats = eulers_to_quaternions(euler_angles)
    y_quats[y_quats[:, 3] < 0] *= -1

    X_seq, y_pos_seq, y_quat_seq = create_sequences(X_poly, y_pos, y_quats, sequence_length)

    X_train, X_val, y_train_pos, y_val_pos, y_train_quat, y_val_quat = \
        train_test_split(X_seq, y_pos_seq, y_quat_seq, test_size=0.2, random_state=42)
    
    # Scaling 3D time-series data
    nsamples, nx, ny = X_train.shape
    X_train_2d = X_train.reshape((nsamples*nx, ny))
    scaler_x = StandardScaler().fit(X_train_2d)
    X_train_scaled_2d = scaler_x.transform(X_train_2d)
    X_train = X_train_scaled_2d.reshape(nsamples, nx, ny)

    nsamples, nx, ny = X_val.shape
    X_val_2d = X_val.reshape((nsamples*nx, ny))
    X_val_scaled_2d = scaler_x.transform(X_val_2d)
    X_val = X_val_scaled_2d.reshape(nsamples, nx, ny)

    scaler_pos = StandardScaler().fit(y_train_pos)
    y_train_pos_scaled = scaler_pos.transform(y_train_pos)
    y_val_pos_scaled = scaler_pos.transform(y_val_pos)
    
    data = {
        'train': (X_train, y_train_pos_scaled, y_train_quat),
        'val': (X_val, y_val_pos_scaled, y_val_quat),
        'val_orig': (y_val_pos, y_val_quat)
    }
    return data, scaler_x, scaler_pos

# --- Dataset and Model Definition ---
class MultiTaskDataset(Dataset):
    def __init__(self, features, pos_labels, rot_labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.pos_labels = torch.tensor(pos_labels, dtype=torch.float32)
        self.rot_labels = torch.tensor(rot_labels, dtype=torch.float32)
    def __len__(self): return len(self.features)
    def __getitem__(self, idx): return self.features[idx], self.pos_labels[idx], self.rot_labels[idx]

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
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
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

# --- Training and Evaluation Functions ---
def train_and_evaluate(model, train_loader, val_loader, pos_criterion, rot_criterion, lambda_rot, optimizer, scheduler, epochs, device, scaler_pos, y_val_orig_pos, y_val_orig_quat, patience=50):
    print("\n--- Training for Sequential Multi-Task Attention Model ---")
    best_val_rot_mae = float('inf')
    early_stopping_counter = 0

    for epoch in range(epochs):
        model.train()
        for features, pos_true, rot_true in train_loader:
            features, pos_true, rot_true = features.to(device), pos_true.to(device), rot_true.to(device)
            optimizer.zero_grad()
            pos_pred, rot_pred = model(features)
            rot_pred_normalized = torch.nn.functional.normalize(rot_pred, p=2, dim=1)
            loss_pos = pos_criterion(pos_pred, pos_true)
            loss_rot = rot_criterion(rot_pred_normalized, rot_true)
            total_loss = loss_pos + lambda_rot * loss_rot
            total_loss.backward()
            optimizer.step()
        
        model.eval()
        val_pos_preds, val_rot_preds = [], []
        with torch.no_grad():
            for features, _, _ in val_loader:
                features = features.to(device)
                pos_pred, rot_pred = model(features)
                val_pos_preds.append(pos_pred.cpu().numpy())
                val_rot_preds.append(torch.nn.functional.normalize(rot_pred, p=2, dim=1).cpu().numpy())
        
        val_pos_preds = scaler_pos.inverse_transform(np.concatenate(val_pos_preds))
        val_rot_preds = np.concatenate(val_rot_preds)
        pos_error = np.mean(np.sqrt(np.sum((val_pos_preds - y_val_orig_pos)**2, axis=1)))
        
        dot_products = np.abs(np.sum(val_rot_preds * y_val_orig_quat, axis=1))
        dot_products = np.clip(dot_products, -1.0, 1.0)
        angular_distances_deg = np.degrees(2 * np.arccos(dot_products))
        rot_mae = np.mean(angular_distances_deg)

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch+1}/{epochs} | Val Pos Error: {pos_error:.4f} | Val Rot MAE: {rot_mae:.4f}° | LR: {current_lr:.6f}')

        if rot_mae < best_val_rot_mae:
            best_val_rot_mae = rot_mae
            torch.save(model.state_dict(), 'best_model_sequential.pth')
            print(f'   -> Best model saved. Val Rot MAE: {rot_mae:.4f}°')
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        if early_stopping_counter >= patience:
            print(f'   !!! Early stopping triggered after {patience} epochs.'); break

    print("--- Finished training for Sequential Model ---")

# -- Main Execution Logic --
if __name__ == "__main__":
    SEQUENCE_LENGTH = 10
    FILE_PATH = 'dataset.txt'; LEARNING_RATE = 1e-4; BATCH_SIZE = 256; EPOCHS = 500
    EARLY_STOPPING_PATIENCE = 50
    LAMBDA_ROT = 20.0
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); print(f'Using device: {device}')

    data, scaler_x, scaler_pos = load_and_preprocess_data(FILE_PATH, SEQUENCE_LENGTH)

    print("Saving scalers to 'scalers/' directory...")
    os.makedirs('scalers', exist_ok=True)
    joblib.dump(scaler_x, 'scalers/scaler_x.pkl')
    joblib.dump(scaler_pos, 'scalers/scaler_pos.pkl')
    print("Scalers saved successfully.")

    
    X_train, y_train_pos_scaled, y_train_quat = data['train']
    X_val, y_val_pos_scaled, y_val_quat = data['val']
    y_val_orig_pos, y_val_orig_quat = data['val_orig']
    
    input_features = X_train.shape[2]
    
    train_dataset = MultiTaskDataset(X_train, y_train_pos_scaled, y_train_quat)
    val_dataset = MultiTaskDataset(X_val, y_val_pos_scaled, y_val_quat)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = SequentialPredictor(input_size=input_features).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2, eta_min=1e-7)

    train_and_evaluate(
        model, train_loader, val_loader, nn.MSELoss(), GeodesicLoss(), LAMBDA_ROT,
        optimizer, scheduler, EPOCHS, device, scaler_pos,
        y_val_orig_pos, y_val_orig_quat, EARLY_STOPPING_PATIENCE
    )