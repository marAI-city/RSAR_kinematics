import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from scipy.spatial.transform import Rotation as R
import os
import joblib

# Import PyTorch libraries
import torch
import torch.nn as nn

print("--- Script Start ---")

# --- Utility Functions ---
def eulers_to_quaternions(eulers_deg):
    rotations = R.from_euler('xyz', eulers_deg, degrees=True)
    return rotations.as_quat()

# --- AI Model Architecture Definition (Original, matching the saved model) ---
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

# --- Data Loading ---
def load_full_dataset(file_path):
    print(f"Loading data from: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()[1:]
    full_content = ' '.join(lines).replace(',', ' ')
    numbers = [float(s) for s in full_content.split()]
    num_records = len(numbers) // 9
    data_array = np.array(numbers).reshape(num_records, 9)
    df = pd.DataFrame(data_array, columns=['id', 'p_deg', 't_deg', 'camPosX', 'camPosY', 'camPosZ', 'camEulerX_deg', 'camEulerY_deg', 'camEulerZ_deg'])
    print("Data loading complete.")
    return df

# --- Main Execution Logic ---
if __name__ == "__main__":
    FILE_PATH = 'dataset.txt'
    SEQUENCE_LENGTH = 10
    
    df_full = load_full_dataset(FILE_PATH)

    X_sincos_full = np.hstack([np.sin(np.deg2rad(df_full[['p_deg']])), np.cos(np.deg2rad(df_full[['p_deg']])),
                               np.sin(np.deg2rad(df_full[['t_deg']])), np.cos(np.deg2rad(df_full[['t_deg']]))])
    y_pos_full = df_full[['camPosX', 'camPosY', 'camPosZ']].values
    y_eulers_full = df_full[['camEulerX_deg', 'camEulerY_deg', 'camEulerZ_deg']].values
    y_quats_full = eulers_to_quaternions(y_eulers_full)
    y_quats_full[y_quats_full[:, 3] < 0] *= -1
    y_full = np.hstack([y_pos_full, y_quats_full])
    
    scaler_y_classical = StandardScaler()
    y_full_scaled_classical = scaler_y_classical.fit_transform(y_full)

    models = {
        "Polynomial_Regression": Pipeline([
            ('poly', PolynomialFeatures(degree=3)),
            ('regressor', MultiOutputRegressor(LinearRegression()))
        ]),
        "SVR": MultiOutputRegressor(SVR(kernel='rbf', C=1.0, epsilon=0.1)),
        "Random_Forest": MultiOutputRegressor(RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1))
    }

    error_excel_writer = pd.ExcelWriter('errors.xlsx', engine='xlsxwriter')
    prediction_excel_writer = pd.ExcelWriter('predictions.xlsx', engine='xlsxwriter')
    
    for name, model in models.items():
        print(f"\n--- Started processing classical model: {name} ---")
        model.fit(X_sincos_full, y_full_scaled_classical)
        y_pred_scaled = model.predict(X_sincos_full)
        y_pred = scaler_y_classical.inverse_transform(y_pred_scaled)
        
        pred_pos = y_pred[:, :3]
        pred_quat_raw = y_pred[:, 3:]
        norm = np.linalg.norm(pred_quat_raw, axis=1, keepdims=True)
        pred_quat = np.divide(pred_quat_raw, norm, out=np.zeros_like(pred_quat_raw), where=norm!=0)
        pred_eulers = R.from_quat(pred_quat).as_euler('xyz', degrees=True)

        prediction_df = pd.DataFrame({
            'p_deg': df_full['p_deg'], 't_deg': df_full['t_deg'],
            'pred_camPosX': pred_pos[:, 0], 'pred_camPosY': pred_pos[:, 1], 'pred_camPosZ': pred_pos[:, 2],
            'pred_camEulerX_deg': pred_eulers[:, 0], 'pred_camEulerY_deg': pred_eulers[:, 1], 'pred_camEulerZ_deg': pred_eulers[:, 2]
        })
        prediction_df.to_excel(prediction_excel_writer, sheet_name=name, index=False)
        print(f"Saved prediction results for '{name}' to 'predictions.xlsx'.")

        error_df = pd.DataFrame({'p_deg': df_full['p_deg'], 't_deg': df_full['t_deg']})
        error_df['error_camPosX'] = pred_pos[:, 0] - df_full['camPosX']
        error_df['error_camPosY'] = pred_pos[:, 1] - df_full['camPosY']
        error_df['error_camPosZ'] = pred_pos[:, 2] - df_full['camPosZ']
        
        for i, axis in enumerate(['X', 'Y', 'Z']):
            error = pred_eulers[:, i] - df_full[f'camEuler{axis}_deg']
            error_df[f'error_camEuler{axis}_deg'] = (error + 180) % 360 - 180
            
        error_df.to_excel(error_excel_writer, sheet_name=name, index=False)
        print(f"Saved error results for '{name}' to 'errors.xlsx'.")

    print("\n--- Started processing AI model: AI_Model ---")
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        scaler_x_ai = joblib.load('scalers/scaler_x.pkl')
        scaler_pos_ai = joblib.load('scalers/scaler_pos.pkl')

        poly_features = PolynomialFeatures(degree=2, include_bias=False)
        X_poly_full = poly_features.fit_transform(X_sincos_full)
        X_scaled_full = scaler_x_ai.transform(X_poly_full)
        
        X_seq = []
        for i in range(len(X_scaled_full) - SEQUENCE_LENGTH + 1):
            X_seq.append(X_scaled_full[i:i + SEQUENCE_LENGTH])
        X_seq = np.array(X_seq)
        
        input_features = X_seq.shape[2]
        ai_model = SequentialPredictor(input_size=input_features).to(device)
        
        # --- ✨ Improvement: Added `weights_only=True` to remove warning ---
        ai_model.load_state_dict(torch.load('best_model_sequential.pth', map_location=device, weights_only=True))
        ai_model.eval()

        with torch.no_grad():
            X_tensor = torch.tensor(X_seq, dtype=torch.float32).to(device)
            pos_preds_scaled, rot_preds_raw = [], []
            batch_size = 256
            for i in range(0, len(X_tensor), batch_size):
                batch_x = X_tensor[i:i + batch_size]
                pos_pred, rot_pred = ai_model(batch_x)
                pos_preds_scaled.append(pos_pred.cpu().numpy())
                rot_preds_raw.append(rot_pred.cpu().numpy())
            
            pos_pred_scaled = np.concatenate(pos_preds_scaled, axis=0)
            rot_pred_raw = np.concatenate(rot_preds_raw, axis=0)

        pred_pos_ai = scaler_pos_ai.inverse_transform(pos_pred_scaled)
        norm_ai = np.linalg.norm(rot_pred_raw, axis=1, keepdims=True)
        
        # --- 🐞 ERROR FIX: Corrected variable name typo ---
        pred_quat_ai = np.divide(rot_pred_raw, norm_ai, out=np.zeros_like(rot_pred_raw), where=norm_ai!=0)
        pred_eulers_ai = R.from_quat(pred_quat_ai).as_euler('xyz', degrees=True)
        
        df_ai_true = df_full.iloc[SEQUENCE_LENGTH - 1:].reset_index(drop=True)

        prediction_df_ai = pd.DataFrame({
            'p_deg': df_ai_true['p_deg'], 't_deg': df_ai_true['t_deg'],
            'pred_camPosX': pred_pos_ai[:, 0], 'pred_camPosY': pred_pos_ai[:, 1], 'pred_camPosZ': pred_pos_ai[:, 2],
            'pred_camEulerX_deg': pred_eulers_ai[:, 0], 'pred_camEulerY_deg': pred_eulers_ai[:, 1], 'pred_camEulerZ_deg': pred_eulers_ai[:, 2]
        })
        prediction_df_ai.to_excel(prediction_excel_writer, sheet_name='AI_Model', index=False)
        print("Saved prediction results for 'AI_Model' to 'predictions.xlsx'.")

        error_df_ai = pd.DataFrame({'p_deg': df_ai_true['p_deg'], 't_deg': df_ai_true['t_deg']})
        error_df_ai['error_camPosX'] = pred_pos_ai[:, 0] - df_ai_true['camPosX'].values
        error_df_ai['error_camPosY'] = pred_pos_ai[:, 1] - df_ai_true['camPosY'].values
        error_df_ai['error_camPosZ'] = pred_pos_ai[:, 2] - df_ai_true['camPosZ'].values
        
        for i, axis in enumerate(['X', 'Y', 'Z']):
            error = pred_eulers_ai[:, i] - df_ai_true[f'camEuler{axis}_deg'].values
            error_df_ai[f'error_camEuler{axis}_deg'] = (error + 180) % 360 - 180
        
        error_df_ai.to_excel(error_excel_writer, sheet_name='AI_Model', index=False)
        print("Saved error results for 'AI_Model' to 'errors.xlsx'.")

    except Exception as e:
        print(f"\n--- ⚠️ An error occurred while processing the AI model ---")
        print(f"Error: {e}")

    error_excel_writer.close()
    prediction_excel_writer.close()
    print("\n--- ✅ All tasks completed ---")
    print("Please check 'errors.xlsx' and 'predictions.xlsx' files.")