import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import json
from datetime import datetime
import wandb

class WorkoutDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.FloatTensor(labels)
    def __len__(self): return len(self.sequences)
    def __getitem__(self, idx): return self.sequences[idx], self.labels[idx]

class FatigueLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super(FatigueLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return torch.sigmoid(out)
    def get_input_size(self): return self.lstm.input_size

class FatigueDataProcessor:
    def __init__(self, window_size, overlap):
        self.window_size = window_size
        self.overlap = overlap
        self.scaler = StandardScaler()
        self.joints = ['left_knee', 'right_knee', 'left_hip', 'right_hip', 'left_ankle', 'right_ankle',
                       'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist',
                       'neck', 'left_spine', 'right_spine', 'left_hip_torso', 'right_hip_torso',
                       'left_shoulder_torso', 'right_shoulder_torso']

    def extract_features(self, df):
        features = []
        for joint in self.joints:
            joint_data = df[joint].values
            features.extend([joint_data, np.gradient(joint_data), np.gradient(np.gradient(joint_data))])
        for left, right in [('left_knee', 'right_knee'), ('left_hip', 'right_hip'), ('left_elbow', 'right_elbow')]:
            symmetry = df[left].values - df[right].values
            features.append(symmetry)
        return np.array(features).T

    def create_sequences(self, features, labels):
        step = int(self.window_size * (1 - self.overlap))
        sequences, sequence_labels = [], []
        for i in range(0, len(features) - self.window_size + 1, step):
            window = features[i:i + self.window_size]
            sequences.append(window)
            sequence_labels.append(labels[i + self.window_size - 1])
        return np.array(sequences), np.array(sequence_labels)
    
    def process_workout_data(self, data_dir):
        all_features, all_labels = [], []
        for filename in os.listdir(data_dir):
            if filename.endswith('.csv'):
                df = pd.read_csv(os.path.join(data_dir, filename))
                features = self.extract_features(df)
                fatigue_labels = df['fatigue_label'].values / 10.0
                all_features.append(features)
                all_labels.append(fatigue_labels)
        
        all_features = np.concatenate(all_features)
        all_labels = np.concatenate(all_labels)
        sequences, labels = self.create_sequences(all_features, all_labels)
        
        train_seq, temp_seq, train_labels, temp_labels = train_test_split(
            sequences, labels, test_size=0.3, random_state=42
        )
        val_seq, test_seq, val_labels, test_labels = train_test_split(
            temp_seq, temp_labels, test_size=0.5, random_state=42
        )

        hist, bins = np.histogram(labels, bins=10, range=(0, 1))
        print("Histogram of fatigue labels (0-1 scale):")
        print(f"Counts: {hist}")
        print(f"Bin edges: {bins}")
        wandb.log({"Fatigue Label Distribution": wandb.Histogram(np_histogram=(hist, bins))})
        
        print(f"Processed {len(sequences)} sequences from {len(os.listdir(data_dir))} files")
        return train_seq, train_labels, val_seq, val_labels, test_seq, test_labels

class FatigueModelTrainer:
    def __init__(self, model, learning_rate, batch_size, weight_decay):
        self.model = model
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def train(self, train_sequences, train_labels, val_sequences, val_labels, scaler, epochs, early_stopping):
        train_sequences_scaled = np.array([scaler.transform(seq) for seq in train_sequences])
        val_sequences_scaled = np.array([scaler.transform(seq) for seq in val_sequences])

        train_dataset = WorkoutDataset(train_sequences_scaled, train_labels)
        val_dataset = WorkoutDataset(val_sequences_scaled, val_labels)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)

        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
        criterion = nn.MSELoss()
        best_val_loss = float('inf')
        epochs_no_improve = 0
        tolerance = 1e-4

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            for sequences, labels in train_loader:
                sequences, labels = sequences.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(sequences)
                loss = criterion(outputs, labels.unsqueeze(1))
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for sequences, labels in val_loader:
                    sequences, labels = sequences.to(self.device), labels.to(self.device)
                    outputs = self.model(sequences)
                    val_loss += criterion(outputs, labels.unsqueeze(1)).item()

            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)

            # Log every epoch
            wandb.log({"epoch": epoch, "train_loss": avg_train_loss, "val_loss": avg_val_loss})

            if avg_val_loss < best_val_loss - tolerance:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
                self.save_model('best_model.pt')
                wandb.run.summary["best_val_loss"] = best_val_loss
            else:
                epochs_no_improve += 1

            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, '
                      f'Epochs No Improve: {epochs_no_improve}, Best Val Loss: {best_val_loss:.6f}')

            if epochs_no_improve >= early_stopping:
                print(f'Early stopping triggered after epoch {epoch+1} with patience {early_stopping}')
                break

        return {"best_val_loss": best_val_loss}

    def evaluate(self, test_sequences, test_labels, scaler):
        test_sequences_scaled = np.array([scaler.transform(seq) for seq in test_sequences])
        test_dataset = WorkoutDataset(test_sequences_scaled, test_labels)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size)
        
        self.model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for sequences, labels in test_loader:
                sequences = sequences.to(self.device)
                outputs = self.model(sequences)
                all_preds.extend(outputs.cpu().numpy().flatten())
                all_labels.extend(labels.numpy().flatten())

        all_preds = np.clip(all_preds, 0, 1)

        metrics = {
            'mse': float(mean_squared_error(all_labels, all_preds)),
            'rmse': float(np.sqrt(mean_squared_error(all_labels, all_preds))),
            'mae': float(mean_absolute_error(all_labels, all_preds)),
            'r2': float(r2_score(all_labels, all_preds))
        }

        baseline_preds = np.roll(test_labels, 1)
        baseline_preds[0] = test_labels[0]
        baseline_r2 = float(r2_score(test_labels, baseline_preds))

        # Log all metrics
        wandb.run.summary.update(metrics)
        wandb.run.summary["baseline_r2"] = baseline_r2
        wandb.log({"Pred vs Actual": wandb.plot.scatter(
            wandb.Table(data=[[x, y] for x, y in zip(all_labels, all_preds)], columns=["Actual", "Predicted"]),
            "Actual", "Predicted", title="Predicted vs Actual Fatigue"
        )})

        print(f"Baseline R^2 (predicting previous label): {baseline_r2:.4f}")
        return metrics, all_preds, all_labels

    def save_model(self, filename):
        model_info = {
            'state_dict': self.model.state_dict(),
            'input_size': self.model.lstm.input_size,  # Fixed reference
            'hidden_size': self.model.hidden_size,
            'num_layers': self.model.num_layers,
            'date_trained': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        torch.save(model_info, filename)

def train_fatigue_model(data_dir='session_data', save_dir='models'):
    wandb.init(project="fatigue_prediction_sweep")
    config = wandb.config

    print(f"Running with config: {dict(config)}")

    processor = FatigueDataProcessor(window_size=config.window_size, overlap=config.overlap)
    train_seq, train_labels, val_seq, val_labels, test_seq, test_labels = processor.process_workout_data(data_dir)

    train_seq_stacked = np.vstack(train_seq)
    processor.scaler.fit(train_seq_stacked)

    input_feature_count = train_seq.shape[2]
    model = FatigueLSTM(
        input_size=input_feature_count,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        output_size=1,
        dropout=config.dropout
    )

    trainer = FatigueModelTrainer(
        model,
        learning_rate=config.learning_rate,
        batch_size=config.batch_size,
        weight_decay=config.weight_decay
    )

    history = trainer.train(
        train_seq, train_labels, val_seq, val_labels, processor.scaler,
        epochs=config.epochs, early_stopping=config.early_stopping
    )

    metrics, preds, actual = trainer.evaluate(test_seq, test_labels, processor.scaler)
    print("Evaluation Metrics:", metrics)

    run_dir = os.path.join(save_dir, wandb.run.id)
    os.makedirs(run_dir, exist_ok=True)
    trainer.save_model(os.path.join(run_dir, 'best_model.pt'))
    torch.save(processor.scaler, os.path.join(run_dir, 'scaler.pt'))

    wandb.finish()

if __name__ == '__main__':
    train_fatigue_model()