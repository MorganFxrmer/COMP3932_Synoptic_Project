import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime

class WorkoutDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.FloatTensor(labels)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

class FatigueLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(FatigueLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True, 
            dropout=dropout
        )
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return torch.sigmoid(out)
        
    def get_input_size(self):
        return self.lstm.input_size

class FatigueDataProcessor:
    def __init__(self, window_size=30, overlap=0.5):
        self.window_size = window_size
        self.overlap = overlap
        self.scaler = StandardScaler()
        self.joints = [
            'left_knee', 'right_knee', 'left_hip', 'right_hip', 'left_ankle', 'right_ankle',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist',
            'neck', 'left_spine', 'right_spine', 'left_hip_torso', 'right_hip_torso',
            'left_shoulder_torso', 'right_shoulder_torso'
        ]
        
    def extract_features(self, df):
        """Extract features from raw joint angles"""
        features = []
        for joint in self.joints:
            if joint not in df.columns:
                continue
            joint_data = df[joint].values
            features.extend([
                joint_data,
                np.gradient(joint_data),
                np.gradient(np.gradient(joint_data))
            ])
        return np.array(features).T
    
    def create_sequences(self, features, labels):
        """Create overlapping sequences for training"""
        step = int(self.window_size * (1 - self.overlap))
        sequences = []
        sequence_labels = []
        
        for i in range(0, len(features) - self.window_size + 1, step):
            sequences.append(features[i:i + self.window_size])
            sequence_labels.append(labels[i + self.window_size - 1])
            
        return np.array(sequences), np.array(sequence_labels)
    
    def process_workout_data(self, data_dir):
        """Process all workout data files in directory"""
        all_sequences = []
        all_labels = []
        
        for filename in os.listdir(data_dir):
            if filename.endswith('.csv'):
                df = pd.read_csv(os.path.join(data_dir, filename))
                features = self.extract_features(df)
                fatigue_labels = df['fatigue_label'].values / 10.0
                sequences, labels = self.create_sequences(features, fatigue_labels)
                all_sequences.extend(sequences)
                all_labels.extend(labels)
        
        shaped_sequences = np.vstack(all_sequences)
        self.scaler.fit(shaped_sequences)
        scaled_sequences = [self.scaler.transform(seq) for seq in all_sequences]
        
        return np.array(scaled_sequences), np.array(all_labels)

class FatigueModelTrainer:
    def __init__(self, model, learning_rate=0.0001, batch_size=32):
        self.model = model
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
    def train(self, train_sequences, train_labels, val_sequences, val_labels, epochs=100, early_stopping=10):
        train_dataset = WorkoutDataset(train_sequences, train_labels)
        val_dataset = WorkoutDataset(val_sequences, val_labels)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)
        
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()
        
        history = {
            'train_loss': [],
            'val_loss': [],
            'best_val_loss': float('inf'),
            'epochs_no_improve': 0
        }
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            for sequences, labels in train_loader:
                sequences = sequences.to(self.device)
                labels = labels.to(self.device)
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
                    sequences = sequences.to(self.device)
                    labels = labels.to(self.device)
                    outputs = self.model(sequences)
                    val_loss += criterion(outputs, labels.unsqueeze(1)).item()
            
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            
            history['train_loss'].append(float(avg_train_loss))  # Convert to Python float
            history['val_loss'].append(float(avg_val_loss))      # Convert to Python float
            
            if avg_val_loss < history['best_val_loss']:
                history['best_val_loss'] = float(avg_val_loss)   # Convert to Python float
                history['epochs_no_improve'] = 0
                self.save_model('best_model.pt')
            else:
                history['epochs_no_improve'] += 1
                
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], '
                      f'Train Loss: {avg_train_loss:.4f}, '
                      f'Val Loss: {avg_val_loss:.4f}')
                
            if history['epochs_no_improve'] >= early_stopping:
                print(f'Early stopping at epoch {epoch+1}')
                break
        
        return history
    
    def evaluate(self, test_sequences, test_labels):
        self.model.eval()
        test_dataset = WorkoutDataset(test_sequences, test_labels)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size)
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for sequences, labels in test_loader:
                sequences = sequences.to(self.device)
                outputs = self.model(sequences)
                all_preds.extend(outputs.cpu().numpy().flatten())
                all_labels.extend(labels.numpy().flatten())
        
        metrics = {
            'mse': float(mean_squared_error(all_labels, all_preds)),   # Convert to Python float
            'rmse': float(np.sqrt(mean_squared_error(all_labels, all_preds))),
            'mae': float(mean_absolute_error(all_labels, all_preds)),
            'r2': float(r2_score(all_labels, all_preds))
        }
        
        plt.figure(figsize=(10, 6))
        plt.scatter(all_labels, all_preds, alpha=0.5)
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlabel('Actual Fatigue')
        plt.ylabel('Predicted Fatigue')
        plt.title('Fatigue Prediction: Actual vs Predicted')
        plt.savefig('evaluation_plot.png')
        
        return metrics, all_preds, all_labels
    
    def save_model(self, filename):
        model_info = {
            'state_dict': self.model.state_dict(),
            'input_size': self.model.lstm.input_size,
            'hidden_size': self.model.hidden_size,
            'num_layers': self.model.num_layers,
            'date_trained': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        torch.save(model_info, filename)
        
    def export_to_mobile(self, filename='mobile_model.pt'):
        example_input = torch.randn(1, self.model.get_input_size()).unsqueeze(0)
        traced_model = torch.jit.trace(self.model, example_input)
        torch.jit.save(traced_model, filename)
        print(f"Model exported for mobile deployment at {filename}")
        
        quantized_model = torch.quantization.quantize_dynamic(
            self.model, {nn.LSTM, nn.Linear}, dtype=torch.qint8
        )
        traced_quantized = torch.jit.trace(quantized_model, example_input)
        torch.jit.save(traced_quantized, 'quantized_' + filename)
        print(f"Quantized model exported at quantized_{filename}")

def train_fatigue_model(data_dir, save_dir='models'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    processor = FatigueDataProcessor()
    sequences, labels = processor.process_workout_data(data_dir)
    
    train_seq, temp_seq, train_labels, temp_labels = train_test_split(
        sequences, labels, test_size=0.3, random_state=42
    )
    
    val_seq, test_seq, val_labels, test_labels = train_test_split(
        temp_seq, temp_labels, test_size=0.5, random_state=42
    )
    
    input_feature_count = train_seq.shape[2]
    model = FatigueLSTM(
        input_size=input_feature_count,
        hidden_size=64,
        num_layers=2,
        output_size=1
    )
    
    trainer = FatigueModelTrainer(model)
    history = trainer.train(train_seq, train_labels, val_seq, val_labels, epochs=100)
    
    metrics, predictions, actual = trainer.evaluate(test_seq, test_labels)
    print("Model Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric.upper()}: {value:.4f}")
    
    trainer.export_to_mobile()
    
    results = {
        'training_history': history,
        'evaluation_metrics': metrics,
        'model_config': {
            'input_size': input_feature_count,
            'hidden_size': 64,
            'num_layers': 2
        }
    }
    
    with open(os.path.join(save_dir, 'model_results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    torch.save(processor.scaler, os.path.join(save_dir, 'scaler.pt'))
    
    print("Training and evaluation completed. Model and artifacts saved in", save_dir)
    return trainer, processor, metrics

if __name__ == '__main__':
    train_fatigue_model('session_data')