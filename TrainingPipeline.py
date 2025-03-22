import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
from datetime import datetime
import json

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

class FatigueDataProcessor:
    def __init__(self, window_size=30, overlap=0.5):
        self.window_size = window_size
        self.overlap = overlap
        self.scaler = StandardScaler()
        
    def extract_features(self, df):
        """Extract features from raw joint angles"""
        features = []
        for joint in ['left_knee', 'right_knee', 'left_hip', 'right_hip', 'left_ankle', 'right_ankle', 'left_shoulder'
                      'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist']:
            joint_data = df[joint].values
            features.extend([
                joint_data,
                # Velocity of the joint
                np.gradient(joint_data), 
                # Acceleration of the joint
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
            # Use the last label in the sequence as the target
            sequence_labels.append(labels[i + self.window_size - 1])
            
        return np.array(sequences), np.array(sequence_labels)
    
    def process_workout_data(self, data_dir):
        """Process all workout data files in directory"""
        all_sequences = []
        all_labels = []
        
        for filename in os.listdir(data_dir):
            if filename.endswith('.csv'):
                df = pd.read_csv(os.path.join(data_dir, filename))
                
                # Extract features
                features = self.extract_features(df)
                
                # Calculate fatigue labels (example based on joint variability)
                # Replace with actual fatigue measurements
                joint_variability = np.std(features, axis=1)
                fatigue_labels = self.calculate_fatigue_labels(joint_variability)
                
                # Create sequences
                sequences, labels = self.create_sequences(features, fatigue_labels)
                
                all_sequences.extend(sequences)
                all_labels.extend(labels)
        
        # Scale features
        shaped_sequences = np.vstack(all_sequences)
        self.scaler.fit(shaped_sequences)
        scaled_sequences = [self.scaler.transform(seq) for seq in all_sequences]
        
        return np.array(scaled_sequences), np.array(all_labels)
    
    def calculate_fatigue_labels(self, joint_variability):
        """Calculate fatigue labels based on joint movement patterns"""
        # This is a simplified example - in practice, you'd want to use actual fatigue measurements
        normalized_variability = joint_variability / np.max(joint_variability)
        smoothed_fatigue = np.convolve(normalized_variability, np.ones(10)/10, mode='same')
        return np.clip(smoothed_fatigue, 0, 1)

class FatigueModelTrainer:
    def __init__(self, model, learning_rate=0.001, batch_size=32):
        self.model = model
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
    def train(self, train_sequences, train_labels, val_sequences, val_labels, epochs=100):
        """Train the fatigue prediction model"""
        # Create data loaders
        train_dataset = WorkoutDataset(train_sequences, train_labels)
        val_dataset = WorkoutDataset(val_sequences, val_labels)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)
        
        # Initialize optimiser and loss function
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'best_val_loss': float('inf')
        }
        
        for epoch in range(epochs):
            # Training
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
            
            # Validation
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for sequences, labels in val_loader:
                    sequences = sequences.to(self.device)
                    labels = labels.to(self.device)
                    outputs = self.model(sequences)
                    val_loss += criterion(outputs, labels.unsqueeze(1)).item()
            
            # Calculate average losses
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            
            # Update history
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            
            # Save the best performing model
            if avg_val_loss < history['best_val_loss']:
                history['best_val_loss'] = avg_val_loss
                self.save_model('best_model.pth')
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], '
                      f'Train Loss: {avg_train_loss:.4f}, '
                      f'Val Loss: {avg_val_loss:.4f}')
        
        return history
    
    def save_model(self, filename):
        """Save the best model with the training parameters"""
        model_info = {
            'state_dict': self.model.state_dict(),
            'input_size': self.model.lstm.input_size,
            'hidden_size': self.model.hidden_size,
            'num_layers': self.model.num_layers,
            'date_trained': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        torch.save(model_info, filename)

def train_fatigue_model(data_dir, save_dir='models'):
    """Main training function"""
    # Create save directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # Initialize data processor
    processor = FatigueDataProcessor()
    # Process workout data
    sequences, labels = processor.process_workout_data(data_dir)
    # Split the data into training and validation
    train_seq, val_seq, train_labels, val_labels = train_test_split(
        sequences, labels, test_size=0.2, random_state=42
    )
    # Initialize model
    model = FatigueLSTM(
        # 10 joints * 3 features (position, velocity, acceleration)
        input_size=30,  
        hidden_size=64,
        num_layers=2,
        output_size=1
    )
    # Initialize trainer
    trainer = FatigueModelTrainer(model)
    # Train model
    history = trainer.train(train_seq, train_labels, val_seq, val_labels)
    # Save training history
    with open(os.path.join(save_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f)
    # Save scaler
    torch.save(processor.scaler, os.path.join(save_dir, 'scaler.pkl'))
    print("Training completed. Model and artifacts saved in", save_dir)

if __name__ == '__main__':
    # Example usage
    train_fatigue_model('session_data')