import argparse
import os

#heerer
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
import helper
import lstm
import pandas as pd
from sklearn.preprocessing import StandardScaler

def process_input(dataset_folder, dataset_name, delim='\t'):
    data_path = os.path.join(dataset_folder, dataset_name)
    
    all_files = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.txt')]
    all_files = os.listdir(os.path.join(dataset_folder,dataset_name, "train"))
    data_path=os.path.join(dataset_folder,dataset_name, "train")
    
    trajectories = {}
    
    for file in all_files:
        df = pd.read_csv(os.path.join(data_path, file), delimiter=delim, names=['frame', 'ped', 'x', 'y', 'z'])
        df.sort_values(by=['ped', 'frame'], inplace=True)
        
        for ped_id, group in df.groupby('ped'):
            if ped_id not in trajectories:
                trajectories[ped_id] = []
            trajectories[ped_id].append(group[['x', 'y', 'z']].values)
    
    # Normalize trajectories
    processed_trajectories = {
        k: torch.tensor()
        for k, v in trajectories.items()
    }
    return processed_trajectories

def autoregressive_prediction(model, input_seq, steps, device):
    model.eval()
    predicted_seq = []
    current_input = input_seq.to(device)
    
    with torch.no_grad():
        for _ in range(steps):
            pred = model(current_input.unsqueeze(0)).squeeze(0)
            predicted_seq.append(pred.cpu().numpy())
            current_input = torch.cat([current_input[1:], pred.unsqueeze(0)], dim=0)
    
    return np.array(predicted_seq)

class EuclideanLoss(nn.Module):
    def __init__(self):
        super(EuclideanLoss, self).__init__()

    def forward(self, predicted, target):
        dist = torch.sqrt(torch.sum((predicted - target) ** 2, dim=1))
        return torch.mean(dist)

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out)
        return torch.tanh(out)

def main():
    parser = argparse.ArgumentParser(description='Train the LSTM model')
    parser.add_argument('--dataset_folder', type=str, default='datasets')
    parser.add_argument('--dataset_name', type=str, default='zara1')
    parser.add_argument('--delim', type=str, default='\t')
    parser.add_argument('--max_epoch', type=int, default=400)
    parser.add_argument('--batch_size', type=int, default=70)
    parser.add_argument('--learning_rate', type=float, default=0.0003)  # Reduced learning rate
    args = parser.parse_args()

    trajectories = process_input(args.dataset_folder, args.dataset_name, args.delim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = lstm.LSTM(3, 256, 3, 3).to(device)
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    criterion = EuclideanLoss()

    for epoch in range(args.max_epoch):
        model.train()
        total_loss = 0
        
        for ped_id, trajectory in trajectories.items():
            optimizer.zero_grad()
            input_seq = trajectory[:-1].to(device)
            target_seq = trajectory[1:].to(device)
            
            pred = model(input_seq.unsqueeze(0)).squeeze(0)
            loss = criterion(pred, target_seq)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{args.max_epoch}, Loss: {total_loss / len(trajectories)}")
    
    test_ped_id = list(trajectories.keys())[0]
    test_input = trajectories[test_ped_id][:10]  # Use first 10 points as seed
    predicted_sequence = autoregressive_prediction(model, test_input, 20, device)
    print("Predicted sequence:", predicted_sequence)
    
if __name__ == '__main__':
    main()
