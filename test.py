import argparse
import os
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import helper
import lstm  # Import once, avoid duplication

def main():
    # Argument parser setup
    parser = argparse.ArgumentParser(description='Test the LSTM model')
    parser.add_argument('--dataset_folder', type=str, default='datasets')
    parser.add_argument('--dataset_name', type=str, default='zara1')
    parser.add_argument('--obs', type=int, default=8)
    parser.add_argument('--preds', type=int, default=12)
    parser.add_argument('--batch_size', type=int, default=70)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--delim', type=str, default='\t')
    parser.add_argument('--name', type=str, default="zara1")
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")
    
    # Load test dataset
    test_dataset, _ = helper.create_dataset(
        args.dataset_folder, args.dataset_name, 0, args.obs, args.preds, 
        delim=args.delim, train=False, eval=True
    )
    test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # Initialize model
    model = lstm.LSTM(3, 128, 3, 3).to(device)
    path = f'models/Individual/zara1/00072.pth'
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    
    criterion = nn.L1Loss()
    total_loss = 0
    
    with torch.no_grad():
        for batch in test_dl:
            inp = batch['src'][:, 1:, 2:5].to(device)
            target = batch['trg'][:, 0, 2:5].to(device)
            
            pred = model(inp)
            print(inp)
            print(pred)
            print(target)
            x = input()
            loss = F.pairwise_distance(
                pred[:, :3].contiguous().view(-1, 3),
                target[:, :3].contiguous().view(-1, 3)
            ).mean()
            
            total_loss += loss.item()
    
    avg_loss = total_loss / len(test_dl)
    print(f"Test Loss: {avg_loss:.4f}")
    
if __name__ == '__main__':
    main()
