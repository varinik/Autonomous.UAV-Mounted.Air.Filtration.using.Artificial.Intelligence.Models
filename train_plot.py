import argparse
import os
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt  # ✅ For graphing
from mpl_toolkits.mplot3d import Axes3D  # ✅ For 3D plotting
import helper
import lstm  # Import once, avoid duplication

# Enable interactive mode for plotting
plt.ion()

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
    path = f'models/Individual/{args.name}/good_models/00047.pth'
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    
    criterion = nn.L1Loss()
    total_loss = 0
    
    # Create 3D plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    batch_colors = plt.cm.viridis(np.linspace(0, 1, len(test_dl)))  # Unique colors per batch
    
    with torch.no_grad():
        for batch_id, batch in enumerate(test_dl):
            inp = batch['src'][:, 1:, 2:5].to(device)  # Observed trajectory
            target = batch['trg'][:, -1, 2:5].to(device)  # Ground truth
            
            pred = model(inp)  # Model predictions
            
            # Debugging prints
            print(f"Batch {batch_id}: Pred Shape: {pred.shape}, Target Shape: {target.shape}")
            
            if pred.shape[0] == 0 or target.shape[0] == 0:
                print("⚠️ Empty batch, skipping plot.")
                continue  # Skip empty batches
            
            loss = F.pairwise_distance(
                pred[:, :3].contiguous().view(-1, 3),
                target[:, :3].contiguous().view(-1, 3)
            ).mean()
            
            total_loss += loss.item()
            
            # Extract real & predicted coordinates
            real_x = target[:, 0].cpu().numpy()  # X from target
            real_y = target[:, 1].cpu().numpy()  # Assuming Y is missing, use zeros
            real_z = target[:, 2].cpu().numpy()  # Z from target
            
            pred_x = pred[:, 0].cpu().numpy()  # 2D predicted X
            pred_y = pred[:, 1].cpu().numpy()  # Fake Y (since we only have 2D prediction)
            pred_z = pred[:, 2].cpu().numpy()  # Fake Z (aligns visually)
            
            # Plot each trajectory
            color = batch_colors[batch_id % len(batch_colors)]  # Cycle through colors
            for i in range(min(len(real_x), 10)):  # Only plot first 10 per batch
                ax.plot(real_x[i], real_y[i], real_z[i], '-', color=color, label="Real" if batch_id == 0 and i == 0 else "", alpha=0.7)  # Real
                ax.plot(pred_x[i], pred_y[i], pred_z[i], '--', color=color, label="Predicted" if batch_id == 0 and i == 0 else "", alpha=0.7)  # Predicted

    # Label axes & save plot
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate (Fixed)")
    ax.set_zlabel("Z Coordinate")
    ax.set_title("3D Trajectory Prediction - All Batches")
    ax.legend()
    plt.savefig("trajectory_3D_all_batches.png")  # Save as an image
    plt.show()  # Display plot
    x = input()
    
    avg_loss = total_loss / len(test_dl)
    print(f"Test Loss: {avg_loss:.4f}")
    
if __name__ == '__main__':
    main()
