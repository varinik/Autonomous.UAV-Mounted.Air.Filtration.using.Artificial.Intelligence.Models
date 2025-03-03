import argparse
import os
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
import helper
import lstm  # Import once, avoid duplication

def main():
    # Argument parser setup
    parser = argparse.ArgumentParser(description='Train the LSTM model')
    parser.add_argument('--dataset_folder', type=str, default='datasets')
    parser.add_argument('--dataset_name', type=str, default='zara1')
    parser.add_argument('--obs', type=int, default=8)
    parser.add_argument('--preds', type=int, default=12)
    parser.add_argument('--emb_size', type=int, default=512)
    parser.add_argument('--heads', type=int, default=8)
    parser.add_argument('--layers', type=int, default=6)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--val_size', type=int, default=0)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--max_epoch', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=70)
    parser.add_argument('--validation_epoch_start', type=int, default=30)
    parser.add_argument('--resume_train', action='store_true')
    parser.add_argument('--delim', type=str, default='\t')
    parser.add_argument('--name', type=str, default="zara1")
    parser.add_argument('--factor', type=float, default=1.0)
    parser.add_argument('--save_step', type=int, default=1)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--evaluate', type=bool, default=True)
    parser.add_argument('--model_pth', type=str)

    args = parser.parse_args()

    # Ensure directories exist
    os.makedirs('models/Individual', exist_ok=True)
    os.makedirs(f'models/Individual/{args.name}', exist_ok=True)
    os.makedirs('output/Individual', exist_ok=True)
    os.makedirs(f'output/Individual/{args.name}', exist_ok=True)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")

    # Load datasets
    train_dataset, _ = helper.create_dataset(
        args.dataset_folder, args.dataset_name, 0, args.obs, args.preds, 
        delim=args.delim, train=True, verbose=args.verbose
    )
    val_dataset, _ = helper.create_dataset(
        args.dataset_folder, args.dataset_name, 0, args.obs, args.preds, 
        delim=args.delim, train=False, verbose=args.verbose
    )
    test_dataset, _ = helper.create_dataset(
        args.dataset_folder, args.dataset_name, 0, args.obs, args.preds, 
        delim=args.delim, train=False, eval=True, verbose=args.verbose
    )

    # Initialize LSTM model
    model = lstm.LSTM(3, 128, 3, 3).to(device)
    path = f'models/Individual/{args.name}/00299.pth'
    model.load_state_dict(torch.load(path, map_location=device))
    test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    batch = []
    for id_b, batch in enumerate(test_dl):
        batch = batch
        break
    inp = batch['src'][:, 1:, 2:5].to(device)
    
    onnx_path = 'transformer_model.onnx'
    torch.onnx.export(model, inp, onnx_path, export_params=True, opset_version=11)

    

if __name__ == '__main__':
    main()
