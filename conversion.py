from torch.autograd import Variable
import torch.onnx
import torch
import helper  # Ensure helper.py exists and contains create_dataset function
import lstm  # Ensure lstm.py defines the LSTM model
import argparse
import os

# Argument parsing setup
parser = argparse.ArgumentParser(description='Train the LSTM model')
parser.add_argument('--dataset_folder', type=str, default='datasets')
parser.add_argument('--dataset_name', type=str, default='zara1')
parser.add_argument('--obs', type=int, default=8)
parser.add_argument('--preds', type=int, default=12)
parser.add_argument('--batch_size', type=int, default=70)
parser.add_argument('--cpu', action='store_true')
parser.add_argument('--delim', type=str, default='\t')
parser.add_argument('--name', type=str, default="zara1")
parser.add_argument('--max_epoch', type=int, default=300)
parser.add_argument('--save_step', type=int, default=1)

args = parser.parse_args()

# Create dataset
test_dataset, _ = helper.create_dataset(
    args.dataset_folder, args.dataset_name, 0, args.obs, args.preds,
    delim=args.delim, train=False, eval=True
)
test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

# Device setup
device = torch.device("cpu" if args.cpu else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load model
model = lstm.LSTM(3, 128, 3, 3).to(device)

# Ensure model path exists
model_path = f'models/Individual/{args.name}/best_test.pth'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

model.load_state_dict(torch.load(model_path, map_location=device))

# Prepare test batch
test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
batch = next(iter(test_dl))  # Get first batch safely

# Ensure batch is a dictionary and contains 'src' key
if isinstance(batch, dict) and 'src' in batch:
    inp = batch['src'][:, 1:, 2:5].to(device)
else:
    raise ValueError("Batch format is incorrect or missing 'src' key")

# Export model to ONNX
print(inp)
# onnx_path = 'lstm_model.onnx'
# torch.onnx.export(model, inp, onnx_path, export_params=True, opset_version=11)

# print(f"ONNX model saved at: {onnx_path}")
