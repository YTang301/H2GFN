import argparse
from predict_mda import case_study1
import sys
import torch
import numpy as np
import random
import os
import dgl

class Logger(object):
    def __init__(self, filename="log_output.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        pass

import time
log_name = f"log_{time.strftime('%m%d_%H%M%S')}.txt"
sys.stdout = Logger(log_name)


def get_args():
    parser = argparse.ArgumentParser(description='Model for MDA Prediction')
    parser.add_argument('--all_epochs', type=int, default=400, help='Training epochs')
    parser.add_argument('--lr', type=float, default=0.0007, help='Learning rate')
    parser.add_argument('--main_feature_dim', type=int, default=16, help='Input node dimension')
    parser.add_argument('--seed', type=int, default=11, help='random seed')
    parser.add_argument('--gamma', type=float, default=0.3, help='Similarity threshold')
    parser.add_argument('--gcn2_layers', type=int, default=45, help='GCNII layers')
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--theta', type=int, default=2)
    parser.add_argument('--pe_dim', type=int, default=15, help='Positional encoding dimension')
    parser.add_argument('--hops', type=int, default=5, help='Number of hops')
    parser.add_argument('--num_heads', type=int, default=12, help='Number of attention heads')
    parser.add_argument('--ffn_dim', type=int, default=64, help='FFN hidden dimension')
    parser.add_argument('--hidden_dim', type=int, default=32, help='Transformer hidden dimension')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--gcn1_innum', type=int, default=256, help='GCN1 input dimension')
    parser.add_argument('--gcn_layer1_units', type=int, default=32)
    parser.add_argument('--gcn_layer2_units', type=int, default=32)
    parser.add_argument('--rwr_c', type=float, default=0.3, help='Restart probability for RWR')
    parser.add_argument('--gcn_threshold', type=float, default=0.75, help='Final output feature dimension')
    parser.add_argument('--r_mask', type=float, default=0.8)
    parser.add_argument('--r_replace', type=float, default=0.5)
    parser.add_argument('--r_drop', type=float, default=0.25)
    parser.add_argument('--negative_slope', type=float, default=0.35)
    parser.add_argument('--attn_drop', type=float, default=0.2)
    parser.add_argument('--in_drop', type=float, default=0.3)
    parser.add_argument('--neighbor', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=800)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument('--maed_layers', type=int, default=8)
    parser.add_argument('--maed_heads', type=int, default=8)
    parser.add_argument("--maed_out", type=int, default=1)
    parser.add_argument('--maed_hid', type=int, default=16)
    parser.add_argument("--residual", action="store_true", default=False)
    parser.add_argument("--norm", type=str, default=None)
    parser.add_argument("--activation", type=str, default='prelu')

    return parser.parse_args()


def seed_everything(seed=11):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    dgl.seed(seed)
    dgl.random.seed(seed)

def main():
    args = get_args()
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    seed_everything(args.seed)
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Current GPU: {torch.cuda.get_device_name(0)}")

    root0 = '/mnt/rgcgt1022/H2GFN/microRNA-disease/HMDD_V1'
    case_study1(root0,args, "case_study_1.1.xlsx")


if __name__ == "__main__":
    sys.__stdout__.write("--- start ---\n")
    print("1")
    main()