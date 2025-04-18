import warnings
warnings.filterwarnings("ignore")

import argparse
import logging
import pickle
import random
import time
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm

import sys
sys.path.append(Path(__file__).parent.as_posix())
from main import Model

# -----------------------------------------------------------------------------
# Configuration & setup
# -----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Fit Transport models on HRRR data")
    p.add_argument(
        "--pre", type=Path, default=Path("Data/Data.pkl"),
        help="preprocessed data file"
    )
    p.add_argument(
        "--out", type=Path, default=Path("Data/Transport.pkl"),
        help="where to save fitted Transport objects"
    )
    p.add_argument(
        "--epochs", type=int, default=50,
        help="number of MLE training epochs"
    )
    p.add_argument(
        "--subsample", type=int, default=2000,
        help="subsample size for MLE"
    )
    p.add_argument(
        "--neighbors", type=int, default=1,
        help="number of neighbors for MLE"
    )
    p.add_argument(
        "--seed", type=int, default=23,
        help="random seed"
    )
    return p.parse_args()

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# -----------------------------------------------------------------------------
# Main model fit + save
# -----------------------------------------------------------------------------
def main():
    args = parse_args()
    setup_logging()
    set_seed(args.seed)

    args.out.parent.mkdir(parents=True, exist_ok=True)

    with open(args.pre, "rb") as f:
        datas = pickle.load(f)

    transports = []
    for data in tqdm(datas, desc="Fitting Transport models"):
        model = Model.Transport(data)
        start = time.time()
        model.TrainMLE(
            Epochs=args.epochs,
            SubSampleSize=args.subsample,
            Neighbors=args.neighbors
        )
        model.Time = time.time() - start
        transports.append(model)

    with open(args.out, "wb") as f:
        pickle.dump(transports, f)

    logging.info(f"Saved {len(transports)} Transports to {args.out}")

if __name__ == "__main__":
    main()
