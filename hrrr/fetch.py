import warnings
warnings.filterwarnings("ignore")

import argparse
import logging
from pathlib import Path
import pickle
import random

import torch
from tqdm import tqdm

import sys
sys.path.append(Path(__file__).parent.parent.as_posix())
import get_data

# -----------------------------------------------------------------------------
# Configuration & setup
# -----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Fetch and cache HRRR data")
    p.add_argument(
        "--levels", nargs="+", default=["500 mb"],
        help="pressure levels to fetch (e.g. '500 mb')"
    )
    p.add_argument(
        "--dates", nargs="+", default=["2024-09-18"],
        help="list of dates (YYYY-MM-DD) to fetch"
    )
    p.add_argument(
        "--hours", type=int, default=4,
        help="forecast lead time in hours"
    )
    p.add_argument(
        "--out", type=Path, default=Path("Datas/Datas.pkl"),
        help="output file (will be overwritten)"
    )
    return p.parse_args()

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

# -----------------------------------------------------------------------------
# Main data fetch + save
# -----------------------------------------------------------------------------
def main():
    args = parse_args()
    setup_logging()

    # ensure output directory exists
    args.out.parent.mkdir(parents=True, exist_ok=True)

    params_list = [
        {"Date": d, "Hours": args.hours, "Level": lvl}
        for d in args.dates
        for lvl in args.levels
    ]

    datas = []
    for params in tqdm(params_list, desc="Fetching HRRR data"):
        try:
            datas.append(get_data.hrrr(**params))
        except Exception as e:
            logging.error(f"Failed for {params}: {e}")

    # overwrite existing file
    with open(args.out, "wb") as f:
        pickle.dump(datas, f)

    logging.info(f"Saved {len(datas)} datasets to {args.out}")

if __name__ == "__main__":
    main()
