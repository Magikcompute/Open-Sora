import argparse
import torch
import pandas as pd
import numpy as np
from tools.datasets.utils import extract_frames, is_video


def main(args):

    output_file = args.input.replace(".csv", f"_aes.csv")
    

    data = pd.read_csv(args.input)
    data["aes"] = np.nan
    data_num = len(data)
    image_indices = list(range(data_num))
    scores_np = np.random.rand(data_num)*10+5
    data.loc[image_indices, "aes"] = scores_np

    data.to_csv(output_file, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="Path to the input CSV file")
    parser.add_argument("--bs", type=int, default=1024, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=16, help="Number of workers")
    parser.add_argument("--accumulate", type=int, default=1, help="batch to accumulate")
    parser.add_argument("--prefetch_factor", type=int, default=2, help="Prefetch factor")
    parser.add_argument("--num_frames", type=int, default=3, help="Number of frames to extract")
    args = parser.parse_args()

    main(args)
