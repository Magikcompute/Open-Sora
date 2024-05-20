import argparse
import pandas as pd
import torch
import nltk
import cv2
import random
from random_words import RandomWords
from random_words import LoremIpsum
import csv


@torch.inference_mode()
def main(args):
    
    data = pd.read_csv(args.input)
    video_files = []
    outputs = []
    video_lengths = []

    rw = RandomWords()

    key_to_vs = {}
    keys = []
    exclude_keys = ['path', 'num_frames']
    for key in data:
        if key not in exclude_keys:
            key_to_vs[key] = []
            keys.append(key)
            for ele in data[key]:
                key_to_vs[key].append(ele)

    for video_path in data['path']:
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_files.append(video_path)
        random_strs = rw.random_words(count=10)
        random_str = ' '.join(random_strs)
        outputs.append(random_str)
        video_lengths.append(frame_count)

    dp_file = open(args.output, "w")
    dp_writer = csv.writer(dp_file)
    dp_writer.writerow(["path", "text", "num_frames"]+keys)

    var_list = [key_to_vs[key] for key in keys]
    result = list(zip(video_files, outputs, video_lengths, *var_list))
    for t in result:
        dp_writer.writerow(t)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="Path to the input CSV file")
    parser.add_argument("--output", type=str, default="video-f1-detail-3ex")

    args = parser.parse_args()
    main(args)
