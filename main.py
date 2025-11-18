import argparse
from train import train_model
from cli import run_background

parser = argparse.ArgumentParser("OS Anomaly Detector")
parser.add_argument("--train", action="store_true")
parser.add_argument("--run", action="store_true")

args = parser.parse_args()

if args.train:
    train_model()

if args.run:
    run_background()