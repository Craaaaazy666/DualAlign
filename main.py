import argparse
from utils import parser
import os
import time

from scripts.train import train
from scripts.evaluate import evaluate

def main():
    args = parser.Parser().get_args()
    # print(args.data_paths)

    if not os.path.exists(os.path.join(args.save_path, args.exp_name)):
        os.makedirs(os.path.join(args.save_path, args.exp_name))

    if args.mode == 'train':
        train(args)
    elif args.mode == 'evaluate':
        evaluate(args)

if __name__ == "__main__":
    main()
