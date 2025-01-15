# utils/parser.py
import os
import time
import argparse
import yaml

class Parser:
    def __init__(self, default_config_path='configs/config.yaml'):
        self.default_config_path = default_config_path
        self.num_class_dict = {'01': 6, '02': 3, '03': 6, '04': 5, '05': 7, '06': 4, '07': 4, 
                               '08': 5, '09': 4, '10': 5, '11': 5, '12': 5, '13': 4, '14': 4,}

    def parse_args(self):
        parser = argparse.ArgumentParser(description="Training script")
        parser.add_argument('--config', type=str, default=self.default_config_path, help='Path to the config file')
        parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training')
        parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for training')
        parser.add_argument('--epochs', type=int, default=100, help='Number of epochs for training')
        parser.add_argument('--save_path', type=str, default='exps', help='Path to save the experiment results')
        parser.add_argument('--checkpoint_path', type=str, default='checkpoints', help='Path to save checkpoints')
        parser.add_argument('--backbone', type=str, choices=['I3D', 'VST'], default='I3D', help='Backbone model which rgb and flow will use')
        parser.add_argument('--dataset', type=str, choices=['Myositis', 'MyositisClassify', 'MyositisTotal'], default='Myositis', help='Dataset to use')
        parser.add_argument('--exp_name', type=str, default='default', help='Name of the experiment')
        parser.add_argument('--log_interval', type=int, default=10, help='Interval for logging')
        parser.add_argument('--save_interval', type=int, default=5, help='Interval for saving checkpoints')
        parser.add_argument('--num_classes', type=int, default=10, help='Number of classes')
        parser.add_argument('--mode', type=str, default='train', choices=['train', 'evaluate'], help="Mode to run the script: 'train' or 'evaluate'")
        parser.add_argument('--action', type=str, default='12', help='Action to train or evaluate')
        parser.add_argument('--no_etf', action="store_true", help='Whether to use ETF')
        parser.add_argument('--save_feat', action="store_true", help='Whether to save intermediate features')
        parser.add_argument('--pretrain_feats', action="store_true", help='Whether to use pretrain Video and Optic flow features')
        parser.add_argument('--miss_modal', type=str, default='', help='Which modal to miss')
        parser.add_argument('--miss_modal_rate', type=float, default=0, help='Rate of missing modal')
        parser.add_argument('--miss_label_rate', type=float, default=0, help='Rate of missing label')

        args = parser.parse_args()
        return args

    def load_config(self, config_path):
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config

    def merge_config_and_args(self, args, config):
        # reverse initial config
        args_dict = vars(args).copy()

        for key, value in config.items():
            setattr(args, key, value)

        # save num_classes, backbone setting
        args.num_classes = args_dict['num_classes']
        # assert args.num_classes == num_classes
        if args.num_classes != self.num_class_dict[args.action]:
            print(f"Action {args.action} should have {self.num_class_dict[args.action]} classes, but got {args.num_classes} classes, which will be corrected.")
            args.num_classes = self.num_class_dict[args.action]
        args.epochs = args_dict['epochs']
        args.backbone = args_dict['backbone']
        args.no_etf = args_dict['no_etf']
        if args_dict['exp_name'] == 'default':
            args.exp_name = time.strftime('%m%d_%H%M%S', time.localtime()) + f"_Act{args.action}"
        if args_dict['dataset'] == 'Myositis':
            args.train_split_path = os.path.join(args.split_root_path, args.action, "train_split.pkl")
            args.test_split_path = os.path.join(args.split_root_path, args.action, "test_split.pkl") 
            # args.train_split_path = os.path.join(args.split_root_path, f"train_split_{args.action}.pkl")
            # args.test_split_path = os.path.join(args.split_root_path, f"test_split_{args.action}.pkl")
        return args

    def get_args(self):
        args = self.parse_args()
        config = self.load_config(args.config)
        args = self.merge_config_and_args(args, config)
        return args
