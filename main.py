import json
import argparse
import os
from trainer import train, run_test

def load_json(settings_path):
    with open(settings_path) as data_file:
        param = json.load(data_file)
    return param

def setup_parser():
    parser = argparse.ArgumentParser(description='Reproduce of multiple continual learning algorthms.')
    parser.add_argument('--config', type=str, default='./exps/cic_iot23_der.json',
                        help='Json file of settings.')
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'test'],
                        help=(
                            'train: chay training day du va luu checkpoint sau moi task. '
                            'test : chi load checkpoint da luu va chay evaluation.'
                        ))
    parser.add_argument('--test_checkpoint_dir', type=str, default='',
                        help='Duong dan thu muc chua checkpoint khi chay --mode test. '
                             'Vi du: ./logs/der/cic_iot23/23-04-26_21-16_seed42_cnn1d')
    parser.add_argument('--debug', action='store_true',
                        help='Che do debug: giam so epoch xuong 2 de test nhanh.')
    parser.add_argument('--resume', type=str, default='',
                        help='Duong dan den checkpoint (.pth) de tiep tuc training.')
    parser.add_argument('--memory_size', type=int, default=None,
                        help='Tong so luong mau luu trong bo nho dem (Exemplar memory).')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size cho training.')
    parser.add_argument('--init_epoch', type=int, default=None,
                        help='So epoch cho task dau tien.')
    parser.add_argument('--epochs', type=int, default=None,
                        help='So epoch cho cac task sau.')
    return parser

def main():
    args_cli = setup_parser().parse_args()
    param = load_json(args_cli.config)
    
    # Merge logic: JSON config acts as base, CLI args override it
    config = load_json(args_cli.config)
    
    # Remove None values from CLI args so they don't overwrite JSON defaults unnecessarily
    cli_args = {k: v for k, v in vars(args_cli).items() if v is not None}
    
    # Final args: JSON base + CLI overrides
    args = config
    args.update(cli_args)

    if args.get("debug"):
        print("[DEBUG] Che do debug: set init_epoch=2, epochs=2")
        args["init_epoch"] = 2
        args["epochs"] = 2

    if args.get("mode", "train") == "test":
        # Giai doan Test: chi load checkpoint va chay evaluation
        if not args.get("run_dir") and not args.get("test_checkpoint_dir"):
            print("[ERROR] --mode test yeu cau 'run_dir' hoac 'test_checkpoint_dir' trong config.")
            return
        run_test(args)
    else:
        # Giai doan Train
        train(args)

if __name__ == '__main__':
    main()
