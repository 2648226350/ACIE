import json
import argparse
from trainer import train


def main(use_nlm, exp):
    args = setup_parser(exp).parse_args()
    param = load_json(args.config)
    args = vars(args)  # Converting argparse Namespace to a dict.
    args.update(param)  # Add parameters from json

    train(args, use_nlm, exp)


def load_json(settings_path):
    with open(settings_path) as data_file:
        param = json.load(data_file)

    return param


def setup_parser(exp):
    parser = argparse.ArgumentParser(description='Reproduce of multiple continual learning algorithms.')
    parser.add_argument('--config', type=str, default=f'./exps/{exp}.json',
                        help='Json file of settings.')

    return parser


if __name__ == '__main__':
    exps = ['lwf', 'finetune', 'ewc', 'wa', 'icarl', 'der']
    # for exp in exps:
    #     main(False, exp)
    for exp in exps[:1]:
        main(True, exp)
