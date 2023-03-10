import argparse
import sys


def main(cfg):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args(sys.argv[1:])
    config_name = args.config_name

    main(config_name)
