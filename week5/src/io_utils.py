import yaml


def open_config_yaml(cfg_path):
    """
    Open config file and return it as a dictionary
    :param cfg_path: path to config file
    :return: config dictionary
    """
    # Read Config file
    with open(cfg_path, 'r') as f:
        config_yaml = yaml.safe_load(f)

    return config_yaml
