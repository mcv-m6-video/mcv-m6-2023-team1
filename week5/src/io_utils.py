import yaml
import os


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


def create_dirs(dst_path):
    os.makedirs("data", exist_ok=True)
    os.makedirs(dst_path, exist_ok=True)
    print(f"Creating dataset in: {dst_path}")
    os.makedirs(f"{dst_path}/train", exist_ok=True)
    os.makedirs(f"{dst_path}/val", exist_ok=True)


def read_gt(gt_path):
    # Open the file for reading
    with open(gt_path, 'r') as file:
        # Initialize an empty dictionary
        my_dict = {}
        # Loop through each line in the file
        for line in file:
            # Split the line into columns
            columns = line.strip().split(",")
            # Use the first column as the key and the rest as the values
            key = columns[0]
            values = columns[1:]
            # Add the key-value pair to the dictionary
            if key not in my_dict.keys():
                my_dict[key] = [values]
            else:
                my_dict[key].append(values)
    return my_dict
