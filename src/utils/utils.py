import json
import yaml
import subprocess

CONFIG_PATH = "config.yaml"
PARAMS_PATH = "params.yaml"

def load_config(config_path=CONFIG_PATH):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config   

def load_params(params_path=PARAMS_PATH):
    with open(params_path, 'r') as file:
        params = yaml.safe_load(file)
    return params
