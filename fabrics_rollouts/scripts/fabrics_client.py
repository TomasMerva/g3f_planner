import yaml
import socket

class FabricsClient():
    def __init__(self, config_file) -> None:
        with open(config_file, 'r') as config_file:
            self.config = yaml.safe_load(config_file)
   