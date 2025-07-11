import json
import yaml

class AttributeDict(dict):
    def __init__(self, *args, **kwargs):
        # Recursively convert dictionaries to AttributeDict
        super().__init__(*args, **kwargs)
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = AttributeDict(value)

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"No attribute named '{key}'")

    def __setattr__(self, key, value):
        self[key] = value

def load_config(config_dir="config/train.yaml"):
    with open(config_dir, 'r') as f:
        config = yaml.safe_load(f)
        config['config_dir'] = config_dir
    return AttributeDict(config)