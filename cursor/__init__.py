import yaml
from pathlib import Path

with open("proj_config.yaml", "r") as f:
    CONFIG = yaml.safe_load(f)

PROJ_DIR = Path(CONFIG["proj_dir"])
DATA_DIR = Path(CONFIG["data_dir"])
OUTPUT_DIR = Path(CONFIG["output_dir"])