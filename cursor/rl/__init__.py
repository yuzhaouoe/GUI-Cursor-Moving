import wandb
import huggingface_hub
import yaml


# load proj_config.yaml
with open("proj_config.yaml", "r") as f:
    proj_config = yaml.safe_load(f)

try:
    wandb.login(key=proj_config["wandb_token"])
except Exception as e:
    print(f"Error logging into Weights & Biases: {e}")
try:
    huggingface_hub.login(token=proj_config["hf_token"])
except Exception as e:
    print(f"Error logging into Hugging Face Hub: {e}")
