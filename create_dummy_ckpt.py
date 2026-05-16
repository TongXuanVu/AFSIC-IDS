
import torch
import os
import sys

# Add current dir to path to import factory
sys.path.append(os.getcwd())
from utils import factory

def create_dummy():
    args = {
        "model_name": "der",
        "convnet_type": "cnn1d",
        "device": [torch.device("cpu")],
        "init_cls": 6,
        "increment": 6,
        "seed": 42,
        "dataset": "cic_iot23",
        "memory_size": 5000,
        "fixed_memory": False,
    }
    
    # Create model and expand it for 3 tasks
    model = factory.get_model("der", args)
    
    # Task 0
    model._cur_task = 0
    model._total_classes = 6
    model._network.update_fc(6)
    
    # Task 1
    model._cur_task = 1
    model._total_classes = 12
    model._network.update_fc(12)
    
    # Task 2
    model._cur_task = 2
    model._total_classes = 18
    model._network.update_fc(18)
    
    ckpt = {
        "task": 2,
        "round": 1,
        "model_state_dict": model._network.state_dict(),
        "known_classes": 12,
    }
    
    torch.save(ckpt, "task02_round001.pth")
    print("Created dummy checkpoint: task02_round001.pth")

if __name__ == "__main__":
    create_dummy()
