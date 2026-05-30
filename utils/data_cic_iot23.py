"""
Dataset adapter cho CIC-IoT23 federated data.
Tích hợp vào SPCIL framework cho Federated Learning.

Data format:
  - Train: federated_data/client_{client_id}_task_{task_id}.pt
  - Test:  global_test_data.pt
"""
import numpy as np
import torch
import os


# --- Đường dẫn tới data ---
_SPCIL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_FL_ROOT = os.path.join(os.path.dirname(_SPCIL_ROOT), "FL")
_DATA_DIR = os.path.join(_FL_ROOT, "core", "data_split")
_FEDERATED_DIR = os.path.join(_DATA_DIR, "federated_data")
_TEST_FILE = os.path.join(_DATA_DIR, "global_test_data.pt")
_NUM_TASKS = 6


class iCICIoT23:
    """
    CIC-IoT23 dataset adapter tương thích với SPCIL DataManager cho Federated Learning.
    """
    use_path = False
    train_trsf = []
    test_trsf = []
    common_trsf = []

    def download_data(self, client_id=None):
        """Load data cho một client cụ thể."""
        assert client_id is not None, "[iCICIoT23] Yêu cầu client_id cho Federated Learning."
        
        task_data_list = []
        total_samples = 0
        num_features = 31 # Default cho CIC-IoT23

        for task_id in range(1, _NUM_TASKS + 1):
            path = os.path.join(_FEDERATED_DIR, f"client_{client_id}_task_{task_id}.pt")
            if os.path.exists(path):
                task_data = torch.load(path, weights_only=False)
                # handle both dict format {"x": tensor, "y": tensor} and tuple format
                if isinstance(task_data, dict):
                    x = task_data["x"]
                    y = task_data["y"]
                else:
                    x, y = task_data
                
                num_features = x.shape[1]
                total_samples += x.shape[0]
                task_data_list.append({"x": x, "y": y})
            else:
                # Client might not have data for this task, append empty to maintain task order
                task_data_list.append({"x": torch.empty((0, num_features)), "y": torch.empty((0,))})

        self.train_data = np.empty((total_samples, num_features), dtype=np.float32)
        self.train_targets = np.empty((total_samples,), dtype=np.int64)
        
        current_idx = 0
        for task_data in task_data_list:
            n_samples = task_data["x"].shape[0]
            if n_samples > 0:
                self.train_data[current_idx:current_idx + n_samples] = task_data["x"].numpy().astype(np.float32)
                self.train_targets[current_idx:current_idx + n_samples] = task_data["y"].numpy().astype(np.int64)
                current_idx += n_samples
                
        del task_data_list

        # Load test set (global 30% split)
        assert os.path.exists(_TEST_FILE), f"[iCICIoT23] Không tìm thấy file test: {_TEST_FILE}"
        test_data_dict = torch.load(_TEST_FILE, weights_only=False)
        if isinstance(test_data_dict, dict):
            self.test_data = test_data_dict["x"].numpy().astype(np.float32)
            self.test_targets = test_data_dict["y"].numpy().astype(np.int64)
        else:
            self.test_data = test_data_dict[0].numpy().astype(np.float32)
            self.test_targets = test_data_dict[1].numpy().astype(np.int64)

        # class_order: giữ thứ tự tự nhiên 0, 1, 2, ..., 33
        # Use a hardcoded or global known classes if possible, but taking unique from test targets covers all classes
        self.class_order = sorted(np.unique(self.test_targets).tolist())

        _print_stats(self, client_id)


def _print_stats(idata, client_id):
    n_classes = len(idata.class_order)
    print(f"[iCICIoT23 - Client {client_id}] Loaded: train={idata.train_data.shape}, test={idata.test_data.shape}")
