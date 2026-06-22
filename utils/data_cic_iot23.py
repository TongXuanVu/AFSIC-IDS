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
_LOCAL_DATA_DIR = os.path.join(_SPCIL_ROOT, "data", "CIC_IoT23")

# Mặc định lấy theo local
_TEST_FILE = os.path.join(_LOCAL_DATA_DIR, "global_test_data.pt")
_FEDERATED_DIR = os.path.join(_LOCAL_DATA_DIR, "federated_data_fewshot")
if not os.path.exists(_FEDERATED_DIR):
    _FEDERATED_DIR = os.path.join(_LOCAL_DATA_DIR, "federated_data")

# Quét độc lập file test và thư mục data trên Kaggle (Vì chúng có thể nằm ở 2 nhánh khác nhau)
if os.path.exists("/kaggle/input"):
    import glob
    print("[iCICIoT23] Đang quét toàn bộ /kaggle/input để tìm dữ liệu...")
    
    # 1. Tìm global_test_data.pt
    test_paths = glob.glob("/kaggle/input/**/global_test_data.pt", recursive=True)
    if test_paths:
        _TEST_FILE = test_paths[0]
        print(f"[iCICIoT23] Auto-detected Test File: {_TEST_FILE}")

    # 2. Tìm thư mục chứa file huấn luyện của client (ưu tiên _fewshot)
    # Lấy thử 1 file bất kỳ để dò đường dẫn thư mục
    fewshot_files = glob.glob("/kaggle/input/**/federated_data_fewshot/client_*_task_*.pt", recursive=True)
    if fewshot_files:
        _FEDERATED_DIR = os.path.dirname(fewshot_files[0])
        print(f"[iCICIoT23] Auto-detected Few-shot Data Dir: {_FEDERATED_DIR}")
    else:
        # Fallback nếu dùng data thường
        normal_files = glob.glob("/kaggle/input/**/federated_data/client_*_task_*.pt", recursive=True)
        if normal_files:
            _FEDERATED_DIR = os.path.dirname(normal_files[0])
            print(f"[iCICIoT23] Auto-detected Normal Data Dir: {_FEDERATED_DIR}")
_NUM_TASKS = 6

# Default supervised task-incremental order from data/final_pt_data_distribution.png.
# Original CIC-IoT23 labels are remapped by DataManager into incremental ids:
# Task 1: [1, 0, 11, 12, 27, 26]
# Task 2: [2, 14, 25, 24, 20, 28]
# Task 3: [3, 7, 30, 29, 19, 16]
# Task 4: [15, 6, 8, 22, 23, 21]
# Task 5: [5, 13, 10, 17, 18]
# Task 6: [4, 31, 32, 33, 9]
DEFAULT_TASK_CLASS_ORDER = list(range(34))


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

        # Load test set (global 30% split) ONLY for client 0 to save RAM
        if client_id == 0:
            assert os.path.exists(_TEST_FILE), f"[iCICIoT23] Không tìm thấy file test: {_TEST_FILE}"
            test_data_dict = torch.load(_TEST_FILE, weights_only=False)
            if isinstance(test_data_dict, dict):
                self.test_data = test_data_dict["x"].numpy().astype(np.float32)
                self.test_targets = test_data_dict["y"].numpy().astype(np.int64)
            else:
                self.test_data = test_data_dict[0].numpy().astype(np.float32)
                self.test_targets = test_data_dict[1].numpy().astype(np.int64)

            # class_order: giữ thứ tự tự nhiên 0, 1, 2, ..., 33
            self.class_order = DEFAULT_TASK_CLASS_ORDER
        else:
            # Các client khác không bao giờ dùng test_data nên không cần load (Tiết kiệm ~1.85GB RAM mỗi client)
            self.test_data = np.empty((0, num_features), dtype=np.float32)
            self.test_targets = np.empty((0,), dtype=np.int64)
            self.class_order = DEFAULT_TASK_CLASS_ORDER

        _print_stats(self, client_id)


def _print_stats(idata, client_id):
    n_classes = len(idata.class_order)
    print(f"[iCICIoT23 - Client {client_id}] Loaded: train={idata.train_data.shape}, test={idata.test_data.shape}")
