"""
Dataset adapter cho CIC-IoT23 centralized data từ FL project.
Tích hợp vào SPCIL framework.

Data format:
  - Train: 6 file .pt, mỗi file {"x": tensor(N,31), "y": tensor(N,)}
  - Test:  1 file .pt, mỗi file {"x": tensor(M,31), "y": tensor(M,)}
"""
import numpy as np
import torch
import os


# --- Đường dẫn tới data ---
_SPCIL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # c:\FederatedLearning\SPCIL
_FL_ROOT = os.path.join(os.path.dirname(_SPCIL_ROOT), "FL")               # c:\FederatedLearning\FL
_DATA_DIR = os.path.join(_FL_ROOT, "core", "data_split")
_CENTRALIZED_DIR = os.path.join(_DATA_DIR, "centralized_data")
_TEST_FILE = os.path.join(_DATA_DIR, "global_test_data.pt")
_NUM_TASKS = 6


class iCICIoT23:
    """
    CIC-IoT23 dataset adapter tương thích với SPCIL DataManager.
    
    Attributes:
        train_data   : np.ndarray (N_train, 31)
        train_targets: np.ndarray (N_train,)
        test_data    : np.ndarray (N_test, 31)
        test_targets : np.ndarray (N_test,)
        class_order  : list[int] — thứ tự classes, không shuffle
        use_path     : bool = False (dùng tensor trực tiếp)
        train_trsf   : list = [] (no transform cần thiết)
        test_trsf    : list = []
        common_trsf  : list = []
    """
    use_path = False
    train_trsf = []
    test_trsf = []
    common_trsf = []

    def download_data(self):
        """Load và ghép tất cả task data. 'download' là tên gọi theo SPCIL convention."""
        # Pre-calculate sizes to pre-allocate memory and avoid OOM
        total_samples = 0
        task_data_list = []
        for task_id in range(1, _NUM_TASKS + 1):
            path = os.path.join(_CENTRALIZED_DIR, f"centralized_task_{task_id}.pt")
            assert os.path.exists(path), f"[iCICIoT23] Không tìm thấy file: {path}"
            task_data = torch.load(path, weights_only=False)
            total_samples += task_data["x"].shape[0]
            task_data_list.append(task_data)
            
        num_features = task_data_list[0]["x"].shape[1]
        self.train_data = np.empty((total_samples, num_features), dtype=np.float32)
        self.train_targets = np.empty((total_samples,), dtype=np.int64)
        
        current_idx = 0
        for task_data in task_data_list:
            n_samples = task_data["x"].shape[0]
            self.train_data[current_idx:current_idx + n_samples] = task_data["x"].numpy().astype(np.float32)
            self.train_targets[current_idx:current_idx + n_samples] = task_data["y"].numpy().astype(np.int64)
            current_idx += n_samples
            
        del task_data_list
        import gc
        gc.collect()

        # Load test set (global 30% split)
        assert os.path.exists(_TEST_FILE), f"[iCICIoT23] Không tìm thấy file test: {_TEST_FILE}"
        test_data_dict = torch.load(_TEST_FILE, weights_only=False)
        self.test_data = test_data_dict["x"].numpy().astype(np.float32)
        self.test_targets = test_data_dict["y"].numpy().astype(np.int64)

        # class_order: giữ thứ tự tự nhiên 0, 1, 2, ..., 33
        all_classes = sorted(np.unique(self.train_targets).tolist())
        self.class_order = all_classes

        _print_stats(self)


def _print_stats(idata):
    n_classes = len(idata.class_order)
    print(f"[iCICIoT23] Loaded: train={idata.train_data.shape}, test={idata.test_data.shape}, classes={n_classes} (labels {idata.class_order[0]}->{idata.class_order[-1]})")
