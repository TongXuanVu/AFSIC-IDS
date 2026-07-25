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
_FEDERATED_DIR = os.path.join(_LOCAL_DATA_DIR, "federated_data_10shot")
if not os.path.exists(_FEDERATED_DIR):
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

    # 2. Tìm thư mục chứa file huấn luyện của client
    # Thử 10shot trước
    tenshot_files = glob.glob("/kaggle/input/**/federated_data_10shot/client_*_task_*.pt", recursive=True)
    if tenshot_files:
        _FEDERATED_DIR = os.path.dirname(tenshot_files[0])
        print(f"[iCICIoT23] Auto-detected 10-Shot Data Dir: {_FEDERATED_DIR}")
    else:
        # Thử fewshot
        fewshot_files = glob.glob("/kaggle/input/**/federated_data_fewshot/client_*_task_*.pt", recursive=True)
        if fewshot_files:
            _FEDERATED_DIR = os.path.dirname(fewshot_files[0])
            print(f"[iCICIoT23] Auto-detected Few-Shot Data Dir: {_FEDERATED_DIR}")
        else:
            # Fallback data thường
            normal_files = glob.glob("/kaggle/input/**/federated_data/client_*_task_*.pt", recursive=True)
            if normal_files:
                _FEDERATED_DIR = os.path.dirname(normal_files[0])
                print(f"[iCICIoT23] Auto-detected Normal Data Dir: {_FEDERATED_DIR}")
            else:
                # Layout PHANG: Kaggle dataset khong giu thu muc con federated_data
                flat_files = glob.glob("/kaggle/input/**/client_*_task_*.pt", recursive=True)
                if flat_files:
                    _FEDERATED_DIR = os.path.dirname(flat_files[0])
                    print(f"[iCICIoT23] Auto-detected Flat Data Dir: {_FEDERATED_DIR}")
# ── Uu tien CAO NHAT: --fed_dir (truyen qua bien moi truong AFSIC_FED_DIR) ──────
# Auto-detect o tren uu tien 10shot > fewshot > full, nen khi attach nhieu dataset
# cung luc rat de chon nham thu muc. Co --fed_dir thi chi dinh tuong minh.
# Chap nhan ca duong dan day du lan chi TEN thu muc (vd: federated_data_fewshot).
_ENV_FED_DIR = os.environ.get("AFSIC_FED_DIR", "").strip()
if _ENV_FED_DIR:
    if os.path.isdir(_ENV_FED_DIR):
        _FEDERATED_DIR = _ENV_FED_DIR
    else:
        import glob as _g
        _hits = _g.glob(f"/kaggle/input/**/{_ENV_FED_DIR}/client_*_task_*.pt", recursive=True)
        if _hits:
            _FEDERATED_DIR = os.path.dirname(_hits[0])
        else:
            raise FileNotFoundError(
                f"[iCICIoT23] --fed_dir='{_ENV_FED_DIR}' khong phai thu muc ton tai "
                f"va cung khong tim thay trong /kaggle/input."
            )
    print(f"[iCICIoT23] --fed_dir override -> {_FEDERATED_DIR}")

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


def _load_task_class_order():
    """
    Bo data 100-client GIU NGUYEN label ID goc (preserve_original_label_ids) voi thu tu task
    phi tuan tu, kem file `task_mapping_label_ids.json`. DataManager remap label bang
    `class_order.index(y)` nen chi can dat class_order = thu tu label goc la khop hoan toan.

    Bo data cu (da remap tuan tu 0..33) khong co file json nay -> giu list(range(34)).
    """
    import json
    candidates = []
    if os.path.exists("/kaggle/input"):
        import glob as _glob
        candidates += _glob.glob("/kaggle/input/**/task_mapping_label_ids.json", recursive=True)
    candidates += [
        os.path.join(_FEDERATED_DIR, "task_mapping_label_ids.json"),
        os.path.join(os.path.dirname(_FEDERATED_DIR), "task_mapping_label_ids.json"),
        os.path.join(_LOCAL_DATA_DIR, "task_mapping_label_ids.json"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "task_mapping_label_ids.json"),
    ]
    for path in candidates:
        if path and os.path.exists(path):
            with open(path, "r") as f:
                task_orders = json.load(f)
            flat = [int(c) for task in task_orders for c in task]
            if sorted(flat) == list(range(len(flat))):
                print(f"[iCICIoT23] class_order theo label goc tu: {path}")
                print(f"[iCICIoT23] class_order = {flat}")
                return flat
            print(f"[iCICIoT23] CANH BAO: {path} khong phu kin 0..N-1, bo qua.")
    return list(range(34))


DEFAULT_TASK_CLASS_ORDER = _load_task_class_order()


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
