import numpy as np
import torch

class LocalExemplarMemory:
    def __init__(self, memory_ratio=0.01, memory_per_class=None):
        self.memory_ratio = memory_ratio
        self.memory_per_class = memory_per_class
        # Mapping from class_id -> numpy array of samples
        self.data_memory = {}
        # Mapping from class_id -> numpy array of targets
        self.targets_memory = {}

    def get_memory(self):
        if not self.data_memory:
            return None
        all_data = []
        all_targets = []
        for cid in sorted(self.data_memory.keys()):
            all_data.append(self.data_memory[cid])
            all_targets.append(self.targets_memory[cid])
        return np.concatenate(all_data, axis=0), np.concatenate(all_targets, axis=0)

    def construct_exemplars(self, class_id, data, targets, features):
        """
        Construct exemplars using herding selection.
        """
        num_samples = len(data)
        if num_samples == 0:
            return
        
        if self.memory_per_class is not None:
            m = min(int(self.memory_per_class), num_samples)
        else:
            m = int(np.ceil(float(self.memory_ratio) * num_samples))
            m = max(1, min(m, num_samples))
        
        if isinstance(features, torch.Tensor):
            features = features.detach().cpu().numpy()
        
        # Normalize features
        norms = np.linalg.norm(features, axis=1, keepdims=True) + 1e-8
        norm_features = features / norms
        class_mean = np.mean(norm_features, axis=0)
        class_mean = class_mean / (np.linalg.norm(class_mean) + 1e-8)
        

        
        selected_indices = []
        S = np.zeros(features.shape[1], dtype=np.float32)
        mask = np.zeros(num_samples, dtype=bool)
        
        for k in range(1, m + 1):
            target_vector = k * class_mean - S
            
            # Tối ưu hoá: Sử dụng Tích vô hướng (Dot Product) thay vì khoảng cách Euclid
            # Quét qua toàn bộ dữ liệu gốc, KHÔNG sử dụng Candidate Pooling
            scores = np.dot(norm_features, target_vector)
            
            # Loại bỏ các mẫu đã được chọn
            scores[mask] = -np.inf
                
            i = np.argmax(scores)
            mask[i] = True
            
            selected_indices.append(i)
            S += norm_features[i]
            
        self.data_memory[class_id] = np.array([data[idx] for idx in selected_indices])
        self.targets_memory[class_id] = np.array([targets[idx] for idx in selected_indices])


class GlobalPrototypeMemory:
    def __init__(self):
        # class_id -> { "prototype": tensor, "count": int, "dispersion": float, "quality": float }
        self.memory = {}

    def update_prototype(self, class_id, prototype, count, dispersion, quality=1.0):
        if isinstance(prototype, np.ndarray):
            prototype = torch.from_numpy(prototype).float()
        
        # Ensure L2 normalized
        norm_proto = prototype / (torch.norm(prototype, p=2) + 1e-8)
        
        self.memory[class_id] = {
            "prototype": norm_proto,
            "count": count,
            "dispersion": dispersion,
            "quality": quality
        }

    def get_prototype(self, class_id):
        if class_id in self.memory:
            return self.memory[class_id]["prototype"]
        return None

    def get_all_prototypes(self):
        # returns class_id -> prototype tensor
        return {cid: info["prototype"] for cid, info in self.memory.items()}

    def has_class(self, class_id):
        return class_id in self.memory

    def get(self, class_id):
        if class_id in self.memory:
            return self.memory[class_id]
        return None

    def get_all(self):
        return dict(self.memory)
