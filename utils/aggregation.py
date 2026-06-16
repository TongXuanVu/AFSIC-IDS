import logging
import numpy as np
import torch
import torch.nn.functional as F

def is_aggregated_state_key(key, task, aggregate_backbone=False):
    if task == 0 or aggregate_backbone:
        return True
    if "plasticity_adapter.frozen_source" in key:
        return False
    return any(sub in key for sub in ["plasticity_adapter.adapter", "gate", "fc"])

def compute_aggregation_weights(
    args,
    global_model,
    client_accs,
    client_protos,
    client_weights,
    global_state_round_start,
    active_client_indices,
    task
):
    Q_list = []
    drift_list = []
    update_norm_list = []
    
    beta_acc = args.get("beta_acc", 1.0)
    beta_proto = args.get("beta_proto", 1.0)
    beta_novelty = args.get("beta_novelty", 0.5)
    beta_drift = args.get("beta_drift", 0.5)
    beta_update = args.get("beta_update", 0.2)
    
    for c_idx, c in enumerate(active_client_indices):
        acc_i = client_accs[c]
        
        # Prototype Consistency
        proto_cons_vals = []
        for class_id in range(global_model._total_classes):
            local_p = client_protos[c].get(class_id, {}).get("prototype")
            global_p = global_model.global_proto_memory.get_prototype(class_id)
            if local_p is not None and global_p is not None:
                sim = torch.sum(F.normalize(local_p, p=2, dim=0) * F.normalize(global_p, p=2, dim=0)).item()
                proto_cons_vals.append(sim)
        proto_cons_i = sum(proto_cons_vals) / len(proto_cons_vals) if proto_cons_vals else 1.0
        
        # Novelty
        novelty_vals = []
        new_classes = range(global_model._known_classes, global_model._total_classes)
        old_classes = range(global_model._known_classes)
        if new_classes and old_classes:
            for n_c in new_classes:
                local_p = client_protos[c].get(n_c, {}).get("prototype")
                if local_p is not None:
                    local_p = F.normalize(local_p, p=2, dim=0)
                    min_dist = 1.0
                    for o_c in old_classes:
                        global_p = global_model.global_proto_memory.get_prototype(o_c)
                        if global_p is not None:
                            global_p = F.normalize(global_p, p=2, dim=0)
                            dist = 1.0 - torch.sum(local_p * global_p).item()
                            if dist < min_dist:
                                min_dist = dist
                    novelty_vals.append(min_dist)
        novelty_i = sum(novelty_vals) / len(novelty_vals) if novelty_vals else 0.5
        
        # Drift and Update Norm
        drift_val = 0.0
        update_val = 0.0
        num_params = 0
        local_dict = client_weights[c_idx]
        for k in local_dict.keys():
            if is_aggregated_state_key(k, task, args.get("aggregate_backbone", False)):
                diff = local_dict[k].float() - global_state_round_start[k].float()
                drift_val += torch.sum(diff ** 2).item()
                update_val += torch.sum(diff ** 2).item()
                num_params += diff.numel()
        
        drift_i = np.sqrt(drift_val / max(1, num_params))
        update_norm_i = np.sqrt(update_val / max(1, num_params))
        
        Q_i = beta_acc * acc_i + beta_proto * proto_cons_i + beta_novelty * novelty_i - beta_drift * drift_i - beta_update * update_norm_i
        Q_list.append(Q_i)
        drift_list.append(drift_i)
        update_norm_list.append(update_norm_i)
        
        logging.info(
            f"Client {c} => Q_i: {Q_i:.4f} | Acc: {acc_i*100:.2f}% | "
            f"ProtoCons: {proto_cons_i:.4f} | Novelty: {novelty_i:.4f} | "
            f"Drift: {drift_i:.4f} | UpdateNorm: {update_norm_i:.4f}"
        )
    
    accepted_positions = list(range(len(Q_list)))
    if args.get("robust_filter_updates", True) and len(Q_list) > 2:
        update_arr = np.array(update_norm_list, dtype=np.float64)
        drift_arr = np.array(drift_list, dtype=np.float64)
        update_med = float(np.median(update_arr))
        drift_med = float(np.median(drift_arr))
        update_mad = float(np.median(np.abs(update_arr - update_med))) + 1e-8
        drift_mad = float(np.median(np.abs(drift_arr - drift_med))) + 1e-8
        z_limit = args.get("robust_z", 3.5)
        max_update_norm = args.get("max_update_norm", None)
        accepted_positions = []
        for pos, c in enumerate(active_client_indices):
            update_ok = (update_arr[pos] - update_med) / update_mad <= z_limit
            drift_ok = (drift_arr[pos] - drift_med) / drift_mad <= z_limit
            norm_ok = max_update_norm is None or update_arr[pos] <= float(max_update_norm)
            if update_ok and drift_ok and norm_ok:
                accepted_positions.append(pos)
            else:
                logging.warning(
                    f"Client {c} rejected by robust filter | "
                    f"Drift: {drift_arr[pos]:.4f} | UpdateNorm: {update_arr[pos]:.4f}"
                )
        if not accepted_positions:
            logging.warning("Robust filter rejected all clients; falling back to all active clients.")
            accepted_positions = list(range(len(Q_list)))

    Q_accepted = [Q_list[pos] for pos in accepted_positions]
    Q_tensor = torch.tensor(Q_accepted, dtype=torch.float32)
    tau_agg = args.get("tau_aggregation", 1.0)
    alpha = torch.softmax(Q_tensor / tau_agg, dim=0).tolist()
    
    return alpha, accepted_positions, Q_accepted
