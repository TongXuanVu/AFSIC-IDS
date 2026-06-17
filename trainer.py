import sys
import logging
import copy
import csv
import os
import glob
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision import transforms

from utils import factory
from utils.data_manager import DataManager, DummyDataset
from utils.toolkit import count_parameters
from utils.aggregation import is_aggregated_state_key, compute_aggregation_weights
import seaborn as sns
from sklearn.metrics import confusion_matrix


def average_weights(w):
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        if 'num_batches_tracked' in key:
            w_avg[key] = w_avg[key].true_divide(len(w))
        else:
            w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def _make_transform(data_manager, mode):
    if mode == "train":
        return transforms.Compose([*data_manager._train_trsf, *data_manager._common_trsf])
    return transforms.Compose([*data_manager._test_trsf, *data_manager._common_trsf])


def _concat_or_none(parts):
    parts = [p for p in parts if p is not None and len(p) > 0]
    if not parts:
        return None
    return np.concatenate(parts, axis=0)


def _take_items(values, indices):
    if isinstance(values, np.ndarray):
        return values[indices]
    return [values[int(i)] for i in indices]


def _build_fewshot_train_dataset(data_manager, local_model, args, task, client_id):
    """Build few-shot new-class data plus local exemplar replay."""
    kshot = args.get("kshot", None) if args.get("fewshot_enabled", True) else None
    rng = np.random.default_rng(args.get("seed", 0) + task * 1009 + client_id * 9173)
    data_parts, target_parts = [], []
    new_count = 0

    for class_idx in range(local_model._known_classes, local_model._total_classes):
        data, targets, _ = data_manager.get_dataset(
            np.arange(class_idx, class_idx + 1),
            source="train",
            mode="test",
            ret_data=True,
        )
        if len(data) == 0:
            continue
        if kshot is None:
            keep = len(data)
            data_parts.append(data)
            target_parts.append(targets)
        else:
            keep = min(int(kshot), len(data))
            selected = rng.choice(len(data), size=keep, replace=False)
            data_parts.append(_take_items(data, selected))
            target_parts.append(_take_items(targets, selected))
        new_count += keep

    memory = local_model._get_memory()
    if memory is not None and len(memory) != 0:
        mem_data, mem_targets = memory
        data_parts.append(mem_data)
        target_parts.append(mem_targets)

    data = _concat_or_none(data_parts)
    targets = _concat_or_none(target_parts)
    if data is None or targets is None:
        return None

    dataset = DummyDataset(data, targets, _make_transform(data_manager, "train"), data_manager.use_path)
    dataset.len_new_data = new_count
    return dataset


def _build_standard_train_dataset(data_manager, local_model, args):
    dataset = data_manager.get_dataset(
        np.arange(local_model._known_classes, local_model._total_classes),
        source="train",
        mode="train",
        appendent=local_model._get_memory(),
    )
    dataset.len_new_data = getattr(dataset, "len_source", len(dataset))
    return dataset


def _build_local_eval_loader(data_manager, total_classes, batch_size):
    indices = np.arange(0, total_classes)
    dataset = data_manager.get_dataset(indices, source="test", mode="test")
    if len(dataset) == 0:
        dataset = data_manager.get_dataset(indices, source="train", mode="test")
    if len(dataset) == 0:
        return None
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)


def _build_global_learned_test_loader(global_data_manager, total_classes, batch_size):
    """Evaluate only classes that have already been introduced.

    CIC-IoT23 global_test_data can contain all 34 classes, while task 0 may
    expose only 6 classes. Filtering here prevents penalizing the model for
    classes that are intentionally unseen at the current incremental stage.
    """
    learned_classes = np.arange(0, total_classes)
    dataset = global_data_manager.get_dataset(learned_classes, source="test", mode="test")
    if len(dataset) == 0:
        return None
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)


def _aggregate_client_prototypes(global_model, client_protos, num_clients, quality_scores=None):
    for class_id in range(global_model._total_classes):
        active_protos = []
        weights = []
        dispersions = []
        for c in range(num_clients):
            if quality_scores is not None and c not in quality_scores:
                continue
            if class_id not in client_protos[c]:
                continue
            info = client_protos[c][class_id]
            active_protos.append(info["prototype"])
            quality = 1.0 if quality_scores is None else max(0.0, float(quality_scores.get(c, 1.0)))
            weights.append(max(1, int(info.get("count", 1))) * quality)
            dispersions.append(float(info.get("dispersion", 0.0)))

        if not active_protos:
            continue

        stacked = torch.stack(active_protos)
        w_tensor = torch.tensor(weights, dtype=stacked.dtype).unsqueeze(1)
        w_tensor = w_tensor / (torch.sum(w_tensor) + 1e-8)
        global_proto = torch.sum(stacked * w_tensor, dim=0)
        global_proto = global_proto / (torch.norm(global_proto, p=2) + 1e-8)
        total_count = int(sum(
            max(1, int(client_protos[c][class_id].get("count", 1)))
            for c in range(num_clients)
            if class_id in client_protos[c] and (quality_scores is None or c in quality_scores)
        ))
        mean_dispersion = sum(dispersions) / len(dispersions)
        mean_quality = float(sum(weights) / max(1, total_count))
        global_model.global_proto_memory.update_prototype(class_id, global_proto, total_count, mean_dispersion, mean_quality)


def _calibrate_classifier_from_prototypes(model):
    if not hasattr(model, "global_proto_memory"):
        return
    if not model.args.get("calibrate_with_prototypes", True):
        return
    model._network.init_new_class_weights_from_prototypes(
        model.global_proto_memory.get_all_prototypes(),
        range(model._total_classes),
    )


def train(args):
    seed_list = copy.deepcopy(args["seed"])
    device = copy.deepcopy(args["device"])

    for seed in seed_list:
        args["seed"] = seed
        args["device"] = device
        _train_federated(args)


def _train_federated(args):
    init_cls = 0 if args["init_cls"] == args["increment"] else args["init_cls"]

    timestamp = datetime.now().strftime("%d-%m-%y_%H-%M")
    run_dir = os.path.join(
        "logs",
        args["model_name"] + "_federated",
        args["dataset"],
        "{}_seed{}_{}_clients{}".format(
            timestamp, args["seed"], args["convnet_type"], args["num_clients"]
        ),
    )
    os.makedirs(run_dir, exist_ok=True)
    ckpt_dir = os.path.join(run_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    logfilename = os.path.join(run_dir, "training.log")
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[logging.FileHandler(filename=logfilename), logging.StreamHandler(sys.stdout)],
    )

    csv_path = os.path.join(run_dir, "metrics_round_by_round.csv")
    csv_file = open(csv_path, "a" if args.get("resume") else "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    if not args.get("resume"):
        csv_writer.writerow([
            "task", "round", "global_round", "method",
            "acc", "prec_mic", "prec_mac", "prec_wei",
            "rec_mic", "rec_mac", "rec_wei",
            "f1_mic", "f1_mac", "f1_wei", "loss", "avg_acc",
        ])

    _set_random()
    _set_device(args)
    print_args(args)
    logging.info("Run directory: {}".format(run_dir))
    args["run_dir"] = run_dir

    logging.info(f"Initializing DataManagers for {args['num_clients']} clients...")
    client_dms = []
    for c in range(args["num_clients"]):
        dm = DataManager(
            args["dataset"],
            args["shuffle"],
            args["seed"],
            args["init_cls"],
            args["increment"],
            client_id=c,
            class_order=args.get("class_order"),
            task_increments=args.get("task_increments"),
        )
        if args.get("debug"):
            dm._train_data = dm._train_data[:2000]
            dm._train_targets = dm._train_targets[:2000]
        client_dms.append(dm)

    nb_tasks = client_dms[0].nb_tasks
    logging.info(f"Task increments: {client_dms[0]._increments}")

    global_model = factory.get_model(args["model_name"], args)
    local_models = [factory.get_model(args["model_name"], args) for _ in range(args["num_clients"])]

    start_task = 0
    start_round = 0
    results_all = []

    # ── Lịch sử metrics để vẽ biểu đồ ──────────────────────────────────────
    history = {
        "cnn":  {"acc": [], "precision": [], "recall": [], "f1": []},
        "nme":  {"acc": [], "precision": [], "recall": [], "f1": []},
    }
    cnn_curve, nme_curve = {"top1": [], "top5": []}, {"top1": [], "top5": []}

    checkpoint = None
    if args.get("resume"):
        if os.path.isfile(args["resume"]):
            logging.info(f"==> Resuming from checkpoint: {args['resume']}")
            checkpoint = torch.load(args["resume"], map_location='cpu', weights_only=False)
            start_task = checkpoint['task']
            if checkpoint.get('is_memory_phase', False):
                start_round = args["num_rounds"]
            else:
                start_round = checkpoint['round'] + 1
                if start_round >= args["num_rounds"]:
                    start_round = args["num_rounds"]
        else:
            raise FileNotFoundError(f"Checkpoint file not found: {args['resume']}")
            
    for task in range(nb_tasks):
        # Save previous global network BEFORE expansion. AFSIC KD needs this old model.
        prev_global_network = copy.deepcopy(global_model._network) if args["model_name"] == "afsic-ids" and task > 0 else None

        # 1. Mở rộng kiến trúc (nhưng không train) để lấy đúng kích thước mô hình
        global_model.incremental_train(client_dms[0], skip_train=True)
        global_model._network.to(args["device"][0])

        for c in range(args["num_clients"]):
            local_models[c].skip_rehearsal = True
            local_models[c].incremental_train(client_dms[c], skip_train=True)
            local_models[c].skip_rehearsal = False

        if checkpoint is not None and task == checkpoint['task']:
            logging.info(f"Phục hồi trạng thái cho Task {task} từ Checkpoint...")
            global_model._network.load_state_dict(checkpoint['model_state_dict'])
            if args["model_name"] == "afsic-ids" and checkpoint.get('global_proto_memory') is not None:
                global_model.global_proto_memory = checkpoint['global_proto_memory']
            for c in range(args["num_clients"]):
                c_state = checkpoint['client_states'][c]
                local_models[c]._data_memory = c_state.get('data_memory')
                local_models[c]._targets_memory = c_state.get('targets_memory')
                if args["model_name"] == "afsic-ids" and c_state.get('local_memory') is not None:
                    local_models[c].local_memory = c_state['local_memory']

        for c in range(args["num_clients"]):
            if args["model_name"] == "afsic-ids" and task > 0:
                local_models[c].train_loader = None
                continue
            else:
                train_dataset = None
                has_file = True
                if args["dataset"] == "cic_iot23":
                    from utils.data_cic_iot23 import _FEDERATED_DIR
                    task_file = os.path.join(_FEDERATED_DIR, f"client_{c}_task_{task+1}.pt")
                    if not os.path.exists(task_file):
                        has_file = False
                        logging.info(f"Client {c} không có file {os.path.basename(task_file)}. Tự động bỏ qua.")
                
                if has_file:
                    train_dataset = _build_standard_train_dataset(client_dms[c], local_models[c], args)
            # Ngăn chặn Model Collapse: Nếu client chỉ có Rehearsal Memory mà không có data mới, thì BỎ QUA không train
            has_new_data = (
                train_dataset is not None
                and getattr(train_dataset, 'len_new_data', getattr(train_dataset, 'len_source', len(train_dataset))) > 0
            )
            
            if train_dataset is not None and len(train_dataset) > 0 and has_new_data:
                local_models[c].train_loader = torch.utils.data.DataLoader(
                    train_dataset, batch_size=args["batch_size"], shuffle=True, num_workers=0
                )
            else:
                local_models[c].train_loader = None

        if task < start_task:
            for c in range(args["num_clients"]):
                local_models[c].after_task()
            global_model.after_task()
            continue

        logging.info(f"========== Bắt đầu Task {task} ==========")
        current_start_round = start_round if task == start_task else 0
        # Stage t >= 1: Supervised task-incremental registration and prototype initialization
        if args["model_name"] == "afsic-ids" and task > 0:
            logging.info("Supervised task-incremental registration: computing prototypes from labeled task data.")
            client_protos = []
            for c in range(args["num_clients"]):
                local_models[c]._network.load_state_dict(global_model._network.state_dict())
                local_models[c]._network.to(args["device"][0])
                old_protos = local_models[c].compute_local_prototypes(
                    client_dms[c],
                    class_ids=range(local_models[c]._known_classes),
                    seed=args.get("seed", 0) + c,
                )
                max_samples = args.get("kshot", 10) if args.get("fewshot_enabled", True) else None
                new_protos = local_models[c].compute_local_prototypes(
                    client_dms[c],
                    class_ids=range(local_models[c]._known_classes, local_models[c]._total_classes),
                    max_samples_per_class=max_samples,
                    seed=args.get("seed", 0) + task * 1009 + c,
                )
                protos = {}
                protos.update(old_protos)
                protos.update(new_protos)
                client_protos.append(protos)
            
            _aggregate_client_prototypes(global_model, client_protos, args["num_clients"])
            
            for c in range(args["num_clients"]):
                local_models[c].global_proto_memory = copy.deepcopy(global_model.global_proto_memory)
                _calibrate_classifier_from_prototypes(local_models[c])
            
            _calibrate_classifier_from_prototypes(global_model)
            logging.info("Class classifier weights successfully initialized from prototypes.")

            for c in range(args["num_clients"]):
                train_dataset = _build_fewshot_train_dataset(
                    client_dms[c],
                    local_models[c],
                    args,
                    task,
                    c
                )
                has_new_data = (
                    train_dataset is not None
                    and getattr(train_dataset, 'len_new_data', getattr(train_dataset, 'len_source', len(train_dataset))) > 0
                )
                if train_dataset is not None and len(train_dataset) > 0 and has_new_data:
                    local_models[c].train_loader = torch.utils.data.DataLoader(
                        train_dataset, batch_size=args["batch_size"], shuffle=True, num_workers=0
                    )
                else:
                    local_models[c].train_loader = None

        # Keep track of best accuracies for forgetting calculations
        if 'best_accs' not in locals():
            best_accs = {}

        for round_idx in range(current_start_round, args["num_rounds"]):
            global_round = task * args["num_rounds"] + round_idx
            logging.info(f"--- Task {task} | Round {round_idx+1}/{args['num_rounds']} (Global {global_round+1}) ---")
            client_weights = []
            client_accs = []
            client_protos = []
            
            global_state_round_start = copy.deepcopy(global_model._network.state_dict())
            
            for c in range(args["num_clients"]):
                if local_models[c].train_loader is None: 
                    client_accs.append(0.0)
                    client_protos.append({})
                    continue
                
                local_models[c]._network.load_state_dict(global_model._network.state_dict())
                local_models[c]._network.to(args["device"][0])
                
                if args["model_name"] == "afsic-ids":
                    local_models[c].global_proto_memory = copy.deepcopy(global_model.global_proto_memory)
                    if task > 0 and prev_global_network is not None:
                        local_models[c]._old_network = copy.deepcopy(prev_global_network).to(args["device"][0])
                        local_models[c]._old_network.eval()
                        for p in local_models[c]._old_network.parameters():
                            p.requires_grad = False
                
                local_models[c].args["epochs"] = args["local_epochs"]
                local_models[c].args["start_round"] = 0
                local_models[c]._train(local_models[c].train_loader, None)
                
                client_weights.append(copy.deepcopy(local_models[c]._network.state_dict()))
                
                if args["model_name"] == "afsic-ids":
                    # Fast eval for client quality validation Acc
                    local_models[c]._network.eval()
                    quality_loader = _build_local_eval_loader(client_dms[c], local_models[c]._total_classes, args["batch_size"])
                    if quality_loader is not None:
                        test_acc_dict = local_models[c]._compute_accuracy(local_models[c]._network, quality_loader)
                        client_accs.append(test_acc_dict["total"] / 100.0)
                    else:
                        client_accs.append(0.0)
                    
                    # Compute local prototypes
                    old_protos = local_models[c].compute_local_prototypes(
                        client_dms[c],
                        class_ids=range(local_models[c]._known_classes),
                        seed=args.get("seed", 0) + c,
                    )
                    new_protos = local_models[c].compute_local_prototypes(
                        client_dms[c],
                        class_ids=range(local_models[c]._known_classes, local_models[c]._total_classes),
                        max_samples_per_class=(
                            args.get("kshot", 10)
                            if task > 0 and args.get("fewshot_enabled", True)
                            else None
                        ),
                        seed=args.get("seed", 0) + task * 1009 + c,
                    )
                    protos = {}
                    protos.update(old_protos)
                    protos.update(new_protos)
                    client_protos.append(protos)
                else:
                    client_accs.append(0.0)
                    client_protos.append({})
            
            if client_weights:
                if args["model_name"] == "afsic-ids":
                    # a. Update global prototypes on the server
                    _aggregate_client_prototypes(global_model, client_protos, args["num_clients"])
                    
                    for c in range(args["num_clients"]):
                        local_models[c].global_proto_memory = copy.deepcopy(global_model.global_proto_memory)
                    
                    # b. Compute quality scores and aggregation weights
                    active_client_indices = [c for c in range(args["num_clients"]) if local_models[c].train_loader is not None]
                    
                    alpha, accepted_positions, Q_list = compute_aggregation_weights(
                        args=args,
                        global_model=global_model,
                        client_accs=client_accs,
                        client_protos=client_protos,
                        client_weights=client_weights,
                        global_state_round_start=global_state_round_start,
                        active_client_indices=active_client_indices,
                        task=task
                    )
                    
                    active_client_indices = [active_client_indices[pos] for pos in accepted_positions]
                    client_weights_accepted = [client_weights[pos] for pos in accepted_positions]
                    
                    for idx, c in enumerate(active_client_indices):
                        logging.info(f"Client {c} Aggregation Weight alpha_i: {alpha[idx]:.4f}")

                    quality_scores = {c: alpha[idx] for idx, c in enumerate(active_client_indices)}
                    _aggregate_client_prototypes(global_model, client_protos, args["num_clients"], quality_scores)
                    for c in range(args["num_clients"]):
                        local_models[c].global_proto_memory = copy.deepcopy(global_model.global_proto_memory)
                    
                    # Perform quality-aware weighted aggregation
                    global_dict = copy.deepcopy(global_model._network.state_dict())
                    aggregate_backbone = args.get("aggregate_backbone", False)
                    
                    for k in global_dict.keys():
                        if is_aggregated_state_key(k, task, aggregate_backbone):
                            val = client_weights_accepted[0][k].float() * alpha[0]
                            for c_idx in range(1, len(client_weights_accepted)):
                                val += client_weights_accepted[c_idx][k].float() * alpha[c_idx]
                            global_dict[k] = val.to(global_dict[k].dtype)
                    global_model._network.load_state_dict(global_dict)
                    _calibrate_classifier_from_prototypes(global_model)
                else:
                    global_weights = average_weights(client_weights)
                    global_model._network.load_state_dict(global_weights)

            # ── Đánh giá Global Model cuối MỖI ROUND ──
            global_model.test_loader = _build_global_learned_test_loader(
                client_dms[0],
                global_model._total_classes,
                args["batch_size"],
            )
            if global_model.test_loader is None:
                logging.warning("Global learned-class test set is empty; skipping evaluation for this round.")
                continue
            logging.info(
                f"Evaluating on global_test_data filtered to learned classes: "
                f"0-{global_model._total_classes - 1}"
            )
            
            cnn_accy, nme_accy, y_pred, y_true = global_model.eval_task()
            
            results_all.append(cnn_accy)
            avg_acc = sum(r['top1'] for r in results_all) / len(results_all)
            
            # Calculate parameter growth and communication cost
            total_params = sum(p.numel() for p in global_model._network.parameters())
            trainable_params = sum(p.numel() for p in global_model._network.parameters() if p.requires_grad)
            comm_cost = trainable_params * 4 / (1024 * 1024) # MB
            
            # Grouped forgetting calculation
            forgetting_vals = []
            for t in range(task):
                start_cls = t * args.get("increment", 10)
                end_cls = start_cls + args.get("increment", 10) - 1
                group_key = f"acc_{start_cls:02d}-{end_cls:02d}"
                if group_key in cnn_accy:
                    curr_acc = cnn_accy[group_key]
                    best_acc_key = f"best_{group_key}"
                    if best_acc_key not in best_accs:
                        best_accs[best_acc_key] = curr_acc
                    else:
                        best_accs[best_acc_key] = max(best_accs[best_acc_key], curr_acc)
                    
                    forgetting = max(0.0, best_accs[best_acc_key] - curr_acc)
                    forgetting_vals.append(forgetting)
            mean_forgetting = sum(forgetting_vals) / len(forgetting_vals) if forgetting_vals else 0.0

            logging.info(
                f"[Task {task} | Round {round_idx+1}] "
                f"Acc: {cnn_accy['top1']:.2f}% | "
                f"Old Acc: {cnn_accy.get('old_acc', 0):.2f}% | "
                f"New Acc: {cnn_accy.get('new_acc', 0):.2f}% | "
                f"Forgetting: {mean_forgetting:.2f}% | "
                f"Params: {total_params:,} | Comm Cost: {comm_cost:.2f} MB"
            )

            # Ghi file CSV
            csv_writer.writerow([
                task, round_idx + 1, global_round + 1, "AFSIC-IDS",
                round(cnn_accy["top1"], 4),
                round(cnn_accy.get("precision_micro", 0), 4),
                round(cnn_accy.get("precision_macro", 0), 4),
                round(cnn_accy.get("precision_weighted", 0), 4),
                round(cnn_accy.get("recall_micro", 0), 4),
                round(cnn_accy.get("recall_macro", 0), 4),
                round(cnn_accy.get("recall_weighted", 0), 4),
                round(cnn_accy.get("f1_micro", 0), 4),
                round(cnn_accy.get("f1_macro", 0), 4),
                round(cnn_accy.get("f1_weighted", 0), 4),
                round(cnn_accy.get("loss", 0), 6),
                round(avg_acc, 4),
            ])
            csv_file.flush()

            if round_idx == args["num_rounds"] - 1:
                cnn_curve["top1"].append(cnn_accy["top1"])
                cnn_curve["top5"].append(cnn_accy["top5"])
                history["cnn"]["acc"].append(cnn_accy["top1"])
                history["cnn"]["precision"].append(cnn_accy.get("precision_macro", 0))
                history["cnn"]["recall"].append(cnn_accy.get("recall_macro", 0))
                history["cnn"]["f1"].append(cnn_accy.get("f1_macro", 0))
                
                if nme_accy is not None:
                    nme_curve["top1"].append(nme_accy["top1"])
                    nme_curve["top5"].append(nme_accy["top5"])
                    history["nme"]["acc"].append(nme_accy["top1"])
                    history["nme"]["precision"].append(nme_accy.get("precision_macro", 0))
                    history["nme"]["recall"].append(nme_accy.get("recall_macro", 0))
                    history["nme"]["f1"].append(nme_accy.get("f1_macro", 0))
                
                plot_confusion_matrix(y_true, y_pred, task, run_dir)

            # Lưu Checkpoint mỗi Round
            client_states = []
            for c in range(args["num_clients"]):
                client_states.append({
                    'data_memory': getattr(local_models[c], '_data_memory', None),
                    'targets_memory': getattr(local_models[c], '_targets_memory', None),
                    'local_memory': getattr(local_models[c], 'local_memory', None),
                })
            ckpt_name = f'ckpt_round{global_round+1:04d}_task{task:02d}_r{round_idx+1:03d}_acc{cnn_accy["top1"]:.1f}.pth'
            torch.save({
                'task': task,
                'round': round_idx,
                'global_round': global_round,
                'model_state_dict': global_model._network.state_dict(),
                'known_classes': global_model._known_classes,
                'client_states': client_states,
                'global_proto_memory': getattr(global_model, 'global_proto_memory', None),
                'metrics': cnn_accy
            }, os.path.join(ckpt_dir, ckpt_name))

        # Cuối Task, xây dựng lại bộ nhớ Rehearsal
        logging.info(f"Xây dựng Rehearsal Memory cho các Clients tại cuối Task {task}...")
        current_client_start = checkpoint.get('last_client_done', -1) + 1 if (checkpoint is not None and task == checkpoint['task']) else 0
        
        for c in range(args["num_clients"]):
            if c >= current_client_start:
                if local_models[c].train_loader is not None:
                    local_models[c]._network.load_state_dict(global_model._network.state_dict())
                    local_models[c]._network.to(args["device"][0])
                    try:
                        local_models[c].build_rehearsal_memory(client_dms[c], local_models[c].samples_per_class)
                    except Exception as e:
                        logging.warning(f"Lỗi khi build memory cho client {c}: {e}")
                
                # ===== LƯU CHECKPOINT NGAY SAU KHI CLIENT C TẠO XONG MEMORY =====
                client_states = []
                for state_c in range(args["num_clients"]):
                    client_states.append({
                        'data_memory': getattr(local_models[state_c], '_data_memory', None),
                        'targets_memory': getattr(local_models[state_c], '_targets_memory', None),
                        'local_memory': getattr(local_models[state_c], 'local_memory', None),
                    })
                    
                ckpt_name = f'ckpt_task{task:02d}_memory_client{c:02d}.pth'
                torch.save({
                    'task': task,
                    'round': args["num_rounds"] - 1,
                    'global_round': checkpoint['global_round'] if (checkpoint is not None and 'global_round' in checkpoint) else (task * args["num_rounds"] + args["num_rounds"] - 1),
                    'is_memory_phase': True,
                    'last_client_done': c,
                    'model_state_dict': global_model._network.state_dict(),
                    'client_states': client_states,
                    'known_classes': global_model._known_classes,
                    'global_proto_memory': getattr(global_model, 'global_proto_memory', None)
                }, os.path.join(ckpt_dir, ckpt_name))
                logging.info(f"Đã lưu Checkpoint Memory an toàn cho Client {c}")
                
            local_models[c].after_task()

        global_model.after_task()

    csv_file.close()
    
    # ── Vẽ biểu đồ ──────────────────────────────────────────────────────────
    _plot_metrics(history, run_dir, args)
    logging.info("Plots saved in: {}".format(run_dir))
    
    logging.info("Training Finished.")


def run_test(args):
    """
    Chế độ TEST: Tải các checkpoint và đánh giá toàn bộ.
    """
    _set_random()
    _set_device(args)
    
    test_ckpt_root = args.get("test_checkpoint_dir", "")
    if not test_ckpt_root or not os.path.exists(test_ckpt_root):
        logging.error(f"[TEST] Thư mục checkpoint không hợp lệ: {test_ckpt_root}")
        return

    ckpt_files = sorted(glob.glob(os.path.join(test_ckpt_root, "checkpoints", "ckpt_round*.pth")))
    if not ckpt_files:
        logging.error(f"[TEST] Không tìm thấy checkpoint nào trong {test_ckpt_root}/checkpoints/")
        return
        
    logging.info(f"[TEST] Tìm thấy {len(ckpt_files)} checkpoint. Bắt đầu đánh giá...")
    
    # Init DataManager cho Client 0 để lấy Test Set chung
    dm = DataManager(
        args["dataset"],
        False,
        args["seed"],
        args["init_cls"],
        args["increment"],
        client_id=0,
        class_order=args.get("class_order"),
        task_increments=args.get("task_increments"),
    )
    
    csv_path = os.path.join(test_ckpt_root, "test_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow([
            "checkpoint", "task", "round", "global_round", "acc", 
            "prec_mic", "prec_mac", "prec_wei", 
            "rec_mic", "rec_mac", "rec_wei", 
            "f1_mic", "f1_mac", "f1_wei"
        ])

        global_model = factory.get_model(args["model_name"], args)
        
        for idx, cp in enumerate(ckpt_files):
            state = torch.load(cp, map_location='cpu', weights_only=False)
            task = state['task']
            
            # Cập nhật kiến trúc Model theo số task
            global_model = factory.get_model(args["model_name"], args)
            for _ in range(task + 1):
                global_model.incremental_train(dm, skip_train=True)
            
            global_model._network.load_state_dict(state['model_state_dict'])
            global_model._network.to(args["device"][0])
            global_model._network.eval()
            
            global_model.test_loader = _build_global_learned_test_loader(
                dm,
                global_model._total_classes,
                args["batch_size"],
            )
            if global_model.test_loader is None:
                logging.warning(f"[TEST] Empty learned-class global test set for checkpoint: {os.path.basename(cp)}")
                continue
            
            cnn_accy, _, y_pred, y_true = global_model.eval_task()
            
            logging.info(f"[TEST] {os.path.basename(cp)} | Task {task} | Acc: {cnn_accy['top1']:.2f}% | F1-Mac: {cnn_accy.get('f1_macro', 0):.2f}%")
            
            writer.writerow([
                os.path.basename(cp), task, state['round'], state['global_round'],
                round(cnn_accy["top1"], 4),
                round(cnn_accy.get("precision_micro", 0), 4),
                round(cnn_accy.get("precision_macro", 0), 4),
                round(cnn_accy.get("precision_weighted", 0), 4),
                round(cnn_accy.get("recall_micro", 0), 4),
                round(cnn_accy.get("recall_macro", 0), 4),
                round(cnn_accy.get("recall_weighted", 0), 4),
                round(cnn_accy.get("f1_micro", 0), 4),
                round(cnn_accy.get("f1_macro", 0), 4),
                round(cnn_accy.get("f1_weighted", 0), 4)
            ])
            
            # Vẽ Confusion Matrix cho checkpoint cuối
            if idx == len(ckpt_files) - 1:
                try:
                    plot_confusion_matrix(y_true, y_pred, task, test_ckpt_root)
                except Exception as e:
                    logging.error(f"[TEST] Lỗi khi vẽ Confusion Matrix: {e}")

    logging.info(f"[TEST] Hoàn thành. Kết quả được lưu tại: {csv_path}")


def _set_device(args):
    device_type = args["device"]
    gpus = []
    for device in device_type:
        if str(device) == "-1" or str(device).lower() == "cpu":
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:{}".format(device))
        gpus.append(device)
    args["device"] = gpus


def _set_random():
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_args(args):
    for key, value in args.items():
        logging.info("{}: {}".format(key, value))


def _plot_metrics(history, run_dir, args):
    tasks = list(range(1, len(history["cnn"]["acc"]) + 1))
    has_nme = len(history["nme"]["acc"]) > 0

    metrics = ["acc", "precision", "recall", "f1"]
    labels  = ["Accuracy (%)", "Precision (%)", "Recall (%)", "F1-Score (%)"]
    colors  = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "DER + {} on {} — {}\nSeed: {}  |  Init: {}  Inc: {}".format(
            args["convnet_type"], args["dataset"],
            datetime.now().strftime("%Y-%m-%d %H:%M"),
            args["seed"], args["init_cls"], args["increment"],
        ),
        fontsize=13, fontweight="bold",
    )

    for idx, (metric, label, color) in enumerate(zip(metrics, labels, colors)):
        ax = axes[idx // 2][idx % 2]
        cnn_vals = history["cnn"][metric]

        ax.plot(tasks, cnn_vals, "o-", color=color, linewidth=2,
                markersize=6, label="CNN")
        if has_nme and len(history["nme"][metric]) == len(tasks):
            nme_vals = history["nme"][metric]
            ax.plot(tasks, nme_vals, "s--", color=color, linewidth=2,
                    markersize=6, alpha=0.6, label="NME")

        ax.set_title(label, fontsize=11)
        ax.set_xlabel("Task", fontsize=10)
        ax.set_ylabel(label, fontsize=10)
        ax.set_xticks(tasks)
        ax.set_ylim(0, 105)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(fontsize=9)

        # Annotate each point
        for t, v in zip(tasks, cnn_vals):
            ax.annotate(f"{v:.1f}", (t, v),
                        textcoords="offset points", xytext=(0, 6),
                        ha="center", fontsize=8, color=color)

    plt.tight_layout()
    plot_path = os.path.join(run_dir, "metrics_plot.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    # ── Biểu đồ tổng hợp 4 metrics trên 1 axes ──────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    for metric, label, color in zip(metrics, labels, colors):
        ax2.plot(tasks, history["cnn"][metric], "o-", color=color,
                 linewidth=2, markersize=6, label=label)
    ax2.set_title("CNN — All Metrics per Task", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Task")
    ax2.set_ylabel("Score (%)")
    ax2.set_xticks(tasks)
    ax2.set_ylim(0, 105)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(fontsize=10)
    plt.tight_layout()
    combined_path = os.path.join(run_dir, "all_metrics_combined.png")
    plt.savefig(combined_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_test_plot(x_vals, y_vals, metric_name, color, marker, args):
    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, y_vals, f'{color}-{marker}', linewidth=2, markersize=4)
    plt.xlabel('Task')
    plt.ylabel(f'{metric_name} (%)' if metric_name != 'Loss' else 'Loss')
    plt.title(f'[TEST - SPCIL] {metric_name} over Tasks')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    safe_name = metric_name.lower().replace("-", "_")
    plt.savefig(os.path.join(args.get('run_dir', '.'), f'test_spcil_{safe_name}.png'), dpi=150)
    plt.close()


def save_combined_plot(x_vals, y_mic, y_mac, y_wei, category_name, args):
    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, y_mic, 'b-o', label=f'Micro-{category_name}', linewidth=1.5, markersize=3)
    plt.plot(x_vals, y_mac, 'g-s', label=f'Macro-{category_name}', linewidth=1.5, markersize=3)
    plt.plot(x_vals, y_wei, 'r-^', label=f'Weighted-{category_name}', linewidth=1.5, markersize=3)
    plt.xlabel('Task')
    plt.ylabel(f'{category_name} (%)')
    plt.title(f'[TEST - SPCIL] {category_name} (Micro vs Macro vs Weighted)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.get('run_dir', '.'), f'test_spcil_{category_name.lower()}_combined.png'), dpi=150)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, task_id, run_dir):
    """Ve va luu Confusion Matrix PNG"""
    # y_pred thuong co dang [N, topk], lay top1
    if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
        y_pred_top1 = y_pred[:, 0]
    else:
        y_pred_top1 = y_pred.flatten()
        
    cm = confusion_matrix(y_true, y_pred_top1)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix - Task {task_id}')
    
    save_path = os.path.join(run_dir, f'confusion_matrix_task_{task_id:02d}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logging.info(f'[TEST] Da luu Confusion Matrix tai: {save_path}')
