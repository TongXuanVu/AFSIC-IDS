import sys
import logging
import copy
import csv
import os
import glob
from datetime import datetime

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import count_parameters
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
        dm = DataManager(args["dataset"], args["shuffle"], args["seed"], args["init_cls"], args["increment"], client_id=c)
        if args.get("debug"):
            dm._train_data = dm._train_data[:2000]
            dm._train_targets = dm._train_targets[:2000]
        client_dms.append(dm)

    nb_tasks = client_dms[0].nb_tasks

    global_model = factory.get_model(args["model_name"], args)
    local_models = [factory.get_model(args["model_name"], args) for _ in range(args["num_clients"])]

    start_task = 0
    start_round = 0
    results_all = []

    if args.get("resume") and os.path.isfile(args["resume"]):
        logging.info(f"==> Resuming from checkpoint: {args['resume']}")
        checkpoint = torch.load(args["resume"], map_location='cpu', weights_only=False)
        start_task = checkpoint['task']
        start_round = checkpoint['round'] + 1
        if start_round >= args["num_rounds"]:
            start_task += 1
            start_round = 0
            
    for task in range(nb_tasks):
        # 1. Mở rộng kiến trúc (nhưng không train) để lấy đúng kích thước mô hình
        global_model.incremental_train(client_dms[0], skip_train=True)
        global_model._network.to(args["device"][0])

        for c in range(args["num_clients"]):
            local_models[c].skip_rehearsal = True
            local_models[c].incremental_train(client_dms[c], skip_train=True)
            local_models[c].skip_rehearsal = False

            train_dataset = client_dms[c].get_dataset(
                np.arange(local_models[c]._known_classes, local_models[c]._total_classes),
                source="train", mode="train", appendent=local_models[c]._get_memory(),
            )
            if len(train_dataset) > 0:
                local_models[c].train_loader = torch.utils.data.DataLoader(
                    train_dataset, batch_size=args["batch_size"], shuffle=True, num_workers=0
                )
            else:
                local_models[c].train_loader = None

        if task < start_task:
            continue

        if task == start_task and args.get("resume"):
            logging.info(f"Phục hồi trạng thái cho Task {task} từ Checkpoint...")
            global_model._network.load_state_dict(checkpoint['model_state_dict'])
            for c in range(args["num_clients"]):
                c_state = checkpoint['client_states'][c]
                local_models[c]._data_memory = c_state['data_memory']
                local_models[c]._targets_memory = c_state['targets_memory']
                
                # Tạo lại loader sau khi có memory
                train_dataset = client_dms[c].get_dataset(
                    np.arange(local_models[c]._known_classes, local_models[c]._total_classes),
                    source="train", mode="train", appendent=local_models[c]._get_memory(),
                )
                if len(train_dataset) > 0:
                    local_models[c].train_loader = torch.utils.data.DataLoader(
                        train_dataset, batch_size=args["batch_size"], shuffle=True, num_workers=0
                    )

        logging.info(f"========== Bắt đầu Task {task} ==========")
        current_start_round = start_round if task == start_task else 0

        for round_idx in range(current_start_round, args["num_rounds"]):
            global_round = task * args["num_rounds"] + round_idx
            logging.info(f"--- Task {task} | Round {round_idx+1}/{args['num_rounds']} (Global {global_round+1}) ---")
            client_weights = []
            
            for c in range(args["num_clients"]):
                if local_models[c].train_loader is None: continue
                local_models[c]._network.load_state_dict(global_model._network.state_dict())
                local_models[c]._network.to(args["device"][0])
                local_models[c].args["epochs"] = args["local_epochs"]
                local_models[c].args["start_round"] = 0
                local_models[c]._train(local_models[c].train_loader, None)
                client_weights.append(copy.deepcopy(local_models[c]._network.state_dict()))
            
            if client_weights:
                global_weights = average_weights(client_weights)
                global_model._network.load_state_dict(global_weights)

            # ── Đánh giá Global Model cuối MỖI ROUND ──
            test_dataset = client_dms[0].get_dataset(
                np.arange(0, global_model._total_classes), source="test", mode="test"
            )
            global_model.test_loader = torch.utils.data.DataLoader(
                test_dataset, batch_size=args["batch_size"], shuffle=False, num_workers=0
            )
            
            cnn_accy, nme_accy, y_pred, y_true = global_model.eval_task()
            
            results_all.append(cnn_accy)
            avg_acc = sum(r['top1'] for r in results_all) / len(results_all)
            
            logging.info(
                f"[Task {task} | Round {round_idx+1}] "
                f"Acc: {cnn_accy['top1']:.2f}% | F1-Mac: {cnn_accy.get('f1_macro', 0):.2f}% | Loss: {cnn_accy.get('loss', 0):.4f}"
            )

            # Ghi file CSV
            csv_writer.writerow([
                task, round_idx + 1, global_round + 1, "SPCIL-FL",
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

            # Lưu Checkpoint mỗi Round
            client_states = []
            for c in range(args["num_clients"]):
                client_states.append({
                    'data_memory': local_models[c]._data_memory,
                    'targets_memory': local_models[c]._targets_memory
                })
            ckpt_name = f'ckpt_round{global_round+1:04d}_task{task:02d}_r{round_idx+1:03d}_acc{cnn_accy["top1"]:.1f}.pth'
            torch.save({
                'task': task,
                'round': round_idx,
                'global_round': global_round,
                'model_state_dict': global_model._network.state_dict(),
                'known_classes': global_model._known_classes,
                'client_states': client_states,
                'metrics': cnn_accy
            }, os.path.join(ckpt_dir, ckpt_name))

        # Cuối Task, xây dựng lại bộ nhớ Rehearsal
        logging.info(f"Xây dựng Rehearsal Memory cho các Clients tại cuối Task {task}...")
        for c in range(args["num_clients"]):
            if local_models[c].train_loader is not None:
                local_models[c]._network.load_state_dict(global_model._network.state_dict())
                local_models[c]._network.to(args["device"][0])
                try:
                    local_models[c].build_rehearsal_memory(client_dms[c], local_models[c].samples_per_class)
                except Exception as e:
                    logging.warning(f"Lỗi khi build memory cho client {c}: {e}")
            local_models[c].after_task()

        global_model.after_task()

    csv_file.close()
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
    dm = DataManager(args["dataset"], False, args["seed"], args["init_cls"], args["increment"], client_id=0)
    
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
            
            test_dataset = dm.get_dataset(
                np.arange(0, global_model._total_classes), source="test", mode="test"
            )
            global_model.test_loader = torch.utils.data.DataLoader(
                test_dataset, batch_size=args["batch_size"], shuffle=False, num_workers=0
            )
            
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
                    cm = confusion_matrix(y_true, y_pred.T[0])
                    plt.figure(figsize=(12, 10))
                    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues')
                    plt.xlabel('Predicted')
                    plt.ylabel('Actual')
                    plt.title(f'Confusion Matrix - {os.path.basename(cp)}')
                    plt.savefig(os.path.join(test_ckpt_root, 'test_spcil_final_cm.png'), dpi=150)
                    plt.close()
                    logging.info(f"[TEST] Saved Confusion Matrix to {test_ckpt_root}")
                except Exception as e:
                    logging.error(f"[TEST] Lỗi khi vẽ Confusion Matrix: {e}")

    logging.info(f"[TEST] Hoàn thành. Kết quả được lưu tại: {csv_path}")


def _set_device(args):
    device_type = args["device"]
    gpus = []
    for device in device_type:
        if str(device) == "-1":
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
