import sys
import logging
import copy
import csv
import os
from datetime import datetime

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import count_parameters
from sklearn.metrics import confusion_matrix
import seaborn as sns


def train(args):
    seed_list = copy.deepcopy(args["seed"])
    device = copy.deepcopy(args["device"])

    for seed in seed_list:
        args["seed"] = seed
        args["device"] = device
        _train(args)


def _train(args):
    init_cls = 0 if args["init_cls"] == args["increment"] else args["init_cls"]

    # ── Tạo thư mục theo timestamp chuẩn: dd-mm-yy_HH-MM ───────────────────
    timestamp = datetime.now().strftime("%d-%m-%y_%H-%M")
    run_dir = os.path.join(
        "logs",
        args["model_name"],
        args["dataset"],
        "{}_seed{}_{}".format(
            timestamp,
            args["seed"],
            args["convnet_type"],
        ),
    )
    os.makedirs(run_dir, exist_ok=True)

    # ── Logging ─────────────────────────────────────────────────────────────
    logfilename = os.path.join(run_dir, "training.log")

    # Xóa handlers cũ để tránh duplicate khi chạy nhiều seed
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            logging.FileHandler(filename=logfilename),
            logging.StreamHandler(sys.stdout),
        ],
    )

    # ── CSV để lưu số liệu theo từng task ───────────────────────────────────
    csv_path = os.path.join(run_dir, "metrics.csv")
    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "task", "method",
        "acc", "prec_mic", "prec_mac", "prec_wei",
        "rec_mic", "rec_mac", "rec_wei",
        "f1_mic", "f1_mac", "f1_wei",
        "loss", "avg_acc",
    ])

    _set_random()
    _set_device(args)
    print_args(args)
    logging.info("Run directory: {}".format(run_dir))
    args["run_dir"] = run_dir

    data_manager = DataManager(
        args["dataset"],
        args["shuffle"],
        args["seed"],
        args["init_cls"],
        args["increment"],
    )
    
    if args.get("debug"):
        logging.info("[DEBUG] Truncating training data to 50,000 samples for quick test.")
        data_manager._train_data = data_manager._train_data[:50000]
        data_manager._train_targets = data_manager._train_targets[:50000]

    model = factory.get_model(args["model_name"], args)

    # ── RESUME LOGIC ────────────────────────────────────────────────────────
    start_task = 0
    start_round = 0
    if args.get("resume"):
        logging.info(f"Resuming from checkpoint: {args['resume']}")
        checkpoint = torch.load(args["resume"], map_location=args["device"][0], weights_only=False)
        start_task = checkpoint.get("task", 0)
        start_round = checkpoint.get("round", 0)
        args["start_round"] = start_round  # Pass to model
        
        # Build model structure up to start_task (not including it)
        # The main loop will handle the start_task itself
        for t in range(start_task):
            model.incremental_train(data_manager, skip_train=True)
            model.after_task()
            
        # Load weights
        model._network.load_state_dict(checkpoint["model_state_dict"])
        model._known_classes = checkpoint.get("known_classes", 0)
        model._network.known_classes = model._known_classes # Sync with network
        
        logging.info(f"Resumed successfully at Task {start_task}, Round {start_round}")

    # ── Lịch sử metrics để vẽ biểu đồ ──────────────────────────────────────
    history = {
        "cnn":  {"acc": [], "precision": [], "recall": [], "f1": []},
        "nme":  {"acc": [], "precision": [], "recall": [], "f1": []},
    }
    cnn_curve, nme_curve = {"top1": [], "top5": []}, {"top1": [], "top5": []}

    for task in range(start_task, data_manager.nb_tasks):
        logging.info("All params: {}".format(count_parameters(model._network)))
        logging.info("Trainable params: {}".format(count_parameters(model._network, True)))

        # Always call incremental_train. 
        # If it's the start_task, it will use the start_round passed in args.
        model.incremental_train(data_manager)

        cnn_accy, nme_accy = model.eval_task()
        model.after_task()

        # ── Luu checkpoint sau moi task ────────────────────────────────────────
        ckpt_dir = os.path.join(run_dir, 'checkpoints')
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_name = (
            f'ckpt_task{task:02d}_acc{round(cnn_accy["top1"], 1)}.pth'
        )
        torch.save({
            'task':             task,
            'model_state_dict': model._network.state_dict(),
            'known_classes':    model._known_classes,
            'metrics': {
                'acc':       cnn_accy['top1'],
                'prec_mic':  cnn_accy.get('precision_micro', 0),
                'prec_mac':  cnn_accy.get('precision_macro', 0),
                'prec_wei':  cnn_accy.get('precision_weighted', 0),
                'rec_mic':   cnn_accy.get('recall_micro', 0),
                'rec_mac':   cnn_accy.get('recall_macro', 0),
                'rec_wei':   cnn_accy.get('recall_weighted', 0),
                'f1_mic':    cnn_accy.get('f1_micro', 0),
                'f1_mac':    cnn_accy.get('f1_macro', 0),
                'f1_wei':    cnn_accy.get('f1_weighted', 0),
                'loss':      cnn_accy.get('loss', 0),
            },
        }, os.path.join(ckpt_dir, ckpt_name))
        logging.info(f'[CKPT] Saved: {ckpt_name}')

        # ── CNN metrics ─────────────────────────────────────────────────────
        cnn_g = cnn_accy["grouped"]
        logging.info("CNN: {}".format(cnn_g))

        cnn_curve["top1"].append(cnn_accy["top1"])
        cnn_curve["top5"].append(cnn_accy["top5"])
        history["cnn"]["acc"].append(cnn_accy["top1"])
        history["cnn"]["precision"].append(cnn_accy.get("precision_macro", 0))
        history["cnn"]["recall"].append(cnn_accy.get("recall_macro", 0))
        history["cnn"]["f1"].append(cnn_accy.get("f1_macro", 0))

        avg_cnn = sum(cnn_curve["top1"]) / len(cnn_curve["top1"])
        csv_writer.writerow([
            task, "CNN",
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
            round(avg_cnn, 4),
        ])

        # ── NME metrics ─────────────────────────────────────────────────────
        if nme_accy is not None:
            nme_g = nme_accy["grouped"]
            logging.info("NME: {}".format(nme_g))

            nme_curve["top1"].append(nme_accy["top1"])
            nme_curve["top5"].append(nme_accy["top5"])
            history["nme"]["acc"].append(nme_accy["top1"])
            history["nme"]["precision"].append(nme_accy.get("precision_macro", 0))
            history["nme"]["recall"].append(nme_accy.get("recall_macro", 0))
            history["nme"]["f1"].append(nme_accy.get("f1_macro", 0))

            avg_nme = sum(nme_curve["top1"]) / len(nme_curve["top1"])
            csv_writer.writerow([
                task, "NME",
                round(nme_accy["top1"], 4),
                round(nme_accy.get("precision_micro", 0), 4),
                round(nme_accy.get("precision_macro", 0), 4),
                round(nme_accy.get("precision_weighted", 0), 4),
                round(nme_accy.get("recall_micro", 0), 4),
                round(nme_accy.get("recall_macro", 0), 4),
                round(nme_accy.get("recall_weighted", 0), 4),
                round(nme_accy.get("f1_micro", 0), 4),
                round(nme_accy.get("f1_macro", 0), 4),
                round(nme_accy.get("f1_weighted", 0), 4),
                round(nme_accy.get("loss", 0), 6),
                round(avg_nme, 4),
            ])

        # ── Log tổng kết sau mỗi task ────────────────────────────────────────
        logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
        logging.info("CNN top5 curve: {}".format(cnn_curve["top5"]))
        if nme_accy is not None:
            logging.info("NME top1 curve: {}".format(nme_curve["top1"]))
            logging.info("NME top5 curve: {}\n".format(nme_curve["top5"]))

        logging.info("Average Accuracy (CNN): {}".format(avg_cnn))
        if nme_accy is not None:
            logging.info("Average Accuracy (NME): {}\n".format(avg_nme))

    csv_file.close()
    logging.info("Metrics CSV saved: {}".format(csv_path))

    # ── Vẽ biểu đồ ──────────────────────────────────────────────────────────
    _plot_metrics(history, run_dir, args)
    logging.info("Plots saved in: {}".format(run_dir))


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


# =============================================================================
# TEST MODE — Phan code rieng biet chi de chay evaluation
# Cach su dung:
#   python main.py --config ./exps/cic_iot23_debug.json  (train)
#   args["mode"] = "test" trong json hoac truyen qua CLI (xem main.py)
# =============================================================================
def run_test(args):
    """
    Load tung checkpoint da luu trong run_dir/checkpoints/ va chay eval.
    Ket qua luu vao run_dir/test_results.csv.
    """
    import csv as _csv
    import glob
    from utils import factory
    from utils.data_manager import DataManager

    # Xac dinh thu muc checkpoint
    test_ckpt_root = args.get('test_checkpoint_dir', '') or args.get('run_dir', './')
    # Tim tat ca cac file .pth trong thu muc checkpoints
    ckpt_dir = os.path.join(test_ckpt_root, 'checkpoints')
    if not os.path.exists(ckpt_dir):
        # Thu lay truc tiep test_ckpt_root neu khong co subfolder checkpoints
        ckpt_dir = test_ckpt_root
    
    ckpt_files = sorted(glob.glob(os.path.join(ckpt_dir, '*.pth')))

    if not ckpt_files:
        logging.error(f'[TEST] Khong tim thay bat ky file .pth nào tai: {ckpt_dir}')
        return

    logging.info(f'[TEST] Tim thay {len(ckpt_files)} checkpoint(s)')

    # Khoi tao data manager va model (kien truc)
    data_manager = DataManager(
        args['dataset'], args['shuffle'], args['seed'],
        args['init_cls'], args['increment'],
    )
    model = factory.get_model(args['model_name'], args)

    # History for plotting
    accuracy_history = []
    prec_mic_history = []
    prec_mac_history = []
    prec_wei_history = []
    rec_mic_history = []
    rec_mac_history = []
    rec_wei_history = []
    f1_mic_history = []
    f1_mac_history = []
    f1_wei_history = []
    loss_history = []
    task_history = []

    out_csv = os.path.join(args.get('run_dir', '.'), 'test_results.csv')
    with open(out_csv, 'w', newline='', encoding='utf-8') as fcsv:
        writer = _csv.writer(fcsv)
        writer.writerow([
            'checkpoint', 'task', 'known_classes',
            'acc', 'prec_mic', 'prec_mac', 'prec_wei',
            'rec_mic', 'rec_mac', 'rec_wei',
            'f1_mic', 'f1_mac', 'f1_wei', 'loss',
        ])

        for _cp in ckpt_files:
            state = torch.load(_cp, map_location=args['device'][0], weights_only=False)
            task        = state['task']
            known_cls   = state.get('known_classes', -1)

            model._network.load_state_dict(state['model_state_dict'], strict=False)
            model._known_classes = known_cls
            model._network.eval()

            # Lay test loader cho task nay
            model._cur_task  = task
            model.test_loader = torch.utils.data.DataLoader(
                data_manager.get_dataset(
                    np.arange(0, known_cls if known_cls > 0 else args['init_cls']),
                    source='test', mode='test'
                ),
                batch_size=args.get('batch_size', 256), shuffle=False, num_workers=0
            )

            cnn_accy, nme_accy, y_pred, y_true_eval = model.eval_task()
            m = cnn_accy
            
            # Ve Confusion Matrix cho tung checkpoint
            plot_confusion_matrix(y_true_eval, y_pred, task, args.get('run_dir', '.'))
            logging.info(
                f'[TEST] {os.path.basename(_cp)} | Task {task} | '
                f"Acc: {m['top1']:.2f}% | F1-Mac: {m.get('f1_macro',0):.2f}% | F1-Mic: {m.get('f1_micro',0):.2f}%"
            )
            writer.writerow([
                os.path.basename(_cp), task, known_cls,
                round(m['top1'], 4),
                round(m.get('precision_micro', 0), 4),
                round(m.get('precision_macro', 0), 4),
                round(m.get('precision_weighted', 0), 4),
                round(m.get('recall_micro', 0), 4),
                round(m.get('recall_macro', 0), 4),
                round(m.get('recall_weighted', 0), 4),
                round(m.get('f1_micro', 0), 4),
                round(m.get('f1_macro', 0), 4),
                round(m.get('f1_weighted', 0), 4),
                round(m.get('loss', 0), 6),
            ])
            fcsv.flush()

            # Save to history for plotting
            accuracy_history.append(m['top1'])
            prec_mic_history.append(m.get('precision_micro', 0))
            prec_mac_history.append(m.get('precision_macro', 0))
            prec_wei_history.append(m.get('precision_weighted', 0))
            rec_mic_history.append(m.get('recall_micro', 0))
            rec_mac_history.append(m.get('recall_macro', 0))
            rec_wei_history.append(m.get('recall_weighted', 0))
            f1_mic_history.append(m.get('f1_micro', 0))
            f1_mac_history.append(m.get('f1_macro', 0))
            f1_wei_history.append(m.get('f1_weighted', 0))
            loss_history.append(m.get('loss', 0))
            task_history.append(task)

    logging.info(f'[TEST] Ket qua luu tai: {out_csv}')

    # --- Ve bieu do (Giong HFIN/MalCL) ---
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    def save_test_plot(x_vals, y_vals, metric_name, color, marker):
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

    def save_combined_plot(x_vals, y_mic, y_mac, y_wei, category_name):
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


    if accuracy_history:
        x_axis = task_history
        save_test_plot(x_axis, accuracy_history, 'Accuracy', 'b', 'o')
        save_test_plot(x_axis, loss_history, 'Loss', 'k', 'X')
        save_combined_plot(x_axis, prec_mic_history, prec_mac_history, prec_wei_history, 'Precision')
        save_combined_plot(x_axis, rec_mic_history, rec_mac_history, rec_wei_history, 'Recall')
        save_combined_plot(x_axis, f1_mic_history, f1_mac_history, f1_wei_history, 'F1-Score')
        logging.info(f'[TEST] Da ve bieu do don va ket hop vao: {args.get("run_dir", ".")}')


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
