# SPCIL-FL / AFSIC-IDS

Project này triển khai setting **supervised federated task-incremental learning** cho bài toán IDS trên dữ liệu CIC-IoT23 đã chia theo client/task.

Setting mặc định hiện tại:

- Federated learning nhiều client.
- Task-incremental supervised learning.
- Task split: `[6, 6, 6, 6, 5, 5]`.
- Unknown buffer/new attack discovery đang **tắt** trong config mặc định.
- Evaluation dùng `global_test_data.pt`, nhưng chỉ test trên các class đã học tới task hiện tại.

## 1. Cấu trúc thư mục

```text
.
├── main.py                         # Entry point train/test
├── trainer.py                      # Federated task-incremental training loop
├── configs/
│   └── exps/
│       └── cic_iot23_afsic_ids.json # Config mặc định hiện tại
├── models/                         # Learner/model implementations
├── utils/                          # Data manager, memory, prototype, buffer
├── convs/                          # Backbone/classifier layers
├── losses/                         # Losses kế thừa từ SPCIL
├── docs/                           # Proposal, notes, figures
├── tools/                          # Script phụ trợ
└── data/                           # Local data, không push lên GitHub
```

Xem thêm mô tả cấu trúc ở `PROJECT_STRUCTURE.md`.

## 2. Cài môi trường

Khuyến nghị dùng Python 3.10+.

```bash
pip install torch torchvision numpy scikit-learn scipy matplotlib seaborn pillow tqdm
```

Nếu chạy CPU, config mặc định đang dùng:

```json
"device": ["cpu"]
```

Nếu chạy GPU, sửa trong `configs/exps/cic_iot23_afsic_ids.json`, ví dụ:

```json
"device": [0]
```

## 3. Chuẩn bị dữ liệu

Code hiện tại expect dữ liệu ở:

```text
data/CIC_IoT23/
├── federated_data/
│   ├── client_0_task_1.pt
│   ├── client_0_task_2.pt
│   ├── ...
│   ├── client_4_task_6.pt
└── global_test_data.pt
```

Format `.pt` có thể là dict:

```python
{"x": tensor_features, "y": tensor_labels}
```

hoặc tuple:

```python
(x, y)
```

Trong đó:

- `x`: tensor shape `[N, 31]`.
- `y`: nhãn gốc CIC-IoT23.

Lưu ý: thư mục `data/` được ignore trong `.gitignore`, không commit dữ liệu lớn lên GitHub.

## 4. Setting task mặc định

Config mặc định nằm ở:

```text
configs/exps/cic_iot23_afsic_ids.json
```

Task split đang theo hình phân phối dữ liệu:

```json
"task_increments": [6, 6, 6, 6, 5, 5],
"class_order": [
  1, 0, 11, 12, 27, 26,
  2, 14, 25, 24, 20, 28,
  3, 7, 30, 29, 19, 16,
  15, 6, 8, 22, 23, 21,
  5, 13, 10, 17, 18,
  4, 31, 32, 33, 9
]
```

Unknown discovery đang tắt:

```json
"supervised_task_incremental": true,
"use_unknown_discovery": false,
"fewshot_enabled": false
```

Vì `fewshot_enabled=false`, mỗi task dùng toàn bộ labeled task data hiện có tại client.

## 5. Chạy nhanh debug

Debug mode giảm epoch/round để kiểm tra pipeline:

```bash
python main.py --debug
```

Hoặc chỉ rõ config:

```bash
python main.py --config ./configs/exps/cic_iot23_afsic_ids.json --debug
```

## 6. Chạy train đầy đủ

```bash
python main.py --config ./configs/exps/cic_iot23_afsic_ids.json
```

Một số tham số có thể override bằng CLI:

```bash
python main.py ^
  --config ./configs/exps/cic_iot23_afsic_ids.json ^
  --num_clients 5 ^
  --num_rounds 5 ^
  --local_epochs 2 ^
  --batch_size 4096
```

Trên Linux/macOS thay `^` bằng `\`.

## 7. Resume training

Sau khi train, checkpoint được lưu trong:

```text
logs/<model>_federated/<dataset>/<run_name>/checkpoints/
```

Resume từ checkpoint:

```bash
python main.py ^
  --config ./configs/exps/cic_iot23_afsic_ids.json ^
  --resume logs/afsic-ids_federated/cic_iot23/<run_name>/checkpoints/<checkpoint>.pth
```

## 8. Test checkpoint

Chạy evaluation trên toàn bộ checkpoint trong một run:

```bash
python main.py ^
  --mode test ^
  --config ./configs/exps/cic_iot23_afsic_ids.json ^
  --test_checkpoint_dir logs/afsic-ids_federated/cic_iot23/<run_name>
```

Evaluation dùng `global_test_data.pt`, nhưng chỉ lấy các class đã học tới checkpoint/task đó:

- Task 1: test class remapped `0..5`.
- Task 2: test class remapped `0..11`.
- ...
- Task 6: test class remapped `0..33`.

## 9. Output chính

Mỗi run tạo thư mục:

```text
logs/afsic-ids_federated/cic_iot23/<timestamp_seed_convnet_clients>/
```

Bên trong gồm:

- `training.log`: log train/eval.
- `metrics_round_by_round.csv`: metrics từng round.
- `checkpoints/`: checkpoint từng round.
- `metrics_plot.png`: biểu đồ metrics.
- `all_metrics_combined.png`: biểu đồ tổng hợp.
- `confusion_matrix_task_*.png`: confusion matrix theo task.

## 10. Kiến trúc hiện tại

Các thành phần chính:

- `models/afsic_ids.py`: learner AFSIC-IDS.
- `utils/inc_net.py`: network, frozen encoder, plastic adapter, cosine classifier.
- `utils/memory.py`: local exemplar memory và global prototype memory.
- `trainer.py`: FL rounds, prototype aggregation, robust aggregation, evaluation.

Workflow figure:

```text
docs/figures/simple_supervised_architecture_2clients.png
```

## 11. Bật unknown discovery sau này

Khi chuyển sang setting open-set/new attack discovery, sửa config:

```json
"use_unknown_discovery": true,
"fewshot_enabled": true,
"kshot": 10
```

Các ngưỡng unknown:

```json
"delta_p": 0.5,
"delta_d": 0.7
```

Điều kiện đưa mẫu vào unknown buffer:

```text
max_c P(y=c|x) < delta_p
and
min_c (1 - cos(z(x), p_c)) > delta_d
```

Trong setting hiện tại, nhánh này chưa dùng.

## 12. Ghi chú GitHub

Không commit các file lớn/local:

- `data/`
- `logs/`
- `checkpoints/`
- `*.pth`
- `*.csv`
- `*.png`
- `no-use/cache/`

Các mục này đã hoặc nên được ignore để repo nhẹ và dễ clone.
