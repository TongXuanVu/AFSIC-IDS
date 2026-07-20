# AFSIC-IDS — Brief (Chat 4) — ĐỀ XUẤT CHÍNH

## Mục tiêu
Adaptive Federated Few-Shot Class-Incremental Learning cho IDS. Mở rộng stability–plasticity
của SPCIL sang federated few-shot: lớp tấn công mới xuất hiện không đồng đều giữa client.
Xem thiết kế chi tiết trong `Cap_nhat_de_xuat_theo_SPCIL.md`.

## Dataset
CIC-IoT23. Cấu hình task/increment trong `configs/exps/cic_iot23_afsic.json`
(vd `task_increments: [6,6,6,6,5,5]`, `num_clients: 10`, `num_rounds: 30`).

## Kiến trúc & thành phần
- Frozen global stability encoder + lightweight plasticity adapter + vector gate.
- Prototype-assisted classifier (cosine), global prototype memory.
- Loss đa mục tiêu: CE + KD + FSP (few-shot sparse pairwise) + proto + RS + prox.
- **Adaptive robust aggregation** (`utils/aggregation.py::compute_aggregation_weights`):
  `Q_i = β_acc·acc + β_proto·proto_cons + β_novelty·novelty − β_drift·drift − β_update·update_norm`,
  lọc MAD z-score, `alpha = softmax(Q/tau)`. Đây là bản gốc mà LCwoF-FL đã copy sang.

## Entry & lệnh chạy
- Train: `python main.py --config configs/exps/cic_iot23_afsic.json`
- Test:  `python main.py --config configs/exps/cic_iot23_afsic.json --test`
- Debug nhanh: dùng `configs/exps/cic_iot23_debug.json`.
- Metrics ghi ra `metrics_round_by_round.csv`.

## File quan trọng
`trainer.py` (vòng FL, lời gọi aggregation ~dòng 400–513), `utils/aggregation.py`,
`utils/memory.py`, `utils/inc_net.py`, `models/der.py`, `losses/`.

## Trạng thái
Đang phát triển. Là "nguồn chuẩn" cho các quyết định aggregation ở các project khác.

## Lưu ý khi viết paper (từ đề xuất)
Few-shot label trong IDS không tự nhiên có sẵn; FL ≠ privacy nếu không có secure agg/DP;
định vị đóng góp là "mở rộng stability–plasticity sang federated few-shot IDS" chứ không phải "áp SPCIL vào FL".
