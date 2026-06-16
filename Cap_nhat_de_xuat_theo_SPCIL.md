# Cập nhật đề xuất theo SPCIL

## 1. Bài toán

Giả sử có (K) tổ chức/client IDS, ký hiệu là C = {C_1, ..., C_K}. Mỗi client giữ dữ liệu mạng cục bộ:

```
D_i^t = D_{i,old}^t ∪ D_{i,new}^t
```

ở incremental stage (t). Tập lớp đã biết đến thời điểm (t) là:

```
Y^t = Y^0 ∪ Y^1 ∪ ... ∪ Y^t
```

Mục tiêu là học mô hình IDS toàn cục M^t có khả năng:

```
M^t: x → y,  y ∈ Y^t ∪ {unknown}
```

trong đó mô hình phải nhận diện tốt lớp cũ, học nhanh lớp tấn công mới chỉ với rất ít mẫu, không cần thu thập dữ liệu thô từ client, và hạn chế catastrophic forgetting.

**Điểm khác biệt quan trọng so với SPCIL gốc:** trong IDS liên bang, lớp mới có thể xuất hiện không đồng đều giữa các client. Một client có thể thấy tấn công DDoS mới, client khác chỉ thấy scan hoặc traffic benign. Vì vậy, nếu áp dụng SPCIL trực tiếp rồi FedAvg đơn giản, mô hình rất dễ bị lệch về client có nhiều mẫu hoặc bị quên lớp hiếm.

---

## 2. Tổng quan phương pháp AFSIC-IDS

AFSIC-IDS gồm sáu thành phần chính:

- **Federated base training** cho các lớp IDS nền.
- **Few-shot incremental adaptation** khi chỉ có vài mẫu lớp mới.
- **Dual-branch stability–plasticity encoder** lấy cảm hứng từ SPCIL.
- **Prototype-assisted classifier fusion** để thêm lớp mới mà không làm lệch classifier.
- **Adaptive robust aggregation** để tổng hợp cập nhật từ client theo chất lượng dữ liệu, độ tin cậy và mức độ tương thích prototype.

---

## 3. Kiến trúc mô hình

### 3.1. Encoder ổn định và encoder thích nghi

Thay vì mở rộng toàn bộ backbone như SPCIL gốc, AFSIC-IDS dùng kiến trúc nhẹ hơn:

```
h_i^t(x) = Fuse(Φ^{t-1}(x), A_i^t(x))
```

Trong đó:

- **Φ^{t-1}**: stability encoder, được học từ các stage trước và bị đóng băng một phần để giữ tri thức lớp cũ.
- **A_i^t**: plasticity adapter, là nhánh nhỏ được huấn luyện cho lớp mới tại client (i).
- **Fuse**: phép ghép đặc trưng hoặc gated fusion.

SPCIL gốc mở rộng feature extractor theo từng incremental step và đóng băng feature extractor cũ để giảm quên lãng. Trong AFSIC-IDS, ta giữ tinh thần này nhưng dùng **adapter/bottleneck/LoRA-style branch** thay vì nhân bản backbone lớn, vì trong FL nếu mở rộng backbone đầy đủ thì chi phí truyền thông và số tham số sẽ tăng rất nhanh.

Công thức fusion:

```
z_i^t(x) = g_i^t(x) · Φ^{t-1}(x) + (1 - g_i^t(x)) · A_i^t(x)
```

với g_i^t(x) ∈ [0,1] là adaptive gate. Nếu mẫu giống lớp cũ, gate ưu tiên Φ^{t-1}. Nếu mẫu giống lớp mới, gate ưu tiên adapter A_i^t.

### 3.2. Classifier động theo prototype

Thay vì chỉ mở rộng classifier bằng trọng số ngẫu nhiên, AFSIC-IDS khởi tạo lớp mới bằng **class prototype**.

Với mỗi lớp (c), client (i) tính prototype:

```
p_{i,c}^t = (1 / |D_{i,c}^t|) · Σ_{x ∈ D_{i,c}^t}  h_i^t(x) / |h_i^t(x)|
```

Server tổng hợp prototype toàn cục:

```
p_c^t = Σ_{i ∈ S_c^t}  α_{i,c}^t · p_{i,c}^t
```

Trong đó S_c^t là tập client có mẫu của lớp (c), còn α_{i,c}^t là trọng số theo số mẫu, chất lượng mẫu và độ tin cậy client.

Classifier dùng cosine classifier:

```
P(y=c|x) = exp(τ · cos(z_i^t(x), w_c)) / Σ_{c' ∈ Y^t} exp(τ · cos(z_i^t(x), w_{c'}))
```

Với lớp mới, w_c được khởi tạo từ prototype p_c^t. Cách này phù hợp với few-shot hơn so với khởi tạo ngẫu nhiên, vì vài mẫu mới vẫn đủ để tạo trung tâm đặc trưng ban đầu.

---

## 5. Bộ nhớ cục bộ và replay riêng tư

SPCIL dùng prestorage memory để giữ một phần dữ liệu cũ cho huấn luyện incremental. Trong FL, lưu dữ liệu thô có thể nhạy cảm. Vì vậy AFSIC-IDS dùng hai tầng memory:

### Local exemplar memory

Mỗi client giữ tối đa (m) mẫu mỗi lớp:

```
M_{i,c}^t ⊂ D_{i,c}^t
```

Mẫu được chọn bằng herding hoặc diversity sampling:

```
x* = argmin_x | (1/k) Σ_{j=1}^{k} h(x_j) - p_{i,c} |
```

### Prototype memory

Server không nhận dữ liệu thô mà chỉ nhận:

```
{p_{i,c}, n_{i,c}, σ_{i,c}, q_{i,c}}
```

gồm prototype, số mẫu, độ phân tán và điểm chất lượng. Prototype memory được dùng để regularize mô hình mà không chia sẻ packet/flow gốc.

---

## 6. Hàm mất mát

SPCIL gốc dùng:

```
L = L_CE + λ_a · L_SP + λ_b · L_RS
```

trong đó SP loss hỗ trợ học đặc trưng phân biệt tốt hơn trong bối cảnh mất cân bằng hoặc ít mẫu, còn sparse loss giúp giảm độ phức tạp mô hình. AFSIC-IDS mở rộng loss này cho federated few-shot IDS:

```
L_i^t = L_CE + λ_KD · L_KD + λ_FSP · L_FSP + λ_proto · L_proto + λ_RS · L_RS + λ_prox · L_prox
```

### Cross-entropy có cân bằng lớp

```
L_CE = -Σ_{(x,y) ∈ B_i^t}  ω_y · log P(y|x)
```

Với ω_y lớn hơn cho lớp mới hoặc lớp hiếm. Điều này rất quan trọng vì trong IDS, lớp tấn công mới thường ít hơn rất nhiều so với benign traffic.

### Knowledge distillation để giữ lớp cũ

```
L_KD = T² · KL(σ(o^{t-1}(x) / T) | σ(o^t(x) / T))
```

Loss này buộc mô hình mới không thay đổi quá mạnh đầu ra đối với lớp cũ.

### Few-shot sparse pairwise loss

Với mỗi lớp (c), chọn positive pair (x_a, x_p) cùng lớp và negative pair (x_a, x_n) khác lớp:

```
L_FSP = (1 / |Y^t|) · Σ_c  log(1 + exp((s^-_c - s^+_c) / T))
```

Trong đó:

```
s^+_c = cos(z(x_a), z(x_p)),   s^-_c = cos(z(x_a), z(x_n))
```

**Điểm chỉnh sửa so với SPCIL:** khi lớp mới chỉ có 1–5 mẫu, không đủ pair thật. Vì vậy AFSIC-IDS tạo pair với **global prototypes**:

```
s^+_c = cos(z(x), p_c),   s^-_c = max_{c' ≠ c} cos(z(x), p_{c'})
```

Nhờ vậy, vài mẫu mới vẫn có thể học biên phân tách với lớp cũ.

### Prototype alignment loss

```
L_proto = Σ_{(x,y) ∈ B_i^t}  |z_i^t(x) - p_y^t|²
```

Loss này giúp giảm client drift trong non-IID FL.

### Sparse regularization

```
L_RS = |m_i^t|_1 + |A_i^t|_1
```

Trong đó m_i^t là channel mask hoặc adapter mask. Mục tiêu là giữ nhánh incremental nhỏ, tránh mô hình phình to sau nhiều stage.

### FedProx-style regularization

```
L_prox = |θ_i^t - θ_G^{t-1}|²
```

Thành phần này làm giảm sự lệch mô hình cục bộ khi dữ liệu giữa các client khác nhau mạnh.

---

## 7. Federated adaptive aggregation

Sau khi client huấn luyện local adapter và classifier cho lớp mới, server không nên FedAvg đơn giản. Server tính điểm tin cậy:

```
Q_i^t = β_1 · Acc_{i,val} + β_2 · ProtoCons_i + β_3 · Novelty_i - β_4 · Drift_i - β_5 · UpdateNorm_i
```

Trong đó:

- **Acc_{i,val}**: hiệu năng trên local validation.
- **ProtoCons_i**: prototype của client có gần prototype toàn cục không.
- **Novelty_i**: mức độ rõ ràng của lớp mới.
- **Drift_i**: mức lệch phân phối so với global model.
- **UpdateNorm_i**: chuẩn update, dùng để phát hiện update bất thường.

Trọng số tổng hợp:

```
α_i^t = exp(Q_i^t / τ) / Σ_j exp(Q_j^t / τ)
```

Server cập nhật:

```
θ_G^t = Σ_{i ∈ S^t}  α_i^t · θ_i^t
```

Nhưng chỉ tổng hợp các phần cần thiết:

- adapter A_i^t,
- classifier weights cho lớp mới,
- prototype,
- mask/sparsity parameters.

Backbone ổn định Φ^{t-1} chỉ được cập nhật khi có đủ bằng chứng rằng cập nhật không làm tăng forgetting.

---

## 8. Quy trình huấn luyện đầy đủ

### Giai đoạn 0: Base federated training

Server khởi tạo mô hình M^0 = (Φ^0, C^0). Các client huấn luyện trên các lớp IDS nền như benign, DoS, scan, brute force, botnet. Server tổng hợp bằng adaptive FedAvg hoặc FedProx. Kết quả là global base model.

### Giai đoạn 1: Local unknown detection

Mỗi client chạy IDS. Các mẫu có confidence thấp hoặc xa prototype được đưa vào unknown buffer. Client gom cụm và yêu cầu xác nhận nhãn nếu cần.

### Giai đoạn 2: Few-shot class registration

Khi một lớp mới có (K)-shot mẫu được xác nhận, client gửi prototype, thống kê lớp và metadata bảo mật lên server. Server kiểm tra xem lớp mới này là lớp thật, lớp trùng với lớp đã có, hay chỉ là phân phối lệch.

### Giai đoạn 3: Incremental federated training

Server phát hành incremental task (t), gồm danh sách lớp mới, prototype ban đầu và adapter template. Mỗi client huấn luyện:

- freeze stability encoder,
- train plasticity adapter,
- replay local exemplars/prototypes lớp cũ,
- tối ưu loss tổng hợp.

### Giai đoạn 4: Robust aggregation

Server lọc update bất thường, tổng hợp adapter và classifier bằng trọng số chất lượng, sau đó cập nhật global prototype memory.

### Giai đoạn 5: Calibration

Server hoặc client hiệu chỉnh classifier để giảm thiên lệch về lớp mới. Có thể dùng balanced fine-tuning với exemplar/prototype replay.

### Giai đoạn 6: Deployment

Client nhận M^t, dùng để phân loại cả lớp cũ, lớp mới và unknown traffic. Quá trình lặp lại khi phát hiện lớp mới tiếp theo.

---

## 9. Pseudocode

```
Algorithm: AFSIC-IDS

Input:
    Clients C1,...,CK
    Initial label set Y0
    Incremental stages t = 1,...,T
    Local memory budget m
    Few-shot threshold Kshot

Stage 0: Base Federated Training
    Server initializes global model M0 = (Phi0, C0)
    for each communication round r:
        Server broadcasts Mr
        each client Ci trains on local base data Di0
        client sends model update and class prototypes
        server performs adaptive robust aggregation
    Save global base model M0 and global prototypes P0

For each incremental stage t:
    For each client Ci:
        Run inference on streaming traffic
        Put low-confidence or far-from-prototype samples into unknown buffer
        Cluster unknown samples
        If a cluster has at least Kshot confirmed samples:
            Register candidate new class y_new
            Compute local prototype p_i,y_new

    Server:
        Merge compatible candidate classes from clients
        Create new label set Yt
        Broadcast adapter template At, old model M^{t-1}, and prototypes P^{t-1}

    For each selected client Ci:
        Freeze stability encoder Phi^{t-1}
        Initialize plasticity adapter Ai^t
        Construct training batch:
            few-shot new samples
            local exemplars of old classes
            prototype replay samples
        Train Ai^t and expanded classifier Ci^t using:
            CE + KD + FSP + Proto + Sparse + Prox loss
        Send adapter update, classifier update, mask, and prototypes

    Server:
        Detect abnormal updates
        Aggregate accepted updates using quality-aware weights
        Update global model Mt
        Update global prototype memory Pt
        Calibrate classifier

Output:
    Adaptive federated IDS model MT
```

---

## 10. Đóng góp khoa học có thể viết trong paper

Có thể định vị phương pháp này với các đóng góp sau:

**Thứ nhất**, đề xuất một framework **federated few-shot class-incremental IDS** cho bối cảnh lớp tấn công mới xuất hiện không đồng đều giữa các client.

**Thứ hai**, mở rộng ý tưởng stability–plasticity của SPCIL thành kiến trúc **frozen global stability encoder + lightweight local/global plasticity adapters**, giúp học lớp mới mà không phải mở rộng toàn bộ backbone.

**Thứ ba**, đề xuất **prototype-assisted few-shot sparse pairwise loss**, cho phép học ranh giới lớp mới ngay cả khi chỉ có vài mẫu.

**Thứ tư**, xây dựng cơ chế **adaptive robust aggregation** dựa trên chất lượng update, độ nhất quán prototype và mức độ drift, phù hợp hơn FedAvg trong IDS non-IID.

**Thứ năm**, dùng **privacy-aware memory** bằng exemplar cục bộ và prototype toàn cục thay vì chia sẻ dữ liệu traffic thô.

---

## 11. Thiết kế thí nghiệm

Nên đánh giá trên nhiều bối cảnh, không chỉ một dataset.

### Dataset

Có thể dùng:

- CICIDS2017
- CSE-CIC-IDS2018
- UNSW-NB15
- Bot-IoT
- ToN-IoT
- USTC-TFC2016 nếu muốn so sánh gần với SPCIL

### Cách chia incremental

Ví dụ:

- Base classes: benign, DoS, scan.
- Increment 1: brute force.
- Increment 2: botnet.
- Increment 3: infiltration.
- Increment 4: web attack.
- Increment 5: zero-day-like unseen attack.

Few-shot setting: K ∈ {1, 5, 10, 20}

Non-IID federation:

- label-skew: mỗi client chỉ có một số lớp.
- quantity-skew: client có số mẫu khác nhau.
- temporal-skew: lớp mới xuất hiện ở client khác nhau tại thời điểm khác nhau.
- feature-skew: tổ chức khác nhau có network behavior khác nhau.

### Baselines

Nên so sánh với:

- Local-only IDS
- Centralized upper bound
- FedAvg
- FedProx
- FedProto
- FedAvg + fine-tuning
- FedAvg + LwF
- FedAvg + iCaRL-style replay
- Standalone SPCIL
- FOSTER/DER/SimpleCIL nếu triển khai được trong setting non-federated

### Metrics

Không nên chỉ báo cáo accuracy. Với IDS, cần:

```
F1,  Recall_attack,  FNR,  FPR,  AA,  OA
```

và các continual learning metrics:

```
Forgetting,  Backward Transfer,  Intransigence
```

Thêm metrics hệ thống:

- số tham số tăng sau mỗi increment,
- communication cost mỗi round,
- thời gian huấn luyện local,
- memory budget mỗi client,
- hiệu năng trên lớp mới sau K-shot,
- hiệu năng lớp cũ sau nhiều increment.

---

## 12. Điểm cần nói thẳng để phương pháp vững hơn

Có bốn rủi ro lớn nếu viết phương pháp này thành paper.

**Thứ nhất**, **few-shot label trong IDS không tự nhiên có sẵn**. Nếu không có analyst, sandbox, threat intelligence hoặc rule xác nhận, "new class" có thể chỉ là unknown cluster, không phải lớp tấn công thật.

**Thứ hai**, **SPCIL gốc dùng setting CIL tập trung**, còn FL có non-IID, client drift, poisoning và privacy leakage. Vì vậy đóng góp không nên viết là "áp dụng SPCIL vào FL", mà nên viết là "mở rộng stability–plasticity incremental learning sang federated few-shot IDS bằng prototype, adapter và adaptive aggregation".

**Thứ ba**, **FL không đồng nghĩa với privacy-preserving**. Nếu paper nói privacy, cần có secure aggregation, differential privacy, hoặc ít nhất phân tích gradient/prototype leakage. Nếu không, nên nói là "raw-data locality".

Tóm lại, phương pháp nên được xây dựng như một hệ thống **federated continual IDS**: SPCIL cung cấp hạt nhân stability–plasticity và sparse pairwise learning, còn phần mới nằm ở **few-shot prototype learning, adaptive federated aggregation, privacy-aware memory, và open-set new attack discovery**.
