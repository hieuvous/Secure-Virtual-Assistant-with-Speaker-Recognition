# Secure Virtual Assistant with Speaker Recognition

Đồ án cuối kỳ: trợ lý ảo tiếng Việt có tương tác bằng giọng nói, Speaker Verification (SV) cho thao tác quan trọng và Speaker Identification (SID) để cá nhân hóa.

## Chức năng

- Ghi âm trên giao diện Streamlit, nhận dạng lời nói tiếng Việt bằng Faster-Whisper và đọc phản hồi bằng gTTS.
- Phân tích câu lệnh theo luật (rule-based intent/entity routing).
- Chức năng chung không cần xác thực, ví dụ hỏi thông tin hoặc tạo nhắc việc.
- Chức năng quan trọng chỉ thực thi sau khi SV thành công.
- Chức năng cá nhân hóa theo người nói đã đăng ký qua SID.
- Enrollment và quản lý người dùng: thu 5 câu nói, tạo embedding trung bình đã L2-normalize và lưu profile vào Supabase.

## Mô hình và dữ liệu

- **Mô hình:** ECAPA-TDNN của SpeechBrain, khởi tạo từ `speechbrain/spkrec-ecapa-voxceleb`.
- **Fine-tuning dataset:** Vietnam-Celeb, chỉ sử dụng danh sách utterance train chính thức.
- **Checkpoint phát hành:** `models/ecapa_vietnamceleb_epoch10.pt` (Epoch 10, embedding 192 chiều).
- **VAD:** `speechbrain/vad-crdnn-libriparty`.
- **SV:** cosine similarity với ngưỡng DEV all-impostor `0.1566438227891922`.

### Split fine-tuning đã dùng

| Phần dữ liệu | Số speaker | Số utterance |
| --- | ---: | ---: |
| Train | 600 | 9,099 |
| Validation | 600 | 1,077 |
| Tổng trước split train/validation | 600 | 10,176 |
| DEV speaker-disjoint | 50 | 688 |

Các số liệu này được lưu trong output của `notebooks/finetune_ecapa_colab.ipynb`. DEV và TEST không chồng speaker với tập fine-tuning.

### Kết quả SV phát hành

| Model | DEV EER | TEST EER |
| --- | ---: | ---: |
| Pretrained ECAPA | 13.30% | 11.91% |
| Fine-tuned Epoch 10 | 9.98% | 8.42% |

Kết quả TEST được đánh giá với 50 speaker chưa xuất hiện trong dữ liệu fine-tuning.

## Kiến trúc

```text
Microphone / audio input
        |
        +--> Faster-Whisper ASR --> Rule-based router --> action type
        |                                               |
        |                    +--------------------------+-------------------------+
        |                    |                            |                         |
        |                 General                       SID                     Sensitive
        |                    |                            |                         |
        |                 execute               identify speaker              verify speaker
        |                    |                            |                         |
        +--------------------+----------------------------+-------------------------+
                                                     |
                                           response + gTTS output
```

Supabase lưu user, speaker profile embedding, task/note/reminder và audit data. Biến môi trường được nạp từ `.env`; tuyệt đối không commit file này.

## Cài đặt

Yêu cầu: Python 3.10 hoặc mới hơn, microphone và Internet ở lần chạy đầu để tải model SpeechBrain/Faster-Whisper khi chưa có cache.

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install "gTTS>=2.5,<3"
```

`gTTS` hiện được dùng bởi `src/speech/tts.py`; cần cài thêm nếu chưa được bổ sung vào `requirements.txt`.

Tạo `.env` từ `.env.example` và điền cấu hình Supabase:

```env
DATABASE_BACKEND=supabase
SUPABASE_URL=https://<project-ref>.supabase.co
SUPABASE_SECRET_KEY=<secret-key>
```

Áp dụng schema trong `supabase/schema.sql` bằng Supabase SQL Editor trước khi chạy ứng dụng.

## Chạy và kiểm tra

```powershell
python scripts/final_system_check.py
pytest -q
python scripts/validate_release_artifacts.py
python scripts/smoke_test_ecapa.py path\to\audio.wav
streamlit run app/main.py
```

Khi chạy lần đầu, SpeechBrain có thể tạo cache tại `models/pretrained_ecapa/` và `models/vad_crdnn/`. Các cache này có thể xóa và tải lại.

## Quy trình enrollment và xác thực

1. Tạo hoặc chọn người dùng trên tab Enrollment.
2. Thu 5 audio enrollment qua giao diện.
3. Audio được chuẩn hóa/VAD, trích xuất embedding ECAPA và gộp bằng normalized mean embedding.
4. Profile 192-D được lưu trong Supabase.
5. Với yêu cầu personal, SID so sánh embedding query với các profile đã đăng ký.
6. Với yêu cầu sensitive, SV so sánh query với profile mục tiêu; action chỉ chạy khi score vượt ngưỡng cố định từ DEV.

## Cấu trúc thư mục

```text
app/          Streamlit UI
src/          ASR, TTS, SV/SID, router, actions và Supabase client
training/     Chuẩn bị subset, fine-tuning, evaluation
models/       Checkpoint Epoch 10 và cache model cục bộ
results/      Kết quả calibration và test
notebooks/    EDA và notebook fine-tuning cuối cùng
supabase/     Schema và migration
tests/        Unit tests
docs/         Kiến trúc, bảng thí nghiệm, kịch bản demo và report outline
```

## Tái lập fine-tuning

Dataset Vietnam-Celeb không được đưa vào repository. Chuẩn bị audio dataset và danh sách `vietnam-celeb-t.txt`, sau đó dùng:

```powershell
python training/prepare_vietnam_celeb_subset.py `
  --data-root <data-root> `
  --official-train-list <vietnam-celeb-t.txt> `
  --output-dir metadata_v2 `
  --train-speakers 600 `
  --dev-speakers 50 `
  --max-utts 20 `
  --seed 42
```

Notebook `notebooks/finetune_ecapa_colab.ipynb` là artifact thí nghiệm cuối cùng; nó chứa output split và lịch sử train/validation Epoch 1--10. Không dùng Vietnam-Celeb-E/H để chọn threshold.

## Tài liệu bổ sung

- `docs/ARCHITECTURE.md`: kiến trúc chi tiết.
- `docs/EXPERIMENT_TABLES.md`: bảng dùng trong báo cáo.
- `docs/DEMO_SCRIPT.md`: kịch bản demo 5--7 phút.
- `docs/REPORT_OUTLINE.md`: dàn ý báo cáo.
- `requirements-experiments.txt`: dependency tùy chọn cho thí nghiệm phoneme-based enrollment, không cần cho ứng dụng lõi.

## Đóng gói nộp bài

Đưa source code, report, checkpoint và tài liệu cần thiết vào một ZIP có tên theo mã số sinh viên. Nếu dataset hoặc checkpoint quá lớn, tải lên Google Drive và nộp file `.txt` chứa link theo đúng định dạng giảng viên yêu cầu.

Không đưa `.env`, `.venv/`, `__pycache__/`, `data/runtime/` hoặc cache model sinh ra khi chạy vào ZIP.
