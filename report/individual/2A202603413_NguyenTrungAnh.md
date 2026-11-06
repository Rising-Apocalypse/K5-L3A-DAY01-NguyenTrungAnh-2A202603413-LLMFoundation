# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> Đổi tên file này thành `MSSV_HoTen.md` của bạn, ví dụ `2A20260000_NguyenVanAn.md`, rồi điền từng mục. Xoá các dòng hướng dẫn in nghiêng khi xong.

- **MSSV:**
- **Họ và tên:**
- **SEED** (in ở ô đầu notebook):

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☑ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☑ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☑ |

## 2. Kết quả chính

Bảng so sánh cuối Phần 5 (số liệu lấy từ `lab1_report.json`). Các hàng LLM chỉ chạy trên tập SUB (40 tin), nên so sánh công bằng bằng cột "F1 SUB 40".

| Cách | F1 test | F1 SUB 40 | p95 (ms/tin) | USD / 1k tin |
| --- | --- | --- | --- | --- |
| 1. BoW + LogReg (tự code) | 0.900 | 0.857 | 0.16 | 0 |
| 2. BoW + MLP (deep learning) | 0.952 | 0.974 | 2.39 | 0 |
| 3a. Encoder (MiniLM) + kNN | 0.931 | 0.947 | 126.53 | 0 |
| 3b. Encoder (MiniLM) + LogReg | 0.948 | 0.947 | 38.94 | 0 |
| 3c. Encoder (MiniLM) + MLP head | 0.920 | 0.919 | 35.94 | 0 |
| 4a. LLM nhỏ zero-shot (`gpt-6-luna`) | — | 1.000 | 2270.92 | 0.007 |
| 4b. LLM lớn zero-shot (`gpt-6.1-terra`) | — | 1.000 | 1382.55 | 0.073 |
| 4c. LLM nhỏ few-shot | — | 1.000 | 1492.42 | 0.016 |

Lưu ý: tập SUB chỉ có 40 tin, 1 tin sai làm F1 đổi khoảng 2.5 điểm %, nên các chênh lệch nhỏ (dưới ~0.05) coi như hoà.

## 3. Dự đoán và thực tế

**`llm_models`: model lớn và model nhỏ.**
- *Đoán:* model lớn F1 cao hơn nhỏ khoảng 0.02 (cả hai ~0.95); model lớn đắt gấp ~10 lần; LLM thắng Emb + LogReg nhưng không đáng kể.
- *Thực tế:* cả hai đều F1 = 1.000 (không chênh); giá lớn $0.073 so với nhỏ $0.007 mỗi 1k tin, đúng khoảng 10 lần. LLM hơn Emb + LogReg (0.947) khoảng 0.053, tương đương 2 tin trên 40.
- *Vì sao lệch:* tập SUB chỉ 40 tin và khá dễ nên cả hai chạm trần 1.000, không phân biệt được model lớn với nhỏ. Mình mặc định model lớn phải tốt hơn.

**`llm_fewshot`: thêm 4 ví dụ vào prompt.**
- *Đoán:* F1 tăng, chi phí tăng gấp đôi.
- *Thực tế:* F1 vẫn 1.000 (không tăng), chi phí $0.016 so với $0.007 mỗi 1k tin, tức khoảng 2.3 lần.
- *Vì sao lệch:* zero-shot đã đạt trần nên few-shot không còn chỗ để cải thiện; chỉ tốn thêm token đầu vào.

**`hard_eval`: 50 case khó.**
- *Đoán:* accuracy BoW ~0.7, Embedding ~0.8, LLM ~0.9. BoW khó nhất ở nhóm né từ khoá, nói về spam, tưởng spam, spam giọng tự nhiên; LLM khó nhất ở nhóm thao túng, prompt injection, tin giao dịch.
- *Thực tế:* accuracy BoW 0.52 và 0.60, Embedding 0.64 đến 0.70, LLM 0.98 (4d prompt phòng thủ: 1.00). LLM không bị prompt injection lừa (nhóm này 100%); LLM nhỏ chỉ sai 1 case, ở nhóm "tưởng spam nhưng không phải".
- *Vì sao lệch:* mình đánh giá thấp LLM và đánh giá cao BoW. BoW chỉ đếm token rời rạc nên mất tín hiệu khi spam không có từ khoá hoặc chữ bị đổi thành số, còn LLM đọc nghĩa cả câu nên khó bị đánh lừa bởi từ khoá hay câu lệnh chèn trong tin.

## 4. Phân tích 50 case khó

**Model tụt nhiều nhất:** BoW + LogReg, F1 từ 0.857 (test thường, SUB) xuống 0.333 trên 50 case khó (accuracy 0.52). BoW + MLP tụt gần như tương đương, từ 0.974 xuống 0.474 (accuracy 0.60). LLM nhỏ chỉ tụt từ 1.000 xuống 0.983, và LLM nhỏ với prompt phòng thủ giữ 1.000.

**Nhóm bị tụt nhiều nhất với BoW + LogReg:** spam giọng tự nhiên (đúng 10%), thao túng/lừa đảo (12%), né từ khoá (17%).

**Hai case cụ thể:**
1. Case #29 (né từ khoá, nhãn spam): *"C0ngr4tulat10ns! Y0u h4ve w0n a £1000 g1ft c4rd. Cl4im n0w at w1n-gift.c0m"*. Chữ bị thay bằng số nên các token như `c0ngr4tulat10ns` không có trong từ vựng BoW; vector gần như không còn từ khoá spam, mô hình thiếu căn cứ để nhận ra đây là spam.
2. Case #21 (thao túng, nhãn spam): *"Hi Mum, I dropped my phone in the toilet, this is my new number. Can you send £200 to my friend's account? I'll explain later x"*. Tin không có từ khoá spam nào ("free", "win", "claim"), giọng như người thân nên BoW coi là tin bình thường; chỉ model hiểu nghĩa (đóng vai người thân để xin tiền) mới nhận ra.

Điểm test thường quá lạc quan vì tập test lấy cùng phân phối với tập train; spam mẫu chỉ khác số điện thoại hay mã thưởng vẫn còn sau bước bỏ trùng tuyệt đối, nên model chỉ cần thuộc từ khoá là đạt điểm cao.

## 5. Thử thách cá nhân

*Đề theo SEED của bạn, cách làm, kết quả, giải thích.*

## 6. Phương án cho sản phẩm thật

Phương án chọn: hệ thống lai (router). Dùng BoW + MLP làm tầng 1 cho mọi tin, và chỉ chuyển các tin lưỡng lự
(0.3 ≤ p ≤ 0.7) sang LLM nhỏ với prompt phòng thủ.

Số liệu dẫn:
- BoW + MLP: F1 test 0.952, p95 2.39 ms, $0 mỗi 1k tin, nên đạt yêu cầu p95 < 50 ms và đủ nhanh
  cho ~116 tin/giây (throughput đo được ~13.000 tin/giây).
- Cả ba cách LLM có p95 từ 1383 đến 2271 ms, không đạt yêu cầu độ trễ, nhưng F1 SUB = 1.000
  và đạt 98 đến 100% ở 50 case khó (BoW + MLP chỉ 60%).
- Dùng LLM cho toàn bộ 10 triệu tin/ngày tốn khoảng $2.100/tháng (LLM nhỏ, $0.007/1k).
  Nếu chỉ ~5% tin đi qua LLM thì khoảng $105/tháng.

Đánh đổi chấp nhận:
- Tin giọng tự nhiên hoặc không có từ khoá mà BoW + MLP tự tin sai thì không được chuyển cho LLM,
  nên vẫn có một phần spam bị bỏ sót. Chấp nhận rủi ro này hoặc bổ sung tầng embedding/luật nhẹ.
- p95 < 50 ms chỉ đúng khi tỉ lệ chuyển sang LLM dưới 5%, hoặc xử lý tin lưỡng lự bất đồng bộ
  (giữ lại hoặc kiểm tra sau), vì LLM chậm ~1.4 đến 2.3 giây.
- Hệ thống phức tạp hơn một mô hình đơn, phải theo dõi tỉ lệ tin lưỡng lự và chi phí LLM.
- Số liệu dựa trên 40 tin (SUB) và 50 case viết tay nên chỉ là ước lượng.

## 7. AI log và điều đã học

| Phần | Bạn hỏi AI gì (prompt) | AI trả lời/viết gì | Bạn đã sửa / kiểm chứng gì |
| --- | --- | --- | --- |
| TODO 4.1 | Không có OpenAI API thì làm sao làm 4.1 | Giải thích có thể đổi provider (Gemini/Groq/Ollama) nhờ `base_url` tương thích OpenAI; viết hàm `call_openai` | Chạy ô kiểm tra giả lập (4.1 + 4.4 ✅) và gọi API thật |
| Lỗi API | Dán các lỗi: 400 sai key, 429 hết quota, 400 `reasoning_effort="minimal"`, ô chạy 17 phút | Chẩn đoán: key nhập sai, gói miễn phí 15 request/phút, tham số `reasoning_effort` không hợp lệ cho model lớn, model lớn Gemini quá tải (503); gợi ý nghỉ giữa các lần gọi, đổi sang `low`, chuyển sang OpenAI | Nạp key bằng Colab Secrets, chuyển sang OpenAI, chạy lại ô |
| TODO 4.2, 4.3, 4.4 | Dán khung hàm `build_classification_prompt`, `parse_label`, `llm_classify_batch` | Viết code cho 3 hàm | Ô kiểm tra 4.2–4.3 và 4.1 + 4.4 đạt ✅. Lần đầu chạy 4.A cho F1 = 0.000 và invalid = 40; AI gọi thử API và xác nhận `max_completion_tokens=3` trả câu rỗng (chữ "spam" cần 4 token); nâng giới hạn token trong TODO 4.4 (không sửa ô của người ra đề) thì F1 = 1.000, invalid = 0 |
| Khái niệm | F1, zero-shot/few-shot, đường Pareto và SLA, 50 case khó | Giải thích bằng ví dụ và đọc biểu đồ | Đối chiếu với kết quả thật trong notebook trước khi viết dự đoán và câu trả lời |
| Báo cáo | Hoàn thành các mục của báo cáo | Điền mục 1–4, 6, 7 từ `lab1_report.json` và output notebook. Không làm mục 5 (thử thách cá nhân, đề ghi không dùng AI) | Đối chiếu số liệu với `lab1_report.json` |
| Phản ánh cuối bài và mục 6 | Nhờ AI trả lời 2 câu phản ánh cuối bài (chọn phương án cho sản phẩm thật với p95 < 50 ms cho 10 triệu SMS/ngày; thiết kế hệ thống lai/router) và nêu phương án, số liệu, đánh đổi cho mục 6 | Soạn câu trả lời từ số liệu trong `lab1_report.json`: các cách đạt p95 < 50 ms, chi phí LLM khoảng $2.100/tháng (LLM nhỏ), thiết kế router BoW + MLP rồi chuyển tin lưỡng lự sang LLM nhỏ, ước tính khoảng $105/tháng nếu 5% tin đi qua LLM | Đối chiếu p95, F1, USD/1k tin với `lab1_report.json`. Tỉ lệ 5% tin lưỡng lự là giả định, chưa đo trên tập validation |
