# Reflection — Lab 19

**Tên:** Pham Duc Hiep
**Cohort:** A20-K3
**Path đã chạy:** lite (fastembed / bge-small-en, Qdrant in-memory, Feast SQLite)

---

## Câu hỏi (≤ 200 chữ)

> Trên golden set 50 queries, mode nào thắng ở loại query nào (`exact` /
> `paraphrase` / `mixed`), và tại sao? Khi nào bạn **không** dùng hybrid
> (i.e. khi nào pure BM25 hoặc pure vector là lựa chọn đúng)?

Trên 50 query (lite, bge-small-en): hybrid thắng trung bình (78.6% > BM25 77.8% > vector 73.2%). `exact` (15): BM25 = hybrid 96.7%, vector 88.7% — từ kỹ thuật verbatim, lexical đủ. `paraphrase` (15): BM25 33.3% > hybrid 32.0% > vector 24.0% — model Anh yếu trên câu Việt diễn đạt lại. `mixed` (20): hybrid 100% — RRF giữ cả tín hiệu exact lẫn gần nghĩa.

Không dùng hybrid khi query chắc chắn exact-term (log, mã lỗi, tên API → BM25 rẻ hơn) hoặc corpus/query cùng ngôn ngữ + model đa ngữ mạnh và user luôn paraphrase (lúc đó vector đủ, khỏi trả chi phí RRF depth=50). Hybrid là mặc định khi không biết trước kiểu query.

---

## Điều ngạc nhiên nhất khi làm lab này

bge-small-en thua BM25 trên paraphrase tiếng Việt (24% vs 33%) — đúng bài học “đổi model = phải đo lại”, không copy ngưỡng từ blog. Trên Windows, `localhost` resolve IPv6 làm NB3 treo trong khi `127.0.0.1` chạy bình thường.

---

## Bonus challenge

- [ ] Đã làm bonus (xem `bonus/`)
- [ ] Pair work với: _<tên đồng đội nếu có>_
