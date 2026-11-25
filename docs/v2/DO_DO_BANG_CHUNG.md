# Độ đo bằng chứng: F1 theo ý (mệnh đề)

> Đo xem phần mô hình tô có **đúng ý** với phần người gán tô hay không, không bắt bẻ từng chữ ở mép cụm.
> Mã: `src/v2/eval/evidence.py`. Test tự động: `tests/v2/test_v2_eval.py`.

## 1. Vì sao cần độ đo này

F1 mức từ (đếm từng từ trùng nhau) quá khắt khe. Với câu "sao m ngu như bò thế", người gán tô `ngu như bò`:

- Mô hình tô cả mệnh đề `sao m ngu như bò thế` chỉ được **0,67**, dù đã tô đúng chỗ.
- Mô hình tô `như bò` được **0,80**, mô hình tô `ngu` chỉ được **0,50**.

Thống kê ngày 24/11/2026 cho thấy khoảng 70% cụm bằng chứng trong test trùng ranh giới dấu câu, tức là người gán
tô theo ý. Vì vậy độ đo lấy **mệnh đề** làm đơn vị: tô lệch mép trong cùng một ý thì không bị trừ, chỉ trừ khi
sót ý hoặc tô lan sang ý khác. Người gán không phải đánh dấu thêm gì.

## 2. Cách tính

1. **Tách từ và mệnh đề.** Mỗi dòng tách thành từ theo dấu cách. Mệnh đề được cắt tại các dấu `, . ! ? ; : …`
   (dấu chấm giữa từ như `xxx[.]com` không cắt). Xuống dòng luôn là ranh giới.
2. **Tìm phần mô hình tô.** Tìm từng cụm mô hình trả về trong tin nhắn, khớp nguyên văn. Cụm xuất hiện nhiều lần
   thì tô hết, giống cách ứng dụng hiển thị. Cụm không có trong tin nhắn thì bỏ và đếm vào tỉ lệ non-verbatim.
3. **Ý gán.** Mỗi cụm người gán được tìm nguyên văn (ký hiệu che như `[STK]`, `[SDT]` giữ nguyên). Phần cụm
   nằm trong một mệnh đề là một **ý gán**; cụm vắt qua hai mệnh đề thành hai ý gán. Cụm gán xuất hiện nhiều lần
   thì lấy lần được tô nhiều nhất.
4. **Recall (bắt được ý).** Mỗi ý gán được `min(1; 1,5 × số từ của ý được tô / số từ của ý)`. Tô trúng từ 2/3
   số từ trở lên là đủ điểm, ít hơn thì được theo tỉ lệ. R là trung bình các ý gán trong đoạn.
5. **Precision (không tô lố).** P = số từ mô hình tô nằm trong các mệnh đề có ý gán / tổng số từ mô hình tô.
6. **Gộp lại:** F1 = 2 × P × R / (P + R). Tính cho từng đoạn nguy hại có bằng chứng gán, rồi lấy trung bình.

Hệ số 1,5 (`HE_SO_PHU_MAC_DINH`) chốt ngày 24/11/2026, trước khi nhìn điểm test, và không được chỉnh sau đó.

## 3. Ví dụ

### Ví dụ 1: câu ngắn

Tin nhắn: `sao m ngu như bò thế. t đã nói nhiều lần rồi mà`. Người gán: `ngu như bò`.
Mệnh đề có ý gán: `sao m ngu như bò thế.`

| Mô hình tô | F1 theo ý | F1 mức từ | Vì sao |
| --- | --- | --- | --- |
| ngu như bò | **1,00** | 1,00 | Đúng y |
| sao m ngu như bò thế | **1,00** | 0,67 | Tô trọn mệnh đề, không bị trừ |
| như bò | 1,00 | 0,80 | Trúng 2/3 từ là đủ điểm |
| ngu | 0,67 | 0,50 | Trúng 1/3 từ: R = 0,5 |
| ngu như bò thế. t đã | 0,80 | 0,67 | "t đã" lố sang mệnh đề sau |
| cả tin nhắn | 0,63 | 0,38 | Lố 6/11 từ |
| t đã nói | **0** | 0 | Sai chỗ |

### Ví dụ 2: hai dòng, hai ý

```
sao mày ngu như bò thế. t đã nói nhiều lần rồi mà
mai ra cổng sau gặp tao, không thì biết tay
```

Người gán: `ngu như bò` · `mai ra cổng sau gặp tao`. Mệnh đề có ý gán: `sao mày ngu như bò thế.` và
`mai ra cổng sau gặp tao,`.

| Mô hình tô | R | P | F1 theo ý |
| --- | --- | --- | --- |
| `ngu như bò thế` · `ra cổng sau gặp tao, không thì biết tay` | 1 | 9/13 = 0,69 | 0,82 |
| `gặp tao` | (0 + 0,5)/2 = 0,25 | 1 | 0,40 |
| `mày ngu` · `không thì biết tay` | (0,5 + 0)/2 = 0,25 | 2/6 = 0,33 | 0,29 |

### Ví dụ 3: hội thoại dài (8 dòng, nhãn `DE_DOA`)

```
1 ê m là đứa đăng bài nói xấu t trên group lớp à
2 ừ thì sao, t nói đúng mà
3 mày ngu như bò thế. t đã nhắn riêng bảo gỡ rồi mà ko nghe
4 kệ t
5 ok vậy mai tan học m ra cổng sau trường gặp t
6 để làm gì
7 ra thì biết. ko ra thì t đăng hết ảnh của m lên confession
8 ừ để xem
```

Người gán: `ngu như bò` · `ra thì biết` · `t đăng hết ảnh của m lên confession`

| Mô hình tô | F1 theo ý | F1 mức từ |
| --- | --- | --- |
| Gọn, đủ 3 ý | **1,00** | 0,96 |
| Đủ ý, tô rộng trong mệnh đề ("mày ngu như bò thế", "ko ra thì t đăng…") | **1,00** | 0,85 |
| Sót "ra thì biết", trúng một phần hai ý còn lại ("ngu", "đăng hết ảnh") | 0,52 | 0,44 |
| Tô cả đoạn 8 dòng | **0,43** | 0,34 |
| Tô sai chỗ (dòng 1, dòng 5) | **0** | 0 |

## 4. Chỉ số phụ (báo cạnh F1 theo ý)

- **F1 mức từ** (khớp từng từ chặt), để người đọc thấy nhóm không tự nới độ đo cho điểm đẹp.
- **Tỉ lệ AN_TOAN bị tô oan:** đoạn `AN_TOAN` không tính F1, chỉ đếm đoạn mô hình có tô.
- **Tỉ lệ đoạn nguy hại không tô gì.**
- **Tỉ lệ cụm non-verbatim:** cụm mô hình trả về không có nguyên văn trong tin nhắn.
- **Tỉ lệ độ dài:** tổng số từ mô hình tô / tổng số từ người gán trên các đoạn nguy hại.

## 5. Giới hạn

- Không có dấu từ mang ý, nên mọi từ trong cụm gán tính như nhau: tô `như bò` (mất chữ "ngu") vẫn đủ điểm
  như tô `ngu như`. Cần nêu rõ điều này trong báo cáo.
- Mệnh đề dài không có dấu câu thì tô lan trong mệnh đề đó không bị trừ. F1 mức từ đặt cạnh để thấy phần này.
