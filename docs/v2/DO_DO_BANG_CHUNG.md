# Độ đo bằng chứng: F1 theo ý

> Đo xem phần mô hình tô có **đúng ý** với phần người gán tô hay không, không bắt bẻ từng chữ ở mép cụm.

## 1. Vì sao cần độ đo này

F1 mức từ (đếm từng từ trùng nhau) quá khắt khe. Với câu "mày ngu như bò", người gán tô `ngu như bò`:

- Mô hình tô `mày ngu` chỉ được **0,40**, dù đã bắt đúng chữ quan trọng nhất là "ngu".
- Mô hình tô `như bò` lại được **0,80**, dù đã bỏ mất chữ "ngu", tức là mất ý.

F1 theo ý sửa hai chỗ này:

- Tô **trúng từ mang ý** là được điểm.
- Chỉ trừ điểm khi **tô lố ra xa**. Lố một chút ngay cạnh cụm thì không bị trừ.

## 2. Người gán cần làm thêm gì

1. Tô cụm bằng chứng như bình thường (mục 5 của `HUONG_DAN_GAN_NHAN.md`).
2. Đặt ngoặc `[ ]` quanh **từ mang ý**, tức là từ mà nếu bỏ đi thì cụm hết nguy hại:
   - `[ngu] như bò`
   - `mình [biết] bạn học [trường nào] rồi đấy` (được đặt nhiều chỗ)
   - Không chọn được từ nào thì để trống. Khi đó mọi từ trong cụm được tính như nhau.

## 3. Cách tính (6 bước)

1. **Tách từ và mệnh đề.** Mỗi dòng tách thành từ theo dấu cách. Mệnh đề được cắt tại các dấu `, . ! ? ; : …`.
2. **Tìm phần mô hình tô.** Tìm từng cụm mô hình trả về trong tin nhắn, khớp nguyên văn. Cụm xuất hiện nhiều lần
   thì tô hết, giống cách ứng dụng hiển thị. Cụm không có trong tin nhắn thì bỏ.
3. **Điểm "bắt được ý" (Recall, R).** Với mỗi cụm người gán, tính tỉ lệ từ trong `[ ]` mà mô hình tô trúng.
   Cụm không có ngoặc thì tính trên mọi từ của cụm. R là trung bình của các cụm.
4. **Vùng cho phép.** Gồm cụm người gán, cộng **tối đa 2 từ mỗi bên**. Vùng này không vượt ra ngoài mệnh đề
   và không sang dòng khác.
5. **Điểm "không tô lố" (Precision, P).** P = số từ mô hình tô nằm trong vùng cho phép / tổng số từ mô hình tô.
6. **Gộp lại:** F1 = 2 × P × R / (P + R). Tính cho từng đoạn, rồi lấy trung bình trên các đoạn.

Lưu ý: chấm theo **vị trí** từ, không theo chữ. Tô đúng chữ "ngu" nhưng ở dòng khác thì vẫn được 0.

## 4. Ví dụ

### Ví dụ 1: câu ngắn

Tin nhắn: `sao m ngu như bò thế. t đã nói nhiều lần rồi mà`. Người gán: `[ngu] như bò`.
Vùng cho phép: `sao m ngu như bò thế.` (dừng ở dấu chấm).

| Mô hình tô | F1 theo ý | F1 mức từ (cũ) | Vì sao |
| --- | --- | --- | --- |
| ngu | **1,00** | 0,50 | Trúng từ mang ý là đủ |
| m ngu | **1,00** | 0,40 | "m" nằm sát cụm, không bị trừ |
| sao m ngu như bò thế | **1,00** | 0,67 | Lố nhưng vẫn trong cùng mệnh đề |
| ngu như bò thế. t đã | 0,80 | 0,67 | "t đã" lố sang câu sau nên bị trừ |
| cả câu (13 từ) | 0,63 | 0,38 | Lố 7 từ nên bị trừ nhiều |
| như bò | **0** | 0,80 | Mất "ngu" là mất ý |

### Ví dụ 2: câu không có một từ lõi duy nhất

Tin nhắn: `mình biết bạn học trường nào rồi đấy`. Người gán: `mình [biết] bạn học [trường nào] rồi đấy`.

| Mô hình tô | F1 theo ý | F1 mức từ (cũ) | Vì sao |
| --- | --- | --- | --- |
| biết bạn học trường nào | **1,00** | 0,77 | Phủ đủ 3 từ mang ý |
| trường nào | 0,80 | 0,40 | Phủ 2/3 từ mang ý |
| rồi đấy | **0** | 0,40 | Chỉ có từ đệm, không có ý |

### Ví dụ 3: hội thoại dài (8 dòng, 69 từ, nhãn `DE_DOA`)

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

Người gán: `[ngu] như bò` · `ra thì biết` · `t [đăng hết ảnh] của m lên [confession]`

| Mô hình tô | F1 theo ý | F1 mức từ (cũ) | Vì sao |
| --- | --- | --- | --- |
| Gọn, đủ 3 ý | **1,00** | 0,88 | Đúng cả 3 ý, không lố |
| Đủ ý, lố nhẹ ("mày ngu như bò thế", "ko ra thì t đăng…") | 0,97 | 0,85 | Chỉ "ko" vượt quá 2 từ, bị trừ rất ít |
| Thiếu một phần ý ("ngu", "đăng hết ảnh") | 0,74 | 0,44 | Sót "ra thì biết" và "confession" |
| Chỉ bắt câu chửi, sót lời dọa | 0,50 | 0,33 | Mất đúng phần làm nên nhãn `DE_DOA` |
| Tô cả dòng 3 và dòng 7 | 0,77 | 0,65 | Đủ ý nhưng lố 10 từ |
| Tô cả đoạn 8 dòng | **0,41** | 0,34 | Tô bừa thì không được lợi |
| Tô sai chỗ (dòng 1, dòng 5) | **0** | 0 | Không trúng ý nào |

## 5. Nhớ nhanh

- **Trúng ý** thì điểm cao, kể cả khi thiếu từ phụ.
- **Lố gần** (trong cùng mệnh đề, cách cụm tối đa 2 từ) thì không bị trừ.
- **Lố xa** (sang câu khác, dòng khác, cả đoạn) thì bị trừ dần theo số từ lố.
- **Sót ý** thì mất điểm, dù tô trúng các từ khác.
- **Tô sai chỗ** thì được 0.

## 6. Lưu ý khi dùng

- Con số 2 (số từ được lố mỗi bên) phải chốt **trước** khi nhìn test, không chỉnh sau.
- Trong báo cáo, đặt F1 theo ý **cạnh** F1 mức từ, để người đọc thấy nhóm không tự nới độ đo cho điểm đẹp.
- Chỉ tính trên đoạn nguy hại có bằng chứng người gán. Đoạn `AN_TOAN` không tính F1 mà đo riêng tỉ lệ bị tô oan.
- Nếu cụm người gán xuất hiện nhiều lần trong một đoạn, người gán cần ghi thêm số dòng.
- Các số trong mục 4 được tính bằng bản chạy thử của độ đo. Khi viết `src/v2/eval/bang_chung.py`, các ví dụ này
  sẽ trở thành test tự động.
