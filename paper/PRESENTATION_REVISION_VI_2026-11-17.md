# Chỉnh author block, Figure 1 và chữ viết tắt

## Author block

- Giữ nguyên thứ tự và danh tính: Phan Cong Danh, Wooguil Pak.
- Thêm `1st`/`2nd` và dấu `*`/`†` như ảnh template người dùng cung cấp.
- Hai hàng affiliation tương ứng in nghiêng; cùng affiliation đã xác nhận, không tự bổ sung cơ sở hoặc chức danh.
- Emails nằm trên một dòng riêng, theo thứ tự tác giả. Các ký hiệu liên kết tên với affiliation, không tự chỉ định corresponding author.

## Đánh giá Figure 1 cũ

1. Ba hàng nguồn và embedding chưa thể hiện rõ việc ghép thành một chuỗi token kèm nhãn nguồn.
2. Hình vector góc quay có tính minh họa nhưng chưa cho thấy nhãn role đi vào phép biến đổi của từng token như thế nào.
3. Nhãn “Existing model parameters” có thể khiến người đọc hiểu nhầm Transformer được giữ frozen.

## Figure 1 đã sửa

- Luồng nguồn → token cùng nhãn → một embedding lookup → rotation theo role → một Transformer → response.
- Đường nét đứt tách nhãn nguồn điều khiển góc quay khỏi luồng embedding.
- Hiển thị ba góc S: 0, U: π/8, D: π/2; cùng góc áp dụng cho mọi cặp tọa độ của một token.
- Transformer được minh họa bằng các decoder layer dùng chung và ghi rõ “Fine-tuned”. Rotation không thêm tham số trainable.
- Hình vector PDF/SVG; chữ 8 pt Times New Roman, không giảm font để đủ trang.

## Chữ viết tắt

- Tên HA-ASIDE đã được người dùng chọn: **Hierarchical Architectural Separation of Instructions and Data**.
- ASIDE: **Architecturally Separated Instruction–Data Embeddings**, theo tên phương pháp trong bài gốc.
- Giải thích LLM, AI, ISE, ISE-2, SEP, MTRAG, IHEval, BIPIA, StruQ, CaMeL, QA, AdamW và ROUGE-L tại lần xuất hiện thích hợp; giải thích F1 bằng precision/recall.
- Abstract và thân bài định nghĩa độc lập các thuật ngữ cần thiết. SEP là nhãn dataset; mô tả rõ mục đích instruction–data separation, không gán một backronym mới.
- Giữ tên riêng như Qwen, ChatQA, AgentDojo và TrustAgent; không tự bịa cách mở rộng.

Các số liệu, công thức, cohort, reference và phạm vi single-turn/future work được giữ nguyên. Các thay đổi trình bày được ghi riêng tại `analysis/presentation_revision.json` để đối chiếu với snapshot trước đó.
