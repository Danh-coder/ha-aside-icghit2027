# Reproducibility và định dạng references

- Thêm URL có thể bấm trong mục Reproducibility: https://github.com/Danh-coder/ha-aside-icghit2027.
- Mô tả đúng nội dung repository: source code, lựa chọn dữ liệu được cố định, hướng dẫn training/evaluation và các output lưu sẵn. Model weights không được phân phối; sinh lại response HA-ASIDE cần training theo recipe công bố.
- Đối chiếu 14 references: 2 journal articles (`@article`) và 12 conference papers (`@inproceedings`). Không có reference kiểu website hoặc preprint.
- Hai journal: *ACM Computing Surveys* và *Transactions of the Association for Computational Linguistics*. Giữ volume, issue khi có, pagination và article number thích hợp.
- Chuẩn hóa tên venue theo dạng viết tắt IEEE. Tách volume 1/2 của KDD và volume 1 của NAACL khỏi tên proceedings.
- Bổ sung trang chính thức từ BibTeX của ICLR cho SEP (67147–67179), ASIDE (112869–112898), ISE (93728–93758), và từ USENIX cho *The Attacker Moves Second* (1467–1486).
- Đồng bộ tên Christoph Lampert và tên bài mtRAG với publisher metadata.
- Tắt phần URL “[Online]. Available” bằng tùy chọn `CTLuse_url = {no}` của IEEEtran. Vẫn giữ `url` trong `.bib` để tra cứu. Hiển thị 8 DOI đã xác minh dưới dạng `doi: ...`, có hyperlink.
- Không chỉnh `IEEEtran.cls` hoặc `IEEEtran.bst`. Bản BST hội nghị không tự hiển thị field `doi`; vì vậy DOI được xuất qua field `note` mà BST hỗ trợ.

Nguồn đối chiếu: metadata do nhà xuất bản gửi lên Crossref; BibTeX chính thức của ICLR/NeurIPS; ACL Anthology; USENIX; PMLR. Các snapshot offline được lưu trong `analysis/reference-metadata-2026-11-17.json` và `analysis/reference-proceedings-bibtex-2026-11-17.json`. `references_audit.json` ghi từng reference, loại tài liệu, trang và DOI. Các kết quả thực nghiệm, equations, figures và thứ tự citation không thay đổi.
