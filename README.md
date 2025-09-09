Bài này là một ứng dụng web dự đoán khối u vú được xây dựng bằng Flask (Python), scikit-learn và HTML/CSS/JS. Nội dung chính gồm:

Xử lý dữ liệu (data.csv)

Dữ liệu ban đầu về đặc trưng khối u.

Loại bỏ các cột không cần thiết (id, Unnamed: 32).

Chuyển đổi cột diagnosis thành số: M (ác tính) = 0, B (lành tính) = 1.

Chọn 5 đặc trưng quan trọng để huấn luyện mô hình:

concave points_mean

concave points_worst

area_worst

concavity_mean

radius_worst

Huấn luyện mô hình học máy

Dùng RandomForestClassifier để phân loại khối u lành tính hay ác tính.

Chia dữ liệu thành tập huấn luyện (70%) và kiểm tra (30%).

Ứng dụng Flask (app.py)

Trang chủ (/) hiển thị form nhập dữ liệu đặc trưng.

API /predict nhận dữ liệu từ form (JSON), đưa vào mô hình dự đoán, trả về kết quả:

✅ LÀNH TÍNH

❌ ÁC TÍNH

Giao diện người dùng (index.html)

Form nhập 5 đặc trưng.

Nút 🔍 Dự đoán gửi dữ liệu lên server bằng fetch API.

Kết quả hiển thị đẹp mắt với màu xanh (lành tính) hoặc đỏ (ác tính).

Thiết kế hiện đại, gradient nền, card form bo tròn, có hiệu ứng hover và animation.

👉 Tóm lại: Đây là một hệ thống web nhỏ cho phép người dùng nhập các thông số y tế, sau đó dựa trên mô hình học máy để dự đoán loại khối u vú (lành tính/ác tính) và trả kết quả trực quan ngay trên giao diện
web.
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/5d3d6b5e-ed05-46d5-9261-6c444161d7fb" />
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/1ee1e998-ff61-41e2-b2fd-9d85163d17dd" />

