import os
import pandas as pd
from flask import Flask, render_template, request, jsonify
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# =============================
# 1. Đọc và xử lý dữ liệu
# =============================
BASE_DIR = os.path.dirname(__file__)  # thư mục chứa app.py
CSV_PATH = os.path.join(BASE_DIR, "data.csv")

du_lieu = pd.read_csv(CSV_PATH)
du_lieu = du_lieu.drop(columns=["id", "Unnamed: 32"], errors="ignore")
du_lieu["diagnosis"] = du_lieu["diagnosis"].map({"M": 0, "B": 1})

# Lựa chọn đặc trưng quan trọng
dac_trung_quan_trong = [
    "concave points_mean",
    "concave points_worst",
    "area_worst",
    "concavity_mean",
    "radius_worst"
]

X = du_lieu[dac_trung_quan_trong]
y = du_lieu["diagnosis"]

# =============================
# 2. Train mô hình Random Forest
# =============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

mo_hinh = RandomForestClassifier(
    n_estimators=100,
    max_features="sqrt",
    random_state=42
)
mo_hinh.fit(X_train, y_train)

# =============================
# 3. Flask app
# =============================
app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html", dac_trung=dac_trung_quan_trong)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json  # lấy dữ liệu JSON từ fetch

        # kiểm tra dữ liệu đầu vào
        if not all(ten in data for ten in dac_trung_quan_trong):
            return jsonify({"error": "Thiếu dữ liệu đầu vào"}), 400

        # chuyển dữ liệu sang dạng phù hợp
        gia_tri_nhap = [float(data[ten]) for ten in dac_trung_quan_trong]
        du_lieu_moi = pd.DataFrame([gia_tri_nhap], columns=dac_trung_quan_trong)

        du_doan_moi = mo_hinh.predict(du_lieu_moi)

        ket_qua = "✅ Khối u LÀNH TÍNH" if du_doan_moi[0] == 1 else "❌ Khối u ÁC TÍNH"

        return jsonify({"result": ket_qua})
    except Exception as e:
        return jsonify({"error": f"Lỗi xử lý: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)
