<h1 align="center">👋 XÂY DỰNG HỆ THỐNG GỢI Ý THỰC ĐƠN DINH DƯỠNG CÁ NHÂN HÓA 🥗</h1>

<div align="center">
  
  <p align="center">
    <img src="images/logo.png" alt="Dai Nam Logo" width="200"/>
    <img src="images/AIoTLab_logo.png" alt="AIoTLab Logo" width="200"/>
  </p>

  [![Made by AIoTLab](https://img.shields.io/badge/Made%20by%20AIoTLab-blue?style=for-the-badge)](https://www.facebook.com/DNUAIoTLab)
  [![Fit DNU](https://img.shields.io/badge/Fit%20DNU-green?style=for-the-badge)](https://fitdnu.net/)
  [![DaiNam University](https://img.shields.io/badge/DaiNam%20University-red?style=for-the-badge)](https://dainam.edu.vn)

</div>

<h2 align="center">💡 Giải pháp gợi ý thực đơn thông minh dựa trên dữ liệu cá nhân<</h2>

<p align="left">
  Dự án này xây dựng một hệ thống **gợi ý thực đơn dinh dưỡng cá nhân hóa** sử dụng dữ liệu người dùng, thông tin thành phần dinh dưỡng và mô hình **Học máy/Deep Learning**. Người dùng có thể nhập thông tin cá nhân (tuổi, cân nặng, chiều cao, mục tiêu dinh dưỡng) và nhận được các đề xuất món ăn phù hợp với nhu cầu dinh dưỡng, sở thích và chế độ ăn uống. Hệ thống kết hợp phân tích dữ liệu, tiền xử lý và mô hình dự đoán để tối ưu hóa chế độ ăn một cách thông minh và tiện lợi. 🚀
</p>

---

## 🌟 Giới thiệu hệ thống

- **Dữ liệu dinh dưỡng:** Bao gồm calo, protein, carbohydrate, chất béo, vitamin và khoáng chất.
- **Mô hình AI:** Sử dụng các thuật toán ML/DL (Random Forest, LSTM, hoặc Mạng nơ-ron) để gợi ý thực đơn.
- **Giao diện tương tác:** Web app/GUI cho phép người dùng nhập dữ liệu, nhận đề xuất, đánh giá và lưu lại các bữa ăn.
- **Tùy chỉnh cá nhân:** Điều chỉnh theo lượng calo mục tiêu, dị ứng, sở thích món ăn hoặc chế độ ăn đặc biệt (Vegetarian, Keto, Low-carb…).

---

## 🏗️ Kiến trúc hệ thống

Hệ thống gồm 3 khối chính:

1. **Khối Dữ liệu:** Thu thập dữ liệu thực phẩm, thành phần dinh dưỡng, dữ liệu người dùng.
2. **Khối Xử lý & AI:** Tiền xử lý dữ liệu, chuẩn hóa, xây dựng và huấn luyện mô hình dự đoán thực đơn phù hợp.
3. **Khối Giao diện Người dùng:** Web app/GUI hiển thị gợi ý thực đơn và cho phép tương tác.

![Kiến trúc hệ thống](images/architecture.png)

---

## 📂 Cấu trúc dự án

```
project/
│
├─ raw-data_recipe.csv # Dữ liệu gốc các công thức nấu ăn
├─ Food_and_Nutrition__.csv # Dữ liệu thông tin người dùng
├─ recipes_clean.csv # Dữ liệu đã làm sạch
├─ model_cal.pkl # Mô hình dự đoán nhu cầu calo
├─ model_prot.pkl # Mô hình dự đoán nhu cầu protein
├─ app.py # Ứng dụng Streamlit cho người dùng
├─ xu_ly_du_lieu.ipynb # Notebook tiền xử lý, huấn luyện mô hình
└─ README.md # Hướng dẫn dự án        
```


---

## 🛠️ Công nghệ & Yêu cầu Hệ thống

| Lĩnh vực | Công nghệ | Chi tiết |
| :--- | :--- | :--- |
| **Dữ liệu** | Python, Pandas, NumPy, ast | Tiền xử lý, chuyển đổi cấu trúc dữ liệu dinh dưỡng (JSON $\to$ cột). |
| **Học máy (ML)** | scikit-learn (RandomForestRegressor) | Dự đoán nhu cầu Calo/Protein cá nhân hóa. |
| **Gợi ý & Phân loại** | TfidfVectorizer, KMeans | Phân loại món ăn thành 5 nhóm cluster để gợi ý món thay thế. |
| **Giao diện** | Streamlit | Xây dựng Web App tương tác (Sidebar, Bảng, Biểu đồ). |

**Yêu cầu Phần mềm:** Python 3.x, và các thư viện trong `requirements.txt` (`streamlit`, `scikit-learn`, `pandas`, `matplotlib`, `pickle`).

---

## 🔬 Khối Xử lý Dữ liệu và Feature Engineering

Dữ liệu thô được chuẩn hóa để xây dựng Ground Truth cho mô hình AI và các tính năng lọc.

### 1. Tính toán Chỉ số Cơ thể & Mục tiêu

* **Chỉ số BMI:** Tính toán dựa trên cân nặng (kg) và chiều cao (m).
    $$BMI = \frac{\text{weight\_kg}}{(\text{height\_m})^2}$$
* **Nhu cầu Calo (Target TDEE):** Tính TDEE dựa trên công thức BMR (Harris-Benedict hoặc Mifflin-St Jeor), mức độ vận động, và điều chỉnh theo mục tiêu/BMI.
    * **Điều chỉnh theo BMI:** BMI > 25 $\to$ giảm **15%** TDEE; BMI < 18.5 $\to$ tăng **15%** TDEE.
* **Nhu cầu Protein (Target Protein):** Thiết lập là **2g / kg** trọng lượng cơ thể (phù hợp với người tập luyện/kiểm soát cân nặng).

### 2. Tiền xử lý Dữ liệu Công thức

* **Làm sạch:** Bỏ các cột không cần thiết (`aver_rate`, `image_url`, v.v.).
* **Parse Dinh dưỡng:** Chuyển đổi cột `nutritions` (chuỗi JSON) thành các cột số (`calories`, `protein`, `fat`, `carbohydrates`, v.v.).
* **Lọc Cá nhân:** Xây dựng hàm lọc món ăn dựa trên danh sách `Thực phẩm muốn tránh` và `Thực phẩm ưu tiên` của người dùng.

---

## 🧠 Huấn luyện Mô hình AI (Random Forest)

Mô hình học máy được sử dụng để dự đoán nhu cầu dinh dưỡng cá nhân hóa.

* **Dữ liệu Đầu vào (X):** Các thông số cơ thể đã được số hóa (`BMI`, `age`, `gender_num`, `height_cm`, `weight_kg`, `activity_num`).
* **Mục tiêu Dự đoán (Y):** `target_calories` và `target_protein`.
* **Mô hình:** `RandomForestRegressor` được huấn luyện độc lập cho Calo (`model_cal.pkl`) và Protein (`model_prot.pkl`).
* **Đánh giá:** Sử dụng **R² Score**, **MAE**, và **RMSE** để đánh giá độ chính xác của mô hình so với Ground Truth.

---

## 💻 Ứng dụng Streamlit và Tính năng Gợi ý

Ứng dụng web cho phép người dùng tương tác và nhận thực đơn tức thì.

### 🌟 Tính năng Chính

* **Dự đoán Nhu cầu:** Sử dụng mô hình `RandomForestRegressor` đã lưu (`.pkl`) để dự đoán Calo và Protein mục tiêu.
* **Clustering Món ăn (KMeans):**
    * Sử dụng **TF-IDF trên Nguyên liệu** và **Chuẩn hóa Dinh dưỡng** làm đầu vào.
    * Phân loại thành **5 Nhóm (Cluster)**: `Low-Calorie`, `High-Protein`, `Balanced`, `High-Fat`, `Carb-Heavy`.
    * Mỗi món ăn được gán một **Nhóm món** (Cluster Label).
* **Tạo Thực đơn:** Chia tổng Calo mục tiêu thành 4 bữa (Sáng, Trưa, Tối, Phụ) theo tỷ lệ phần trăm cố định. Gợi ý món ăn trong phạm vi Calo mục tiêu của từng bữa.
* **Gợi ý Món Thay thế:** Đề xuất các món ăn **cùng nhóm Cluster** với món ăn hiện tại, cho phép người dùng thay thế và cập nhật dinh dưỡng.
* **Lưu Lịch sử:** Lưu lại tối đa 10 thực đơn đã tạo.

### 📊 Ví dụ Kết quả Thực đơn

| Bữa | Món ăn | Calo | Protein (g) | Chất béo (g) | Nhóm món |
| :---: | :---: | :---: | :---: | :---: | :---: |
| Bữa sáng | Trứng ốp la | 300 | 20 | 15 | High-Protein |
| Bữa trưa | Cơm gà | 550 | 35 | 20 | Balanced |
| Bữa tối | Salad cá hồi | 400 | 25 | 18 | Low-Calorie |
| Bữa phụ | Sữa chua | 150 | 8 | 5 | Balanced |

---

## 🚀 Hướng dẫn Chạy Ứng dụng

### 1. 📂 Chuẩn bị Dữ liệu Gốc

Tải các tập dữ liệu gốc về và đặt chúng trong thư mục chính của dự án:

| Tên file | Liên kết Tải xuống |
| :--- | :--- |
| `raw-data_recipe.csv` | [Link Google Drive] |
| `Food_and_Nutrition__.csv` | [Link Google Drive] |

### 2. 🧠 Tiền xử lý và Huấn luyện Mô hình

* **Chạy Notebook:** Mở file `xu_ly_du_lieu.ipynb` (hoặc tên file notebook tương ứng) và thực hiện **Run All**.

    ```bash
    Run All trong file ipynb
    # Bước này sẽ tạo ra recipes_clean.csv, model_cal.pkl, model_prot.pkl
    ```

### 3. ⚙️ Khởi động Ứng dụng Streamlit

* **Cài đặt Thư viện:** Đảm bảo bạn đã cài đặt tất cả các thư viện cần thiết (`pip install -r requirements.txt`).

* **Khởi động Ứng dụng:** Chạy ứng dụng bằng lệnh sau trong terminal:

    ```bash
    streamlit run app.py
    # Lưu ý: Nếu tên file là app_multilang_full.py, hãy chạy lệnh tương ứng.
    ```

### 4. 🍽️ Sử dụng Hệ thống

* Nhập thông tin cá nhân trên **Sidebar**, sau đó nhấn **Tạo thực đơn AI** để nhận gợi ý.

---

## ✅ Kết luận và Hướng phát triển

### 📌 Giá trị Hệ thống

* **Cá nhân hóa Sâu:** Kết hợp phân tích BMI, TDEE, Mục tiêu và Thói quen ăn uống.
* **Độ tin cậy:** Sử dụng Mô hình AI (Random Forest) để xác định nhu cầu dinh dưỡng, tăng cường tính chính xác so với công thức truyền thống.
* **Tính linh hoạt:** Cho phép người dùng tùy chỉnh và thay thế món ăn theo nhóm dinh dưỡng.

### 🌟 Mở rộng trong Tương lai

* **Tối ưu hóa Vĩ mô:** Áp dụng các thuật toán **Tối ưu hóa (Optimization)** để đảm bảo tổng lượng Calo, Protein, Carb và Fat trong ngày đạt chính xác mục tiêu đề ra (thay vì chỉ cố gắng đạt mục tiêu Calo cho từng bữa).
* **Ràng buộc Bệnh lý:** Thêm logic lọc món ăn theo các bệnh lý đặc biệt (ví dụ: Tăng huyết áp $\to$ giảm Sodium).

---

## 🤝 TÁC GIẢ 👥

Dự án được phát triển bởi:

* **Trần Hồng Quân**

© 2025 NHÓM 20, KHOA CÔNG NGHỆ THÔNG TIN, TRƯỜNG ĐẠI HỌC ĐẠI NAM.
