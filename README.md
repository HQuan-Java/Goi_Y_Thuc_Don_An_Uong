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

## 🛠️ Công nghệ sử dụng

- **Python**: Pandas, NumPy, ast, Matplotlib
- **Machine Learning**: scikit-learn (RandomForestRegressor, StandardScaler, KMeans)
- **Text Processing**: TfidfVectorizer
- **Web App**: Streamlit
- **Serialization**: Pickle

---

## 🛠️ Yêu cầu hệ thống

### Phần mềm
- Python 3.x
- Thư viện: `numpy`, `pandas`, `scikit-learn`, `tensorflow`, `joblib`, `streamlit`, `matplotlib`, `seaborn`

### Lưu ý
- Chuẩn hóa dữ liệu người dùng và dữ liệu thực phẩm trước khi huấn luyện.
- Máy tính có GPU sẽ tăng tốc quá trình huấn luyện mô hình Machine Learning.

---

## 🔹 Xử lý dữ liệu

1. **Đọc dữ liệu**: `raw-data_recipe.csv` chứa thông tin công thức, nguyên liệu, dinh dưỡng.
2. **Làm sạch dữ liệu**:
   - Bỏ các cột không cần thiết: `aver_rate`, `image_url`, `reviews`, `review_nums`.
   - Parse cột `nutritions` từ chuỗi JSON sang các cột dinh dưỡng: `calories`, `protein`, `fat`, `carbohydrates`, `fiber`, `sodium`.
   - Tách danh sách nguyên liệu (`ingredients_list`).
3. **Xuất dữ liệu sạch**: `recipes_clean.csv`.

```python
df_final.to_csv('recipes_clean.csv', index=False)
4. **Lọc món ăn theo thói quen:**:
   - Tránh thực phẩm người dùng không thích.
   - Ưu tiên thực phẩm người dùng yêu cầu.

## 🔹 Tính toán chỉ số cơ thể

- **BMI**: 

\[
BMI = \frac{\text{weight\_kg}}{(\text{height\_m})^2}
\]

- **TDEE (Total Daily Energy Expenditure)**: dựa trên BMR, giới tính, tuổi, chiều cao, cân nặng và mức độ vận động.

- **Điều chỉnh TDEE theo BMI**:

  - BMI > 25 → giảm 15%
  - BMI < 18.5 → tăng 15%

```python
def adjust_tdee(row):
    if row['BMI'] > 25:
        return row['tdee'] * 0.85
    elif row['BMI'] < 18.5:
        return row['tdee'] * 1.15
    else:
        return row['tdee']
- **Nhu cầu protein: 2g / kg trọng lượng cơ thể.
---

## 🔹 Huấn luyện mô hình AI

- **Dữ liệu đầu vào**: BMI, tuổi, giới tính, chiều cao, cân nặng, mức độ vận động
- **Mục tiêu dự đoán**: `target_calories` và `target_protein`
- **Mô hình sử dụng**: `RandomForestRegressor`
- **Đánh giá mô hình**:
  - R² Score
  - MAE (Mean Absolute Error)
  - RMSE (Root Mean Squared Error)

- **Lưu mô hình**:

```python
with open('model_cal.pkl', 'wb') as f:
    pickle.dump(model_cal, f)

with open('model_prot.pkl', 'wb') as f:
    pickle.dump(model_prot, f)

# 🥗 Tổng quan Hệ thống Gợi ý Thực đơn Cá nhân (AI Nutrition Recommender)

Dự án này sử dụng Học máy (Machine Learning) và Thuật toán tối ưu hóa để tạo ra thực đơn ăn uống hàng ngày được cá nhân hóa dựa trên thông tin cơ thể (BMI), mục tiêu cân nặng, và thói quen ăn uống của người dùng.

---

## 💻 Ứng dụng Streamlit và Tính năng Cá nhân hóa

Ứng dụng được triển khai bằng Streamlit, cung cấp giao diện trực quan và các tính năng chính sau:

### ⚙️ Sidebar: Thu thập Thông tin Người dùng

* **Thông số Sinh học:** Nhập **Tuổi**, **Giới tính**, **Chiều cao**, **Cân nặng**.
* **Hoạt động & Mục tiêu:** Chọn **Mức độ vận động** và **Mục tiêu cân nặng** (`giảm cân`, `giữ cân`, `tăng cân`).
* **Sở thích Cá nhân:** Nhập **Thực phẩm muốn tránh / ưu tiên** (sử dụng để lọc công thức).

### 🌟 Tính năng Chính

* **Dự đoán Nhu cầu:** Mô hình Random Forest Regressor dự đoán **Calo** và **Protein** cần thiết hàng ngày (được điều chỉnh theo BMI và Mục tiêu).
* **Lọc Công thức:** Lọc công thức dựa trên sở thích (`avoid_foods`/`prefer_foods`) và nhóm món ăn.
* **Chia Bữa Ăn:** Thực đơn được chia thành 4 bữa: **Bữa sáng**, **Bữa trưa**, **Bữa tối**, **Bữa phụ**.
* **Hiển thị Chi tiết:** Bảng kết quả hiển thị chi tiết **Calo**, **Protein (g)**, **Chất béo (g)** và **Nhóm món** (Cluster) của từng món ăn.
* **Gợi ý Thay thế:** Gợi ý món ăn thay thế dựa trên **nhóm cluster món ăn** (cùng nhóm dinh dưỡng/thành phần).
* **Lịch sử:** Lưu **Lịch sử thực đơn** tối đa 10 lần tạo.

---

## 🔬 Clustering Công thức Món ăn (KMeans)

Để phân loại và gợi ý món thay thế hợp lý, chúng tôi đã áp dụng thuật toán **KMeans Clustering** trên dữ liệu công thức.

* **Kỹ thuật Feature Engineering:**
    * **TF-IDF** trên trường **Nguyên liệu** để nhận diện thành phần nguyên liệu.
    * **Chuẩn hóa (StandardScaler)** các trường **dinh dưỡng chính** (Calo, Protein, Fat, Carb, Fiber, Sodium).
    * Kết hợp hai vector này để tạo ra ma trận đầu vào cho Clustering.
* **Phân loại (5 Cluster):**
    * **Low-Calorie** (ít calo)
    * **High-Protein** (nhiều protein)
    * **Balanced** (cân bằng)
    * **High-Fat** (nhiều chất béo)
    * **Carb-Heavy** (nhiều tinh bột)

### 📊 Ví dụ Kết quả Thực đơn

| Bữa | Món ăn | Calo | Protein (g) | Chất béo (g) | Nhóm món |
| :---: | :---: | :---: | :---: | :---: | :---: |
| Bữa sáng | Trứng ốp la | 300 | 20 | 15 | High-Protein |
| Bữa trưa | Cơm gà | 550 | 35 | 20 | Balanced |
| Bữa tối | Salad cá hồi | 400 | 25 | 18 | Low-Calorie |
| Bữa phụ | Sữa chua | 150 | 8 | 5 | Balanced |

---

## 📈 Hình ảnh Trực quan & Đánh giá Mô hình AI

Các biểu đồ được sử dụng để đánh giá hiệu quả của mô hình Random Forest Regressor:

* **Scatter plot:** So sánh giá trị **thực tế (Ground Truth)** và **dự đoán** cho Calo và Protein (Độ gần của các điểm so với đường $y=x$ đánh giá hiệu suất mô hình).
* **Histogram:** Phân phối **Sai số (Residuals)** giúp kiểm tra độ chệch và độ chính xác của mô hình.
* Các biểu đồ này giúp **đánh giá hiệu quả mô hình AI** trong việc xác định nhu cầu dinh dưỡng cá nhân.

---

## 🚀 Hướng dẫn Chạy Ứng dụng

Để khởi động ứng dụng Streamlit cục bộ, thực hiện các bước sau:

1.  **Cài đặt Thư viện:**
    ```bash
    pip install -r requirements.txt
    ```
2.  **Khởi động Ứng dụng:**
    ```bash
    streamlit run app_multilang_full.py
    ```
3.  **Sử dụng:** Nhập thông tin cá nhân trên **Sidebar**, sau đó nhấn **Tạo thực đơn AI** để xem gợi ý. Món thay thế có thể chọn trực tiếp để cập nhật thực đơn.

---

## ✅ Kết luận và Hướng phát triển

### 📌 Giá trị Hệ thống

Hệ thống giúp người dùng:
* **Dự đoán nhu cầu** Calo/Protein chính xác theo cơ thể và hoạt động.
* **Tạo thực đơn cân bằng** và được **cá nhân hóa** theo mục tiêu.
* **Lọc món ăn** theo sở thích và thói quen.
* **Gợi ý món thay thế** để đa dạng hóa bữa ăn mà vẫn giữ nguyên nhóm dinh dưỡng.

### 🌟 Mở rộng trong Tương lai

* **Dữ liệu:** Thêm dữ liệu dinh dưỡng chi tiết hơn (như chất béo bão hòa, đường, vitamin, khoáng chất).
* **Thuật toán:** Tối ưu thuật toán chọn món bằng các kỹ thuật **Tối ưu hóa (Optimization)** để đảm bảo thực đơn **tổng thể** đạt chính xác các mục tiêu dinh dưỡng vĩ mô.
* **Giao diện:** Hỗ trợ đa ngôn ngữ và giao diện di động.
## 🤝 TÁC GIẢ 👥

Dự án được phát triển bởi:

- **Trần Hồng Quân**

© 2025 NHÓM 20, KHOA CÔNG NGHỆ THÔNG TIN, TRƯỜNG ĐẠI HỌC ĐẠI NAM.

