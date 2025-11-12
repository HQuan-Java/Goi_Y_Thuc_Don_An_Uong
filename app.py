import streamlit as st
import pandas as pd
import numpy as np
import pickle
import random
import ast
import matplotlib.pyplot as plt 
from datetime import datetime 

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# =================== CẤU HÌNH TRANG ===================
st.set_page_config(page_title="AI Gợi Ý Thực Đơn Cá Nhân Hóa", layout="wide")
st.title("AI Gợi Ý Thực Đơn Cá Nhân Hóa")

# =================== LOAD DATA & MODELS ===================
@st.cache_data
def load_data():
    df = pd.read_csv('recipes_clean.csv')
    df.columns = [col.strip().replace(' ', '_').replace('.', '').lower() for col in df.columns]

    df['ingredients_list'] = df['ingredients_list'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else x
    )

    def parse_directions(d):
        if isinstance(d, str):
            try:
                d_dict = ast.literal_eval(d)
                if isinstance(d_dict, dict) and 'directions' in d_dict:
                    return d_dict['directions'].replace('\n', ' ').replace('  ', ' ').strip()
            except:
                return d.replace('\n',' ').replace('  ',' ').strip()
        return d
    df['cooking_directions'] = df['cooking_directions'].apply(parse_directions)

    # --- Clustering ---
    df['ingredients_str'] = df['ingredients_list'].apply(
        lambda x: ' '.join(x) if isinstance(x, list) else ''
    )
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    tfidf_matrix = vectorizer.fit_transform(df['ingredients_str'])

    nutrient_cols = ['calories', 'protein', 'fat', 'carbohydrates', 'fiber', 'sodium']
    df_nutrients = df[nutrient_cols].fillna(0)

    scaler = StandardScaler()
    nutrient_matrix = scaler.fit_transform(df_nutrients)

    combined_matrix = np.hstack([tfidf_matrix.toarray(), nutrient_matrix])
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(combined_matrix)

    cluster_labels = {
        0: 'Low-Calorie (ít calo)',
        1: 'High-Protein (nhiều protein)',
        2: 'Balanced (cân bằng)',
        3: 'High-Fat (nhiều chất béo)',
        4: 'Carb-Heavy (nhiều tinh bột)'
    }
    df['cluster_label'] = df['cluster'].map(cluster_labels)
    return df

@st.cache_resource
def load_models():
    with open('model_cal.pkl', 'rb') as f:
        model_cal = pickle.load(f)
    with open('model_prot.pkl', 'rb') as f:
        model_prot = pickle.load(f)
    return model_cal, model_prot

df_recipes = load_data()
model_cal, model_prot = load_models()

# =================== TÍNH TOÁN DỮ LIỆU ===================
multipliers = {'ít vận động': 1.2, 'nhẹ':1.375, 'vừa':1.55, 'nặng':1.725, 'rất nặng':1.9}

def calculate_bmi(weight_kg, height_cm):
    return round(weight_kg / ((height_cm / 100) ** 2), 1)

def calculate_tdee(weight_kg, height_cm, age, gender, activity_level):
    if gender.lower() in ['nam', 'male']:
        bmr = 88.362 + 13.397 * weight_kg + 4.799 * height_cm - 5.677 * age
    else:
        bmr = 447.593 + 9.247 * weight_kg + 3.098 * height_cm - 4.330 * age
    return round(bmr * multipliers.get(activity_level, 1.55), 0)

def determine_body_status(bmi):
    if bmi < 18.5:
        return "Gầy"
    elif bmi < 25:
        return "Bình thường"
    elif bmi < 30:
        return "Thừa cân"
    else:
        return "Béo phì"

def predict_nutrition_needs(BMI, age, gender, height_cm, weight_kg, activity_level='vừa'):
    gender_num = 1 if gender.lower() in ['nam', 'male'] else 0
    activity_num = multipliers.get(activity_level, 1.55)
    input_data = [[BMI, age, gender_num, height_cm, weight_kg, activity_num]]
    cal = model_cal.predict(input_data)[0]
    prot = model_prot.predict(input_data)[0]
    return round(cal, 0), round(prot, 1)

# =================== LỌC MÓN ĂN ===================
def filter_recipes_by_habit(df, avoid_foods=[], prefer_foods=[]):
    filtered = df.copy()
    if avoid_foods:
        avoid_lower = [f.lower() for f in avoid_foods]
        mask = filtered['ingredients_list'].apply(
            lambda ings: all(food not in ' '.join(ings).lower() for food in avoid_lower)
        )
        filtered = filtered[mask]
    if prefer_foods:
        prefer_lower = [f.lower() for f in prefer_foods]
        mask = filtered['ingredients_list'].apply(
            lambda ings: any(food in ' '.join(ings).lower() for food in prefer_lower)
        )
        temp = filtered[mask]
        if not temp.empty:
            filtered = temp
    return filtered.reset_index(drop=True)

# =================== TẠO THỰC ĐƠN ===================
def generate_daily_meal_plan_ai(df_filtered, weight_kg, height_cm, age, gender,
                                activity_level='vừa', goal='giữ cân',
                                avoid_foods=[], prefer_foods=[], seed=None):
    if seed is None:
        seed = random.randint(0, 999999)
    random.seed(seed)
    np.random.seed(seed)

    bmi = calculate_bmi(weight_kg, height_cm)
    pred_cal, pred_prot = predict_nutrition_needs(bmi, age, gender, height_cm, weight_kg, activity_level)

    if goal in ['giảm cân', 'Lose']:
        pred_cal *= 0.8
    elif goal in ['tăng cân', 'Gain']:
        pred_cal *= 1.15

    # Sử dụng trực tiếp tên bữa ăn tiếng Việt
    meal_keys = [('Bữa sáng', 0.25), ('Bữa trưa', 0.35), ('Bữa tối', 0.30), ('Bữa phụ', 0.10)]
    plan = []
    df_avail = filter_recipes_by_habit(df_filtered, avoid_foods, prefer_foods)
    used_recipes = set()

    for meal, ratio in meal_keys:
        target_cal = pred_cal * ratio
        min_cal, target_cal_max = target_cal * 0.6, target_cal * 1.8
        candidates = df_avail[(df_avail['calories'].between(min_cal, target_cal_max)) &
                              (~df_avail['recipe_name'].isin(used_recipes))]
        if candidates.empty:
            candidates = df_avail[~df_avail['recipe_name'].isin(used_recipes)]
        if candidates.empty:
            continue

        recipe = candidates.sample(n=1, random_state=seed).iloc[0]
        seed += 1

        factor = np.clip(target_cal / max(recipe.get('calories', 1), 1), 0.7, 1.3)
        scaled_cal = recipe.get('calories', 0) * factor
        scaled_prot = recipe.get('protein', 0) * factor
        scaled_fat = recipe.get('fat', 0) * factor

        raw_name = recipe['recipe_name']
        used_recipes.add(raw_name)

        plan.append({
            'Bữa': meal,  # Sử dụng trực tiếp tên bữa ăn tiếng Việt
            'Món ăn': raw_name,
            'Calo': int(round(scaled_cal)),
            'Protein (g)': scaled_prot,
            'Chất béo (g)': scaled_fat,
            'Nhóm món': recipe.get('cluster_label', 'N/A')
        })
    return pd.DataFrame(plan), int(round(pred_cal)), round(pred_prot, 1), round(bmi, 1)

# =================== SESSION STATE ===================
if 'plan_df' not in st.session_state:
    st.session_state.plan_df = pd.DataFrame()
if 'pred_cal' not in st.session_state:
    st.session_state.pred_cal = 0
if 'pred_prot' not in st.session_state:
    st.session_state.pred_prot = 0
if 'bmi' not in st.session_state:
    st.session_state.bmi = 0
if 'history' not in st.session_state:
    st.session_state.history = []  # List chứa lịch sử thực đơn
if 'alt_view' not in st.session_state: # Thêm state để kiểm soát món thay thế đang xem
    st.session_state.alt_view = None

# =================== GIAO DIỆN NGƯỜI DÙNG ===================
with st.sidebar:
    st.header("Thông tin người dùng")
    age = st.number_input("Tuổi", min_value=18, max_value=80)
    gender = st.selectbox("Giới tính", ['Nam', 'Nữ'])
    height_cm = st.number_input("Chiều cao (cm)", min_value=100, max_value=250)
    weight_kg = st.number_input("Cân nặng (kg)", min_value=30, max_value=200)
    activity_options = ['ít vận động', 'nhẹ', 'vừa', 'nặng', 'rất nặng']
    activity_level = st.selectbox("Mức độ vận động", activity_options)
    goal_options = ['giữ cân', 'giảm cân', 'tăng cân']
    goal = st.selectbox("Mục tiêu", goal_options)
    avoid_foods = st.text_input("Thực phẩm muốn tránh (ngăn cách bằng dấu phẩy)").split(',')
    prefer_foods = st.text_input("Thực phẩm ưu tiên (ngăn cách bằng dấu phẩy)").split(',')

# =================== NÚT TẠO THỰC ĐƠN ===================
if st.button("Tạo thực đơn AI"):
    plan_df, pred_cal, pred_prot, bmi = generate_daily_meal_plan_ai(
        df_filtered=df_recipes,
        weight_kg=weight_kg,
        height_cm=height_cm,
        age=age,
        gender=gender,
        activity_level=activity_level,
        goal=goal,
        avoid_foods=[f.strip() for f in avoid_foods if f.strip()],
        prefer_foods=[f.strip() for f in prefer_foods if f.strip()]
    )
    st.session_state.plan_df = plan_df
    st.session_state.pred_cal = pred_cal
    st.session_state.pred_prot = pred_prot
    st.session_state.bmi = bmi
    st.session_state.alt_view = None # Reset view khi tạo thực đơn mới
    
    # Lưu vào lịch sử
    history_entry = {
        'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'plan_df': plan_df.copy(),
        'pred_cal': pred_cal,
        'pred_prot': pred_prot,
        'bmi': bmi
    }
    st.session_state.history.append(history_entry)
    if len(st.session_state.history) > 10:  # Giới hạn 10 mục
        st.session_state.history.pop(0)

# =================== HIỂN THỊ KẾT QUẢ ===================
if not st.session_state.plan_df.empty:
    plan_df = st.session_state.plan_df
    body_status = determine_body_status(st.session_state.bmi)
    tdee = calculate_tdee(weight_kg, height_cm, age, gender, activity_level)

    st.subheader("Thông tin cơ bản")
    st.markdown(f"- **BMI**: {st.session_state.bmi:.1f} ({body_status})")
    st.markdown(f"- **TDEE**: {tdee} kcal/ngày")
    st.markdown(f"- **Nhu cầu protein**: {st.session_state.pred_prot:.1f} g/ngày")
    st.markdown(f"- **Nhu cầu calo (AI dự đoán)**: {st.session_state.pred_cal} kcal/ngày")

    st.subheader("Thực đơn AI hôm nay")
    
    # Định dạng cột Protein và Fat để loại bỏ số 0 thừa
    def format_nutrition_value(x):
      try:
        x = float(x)
        if x == int(x):
            return str(int(x))
        else:
            return str(round(x, 1))
      except:
        return str(x)  # Nếu lỗi, giữ nguyên chuỗi
    
    plan_df_display = plan_df.copy() # Sử dụng copy để định dạng hiển thị
    plan_df_display['Protein (g)'] = plan_df_display['Protein (g)'].apply(format_nutrition_value)
    plan_df_display['Chất béo (g)'] = plan_df_display['Chất béo (g)'].apply(format_nutrition_value)
    
    st.table(plan_df_display)

    # =================== GỠI Ý MÓN THAY THẾ (Cập nhật logic) ===================
    st.subheader("Gợi ý món thay thế")
    for idx, row in st.session_state.plan_df.iterrows():
        cluster = df_recipes[df_recipes['recipe_name'] == row['Món ăn']]['cluster'].values
        if len(cluster) > 0:
            cluster_val = cluster[0]
            # Lấy 2 món thay thế cùng nhóm
            alternatives = df_recipes[(df_recipes['cluster'] == cluster_val) & 
                                      (df_recipes['recipe_name'] != row['Món ăn'])].sample(min(2, len(df_recipes[df_recipes['cluster'] == cluster_val])), random_state=42)
            if not alternatives.empty:
                with st.expander(f"Thay thế cho {row['Món ăn']} ({row['Bữa']})"):
                    for _, alt in alternatives.iterrows():
                        st.write(f"- **{alt['recipe_name']}** (Calo: {alt['calories']:.0f}, Protein: {alt['protein']:.1f}g)")
                        
                        col1, col2 = st.columns([1, 1])
                        
                        # --- Nút Xem chi tiết (Cập nhật Session State) ---
                        with col1:
                            if st.button(f"Xem chi tiết", key=f"view_alt_{row['Món ăn']}_{alt['recipe_name']}"):
                                st.session_state.alt_view = alt['recipe_name']
                                st.rerun() 

                        # --- Nút Chọn món thay thế (Duy trì logic cũ) ---
                        with col2:
                            if st.button(f"Chọn món", key=f"select_alt_{row['Món ăn']}_{alt['recipe_name']}"):
                                # Lấy recipe mới và tính lại dinh dưỡng
                                new_recipe = df_recipes[df_recipes['recipe_name'] == alt['recipe_name']].iloc[0]
                                meal = row['Bữa']
                                ratio = 0.25 if meal == 'Bữa sáng' else 0.35 if meal == 'Bữa trưa' else 0.30 if meal == 'Bữa tối' else 0.10
                                target_cal = st.session_state.pred_cal * ratio
                                factor = np.clip(target_cal / max(new_recipe.get('calories', 1), 1), 0.7, 1.3)
                                scaled_cal = new_recipe.get('calories', 0) * factor
                                scaled_prot = new_recipe.get('protein', 0) * factor
                                scaled_fat = new_recipe.get('fat', 0) * factor
                                
                                # Cập nhật hàng với dinh dưỡng mới (dùng idx)
                                st.session_state.plan_df.loc[idx, ['Món ăn', 'Calo', 'Protein (g)', 'Chất béo (g)', 'Nhóm món']] = [
                                    alt['recipe_name'], int(round(scaled_cal)), scaled_prot, scaled_fat, alt['cluster_label']
                                ]
                                st.session_state.alt_view = None # Xóa trạng thái xem chi tiết sau khi chọn món
                                st.success(f"Đã thay thế {row['Món ăn']} bằng {alt['recipe_name']}!")
                                st.rerun() 
                                
                        st.markdown("---") 

    # =================== XỬ LÝ HIỂN THỊ CHI TIẾT MÓN THAY THẾ ===================
    if st.session_state.alt_view:
        alt_name = st.session_state.alt_view
        st.subheader(f"Chi tiết món thay thế: {alt_name} 📝")
        
        # Tìm chi tiết món trong DataFrame gốc
        recipe = df_recipes[df_recipes['recipe_name'] == alt_name].iloc[0]
        
        st.markdown("**Nguyên liệu:**")
        ings = recipe.get('ingredients_list', [])
        if isinstance(ings, str):
            try:
                ings = ast.literal_eval(ings)
            except:
                ings = [ings]
        for ing in ings:
            st.markdown(f"- {ing}")

        st.markdown("**Hướng dẫn nấu:**")
        st.markdown(recipe.get('cooking_directions', ''))
        
        # Thêm nút đóng 
        if st.button("Ẩn chi tiết món thay thế", key="hide_alt_view"):
            st.session_state.alt_view = None
            st.rerun()
            
    # ---
    
    st.subheader("Chi tiết từng món") # Vị trí cũ của phần chi tiết, đảm bảo món mới được cập nhật
    for _, row in plan_df.iterrows():
        recipe = df_recipes[df_recipes['recipe_name'] == row['Món ăn']]
        if recipe.empty:
            st.warning(f"Không tìm thấy chi tiết món: {row['Món ăn']}")
            continue
        recipe = recipe.iloc[0]
        with st.expander(f"{row['Món ăn']} ({row['Bữa']})"):
            st.markdown("**Nguyên liệu:**")
            ings = recipe.get('ingredients_list', [])
            if isinstance(ings, str):
                try:
                    ings = ast.literal_eval(ings)
                except:
                    ings = [ings]
            # Hiển thị nguyên liệu gốc mà không dịch
            for ing in ings:
                st.markdown(f"- {ing}")

            st.markdown("**Hướng dẫn nấu:**")
            # Hiển thị hướng dẫn nấu gốc mà không dịch
            st.markdown(recipe.get('cooking_directions', ''))


    # =================== BIỂU ĐỒ DINH DƯỠNG ===================
    if not st.session_state.plan_df.empty:
        st.subheader("Biểu đồ dinh dưỡng tổng quan")
        
        # Tính tổng
        total_cal = st.session_state.plan_df['Calo'].sum()
        # Chuyển Protein/Fat về dạng số trước khi tính tổng
        total_prot = st.session_state.plan_df['Protein (g)'].apply(lambda x: float(x) if isinstance(x, str) and x.replace('.', '', 1).isdigit() else x).astype(float).sum()
        total_fat = st.session_state.plan_df['Chất béo (g)'].apply(lambda x: float(x) if isinstance(x, str) and x.replace('.', '', 1).isdigit() else x).astype(float).sum()
        
        # Biểu đồ tròn cho tổng dinh dưỡng
        fig1, ax1 = plt.subplots()
        ax1.pie([total_cal, total_prot * 4, total_fat * 9],  # Chuyển protein/fat sang calo tương đương
                labels=['Calo', 'Protein (kcal equiv)', 'Chất béo (kcal equiv)'],
                autopct='%1.1f%%', startangle=90)
        ax1.axis('equal')
        st.pyplot(fig1)
        
        # Biểu đồ cột cho calo theo bữa
        fig2, ax2 = plt.subplots()
        ax2.bar(st.session_state.plan_df['Bữa'], st.session_state.plan_df['Calo'])
        ax2.set_ylabel('Calo')
        ax2.set_title('Phân bố Calo theo bữa ăn')
        st.pyplot(fig2)

     # =================== LỊCH SỬ THỰC ĐƠN ===================
    st.subheader("Lịch sử thực đơn")
    if st.session_state.history:
        for i, entry in enumerate(reversed(st.session_state.history)):  # Hiển thị từ mới nhất
            with st.expander(f"Thực đơn {entry['date']} (Calo: {entry['pred_cal']}, BMI: {entry['bmi']})"):
                st.write(f"Protein: {entry['pred_prot']}g")
                # Định dạng lại plan_df trong lịch sử trước khi hiển thị
                history_df_display = entry['plan_df'].copy()
                history_df_display['Protein (g)'] = history_df_display['Protein (g)'].apply(format_nutrition_value)
                history_df_display['Chất béo (g)'] = history_df_display['Chất béo (g)'].apply(format_nutrition_value)
                st.table(history_df_display)