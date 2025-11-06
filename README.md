
# 🌾 Farm Biogas Prediction App

⚙️ A simple Machine Learning web app that predicts how much biogas a farm can produce each day.

It uses a CatBoost regression model trained on U.S. livestock farm data, considering factors such as:
- Number of dairy cows 🐄
- Digester type ⚗️
- Operational years 📅
- Energy and waste efficiency ⚡🗑️

💡 The model achieves an R² accuracy of approximately 93%, providing reliable predictions to support farmers and sustainability researchers in estimating renewable energy potential and planning eco-friendly farm operations.
This helps farmers and researchers estimate renewable energy potential and plan sustainable farm operations.

## 🧠 Steps to Run the Project
🚀 **1️⃣ Clone the Repository**
```bash
git clone https://github.com/SaikiranC08/farm-biogas-prediction.git
cd farm-biogas-prediction
`````

📦 2️⃣ (Optional) Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate     # For Mac/Linux
venv\Scripts\activate        # For Windows
````


📦 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```
▶️ 4️⃣ Run the Streamlit App
```bash
streamlit run app.py
```

🌐 5️⃣ Open in Browser

Go to 👉 http://localhost:8501


---

####  👨‍💻 Author

Saikiran Chevula

🌱 Exploring Machine Learning for Agricultural Sustainability