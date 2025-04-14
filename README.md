# 🏥 Triage AI - Intelligent Patient Prioritization

Triage AI is an intelligent clinical decision support tool that uses machine learning to prioritize patients based on the severity of their condition. Designed to assist healthcare professionals in emergency scenarios, the system ensures faster attention to high-risk patients, thereby improving outcomes and reducing strain on resources.

---

## 🚀 Demo

🔗 [Try the demo here](https://samyuktha-12-triage-ai-app-app-s4lylf.streamlit.app/)
<br>
<br>
![image](https://github.com/user-attachments/assets/66669315-0cb8-46e9-a733-86ba80872e87)
![image](https://github.com/user-attachments/assets/ae100635-d14f-4513-a7e7-6e8aed382690)


---

## 📌 Features

- 🧠 **Machine Learning–based Patient Triage**
- 📊 **SHAP Explanations** for model interpretability
- 🩺 **Human-in-the-Loop**: AI as an assistant, not a replacement
- ✅ **Fairness & Trust** powered by IBM’s [AI Fairness 360 (AIF360)](https://aif360.mybluemix.net/) and [XAI](https://xai-tools.mybluemix.net/)
- 🖥️ Streamlit-based intuitive user interface

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Backend**: Python (scikit-learn, SHAP)
- **Explainability**: SHAP
- **Model**: RandomForestClassifier
- **Deployment**: Streamlit Cloud

---

## 🧪 Local Setup Instructions

### ✅ Prerequisites

- Python 3.8+
- pip (Python package installer)

### 🧰 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/samyuktha-12/triage-ai-app.git
   cd triage-ai-app
   ```
2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```
3. Train the Model
   ```bash
   python model/train_baseline_model.py
   ```
4. Run the App
   ```bash
   streamlit run app.py
   ```

## 🤖 Model Explanation

Triage AI uses SHAP (SHapley Additive exPlanations) to identify the top contributing feature for each patient’s predicted risk. This ensures transparency in decision-making and helps build trust with medical professionals.

When you interact with the app, you'll see insights like:

**Top contributing factor for Patient 1:**
Heart_Rate (SHAP value: +0.134)

The SHAP value indicates the impact of the feature on the predicted risk. A higher positive SHAP value means the feature contributes more to a higher risk score, whereas a negative value indicates a reduction in the risk score.


## 📊 Dataset Explanation

The dataset used in this project is derived from a cross-sectional retrospective study conducted on 1267 adult patients who were admitted to two emergency departments between October 2016 and September 2017. The dataset includes 24 variables, including patient vital signs, chief complaints, and clinical outcomes. Three triage experts determined the true KTAS (Korean Triage and Acuity Scale) based on their experience and expertise in emergency care.

### **Variables:**
- **Sex**: Sex of the patient (1 = Female, 2 = Male)
- **Age**: Age of the patient
- **Arrival Mode**: Type of transportation to the hospital (1 = Walking, 2 = Public Ambulance, 3 = Private Vehicle, 4 = Private Ambulance, 5-7 = Other)
- **Injury**: Whether the patient is injured or not (1 = No, 2 = Yes)
- **Chief Complaint**: The patient's main complaint
- **Mental**: The mental state of the patient (1 = Alert, 2 = Verbal Response, 3 = Pain Response, 4 = Unresponsive)
- **Pain**: Whether the patient has pain (1 = Yes, 0 = No)
- **NRS Pain**: Nurse's assessment of pain for the patient
- **SBP**: Systolic Blood Pressure
- **DBP**: Diastolic Blood Pressure
- **HR**: Heart Rate
- **RR**: Respiratory Rate
- **BT**: Body Temperature

### **Categorical Data:**
Some numerical values in the dataset are actually categorical, and the following variables are treated accordingly:

- **Reason Visit**: Injury [1 = No, 2 = Yes]
- **Gender**: Sex [1 = Female, 2 = Male]
- **Pain**: Pain [1 = Yes, 0 = No]
- **Mental State**: Mental [1 = Alert, 2 = Verbal Response, 3 = Pain Response, 4 = Unresponsive]
- **Type of ED**: Group [1 = Local ED 3rd Degree, 2 = Regional ED 4th Degree]
- **Mode of Arrival**: Arrival Mode [1 = Walking, 2 = Public Ambulance, 3 = Private Vehicle, 4 = Private Ambulance, 5-7 = Other]
- **Disposition**: Disposition [1 = Discharge, 2 = Admission to Ward, 3 = Admission to ICU, 4 = Discharge, 5 = Transfer, 6 = Death, 7 = Surgery]
- **KTAS**: KTAS [1,2,3 = Emergency, 4,5 = Non-Emergency]

These variables play a critical role in determining the urgency and severity of the patient's condition, helping emergency medical professionals prioritize care efficiently.

## 📞 Contact

- **Email**: [samyuktha1262@gmail.com](mailto:samyuktha1262@gmail.com)
- **LinkedIn**: [Samyuktha M S](https://www.linkedin.com/in/samyuktha-m-s)

