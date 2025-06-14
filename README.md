# 🚀 Insurance Fraud Detection Project  

## 📌 Overview  
This project leverages **machine learning** and **web-based tools** to detect fraudulent insurance claims. Built with **Python, MongoDB, Flask**, and hosted on **Render**, the system allows users to upload claim data, dynamically predict fraud probabilities, and download enriched datasets.

## 🎯 Features  
✅ **Customizable Data Upload** – Users select available claim data and generate a custom template.  
✅ **Machine Learning-Based Fraud Detection** – Trained **Random Forest classifier** dynamically adjusts to available features.  
✅ **Flexible File Support** – Accepts both `.csv` and `.xlsx` uploads.  
✅ **MongoDB Integration** – Stores enriched claim datasets with fraud probabilities.  
✅ **Dynamic Model Training** – Adapts predictions based on provided inputs.  
✅ **Automated Versioning** – Prevents duplicate uploads by creating uniquely named datasets.

## 🛠️ Technologies Used  
- **Python** (pandas, sklearn, openpyxl)  
- **Flask** (backend API)  
- **MongoDB Atlas** (database storage)  
- **Render** (cloud hosting)  
- **Machine Learning** (Random Forest classifier)  

## 🔧 Installation & Setup  
Clone this repository:  
```bash
git clone https://github.com/yourusername/insurance-fraud-detection.git
cd insurance-fraud-detection
```
Install dependencies:  
```bash
pip install -r requirements.txt
```
Run Flask locally:  
```bash
python app.py
```
Access the web interface at:  
```
http://127.0.0.1:5000
```

## 📂 Project Structure  
```
📁 insurance-fraud-detection/
 ├── app.py               # Flask app
 ├── model.py             # ML model training & fraud prediction
 ├── requirements.txt     # Dependencies
 ├── static/template_output/  # User-generated templates
 ├── templates/           # HTML frontend
 ├── README.md            # Project documentation
```

## 📊 How It Works  
1️⃣ **Select Features & Download Template**  
Users choose the available claim data fields and download a tailored spreadsheet for input.  

2️⃣ **Populate & Upload Data**  
Users fill in the spreadsheet and upload it via the web interface.  

3️⃣ **Fraud Probability Calculation**  
The machine learning model analyzes each claim, appends **Fraud Probability**, and stores results in MongoDB.  

4️⃣ **Download Enriched Results**  
Users receive an enriched dataset with fraud probabilities appended.

## 🎯 Next Steps  
✔ **Improve Model Accuracy with Larger Datasets**  
✔ **Deploy an External Model Training Pipeline**  
✔ **Enhance Results Page with Insights & Visualization**  


