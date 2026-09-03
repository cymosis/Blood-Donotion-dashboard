# 🩸 Blood Donation Eligibility Dashboard

A comprehensive analytics dashboard and AI-powered eligibility checker for blood donation screening using machine learning and data visualization.

## 📋 Overview

This dashboard provides:
- **Analytics Dashboard**: Visualize blood donation patterns and donor demographics
- **Eligibility Checker**: AI-powered prediction of blood donation eligibility
- **Data Insights**: Comprehensive analysis of donor health profiles and eligibility criteria

## 🚀 Features

- **Interactive Dashboard**: Filter and explore donor data by age, gender, and blood type
- **ML-Based Predictions**: Machine learning model for eligibility classification
- **Real-time Visualizations**: Plotly charts for eligibility distribution, blood type analysis, and health metrics
- **Gender-Specific Screening**: Women-specific health checks (menstrual cycle, pregnancy status)
- **Responsive Design**: Works seamlessly on desktop and mobile devices

## 📊 Dashboard Pages

### 1. **Dashboard**
- Overview of all donor data with real-time filtering
- Eligibility distribution pie chart
- Age vs. Hemoglobin scatter plot
- Blood type distribution histogram
- Eligibility breakdown by gender
- Days since last donation analysis

### 2. **Check Your Eligibility**
- Personal eligibility prediction tool
- Input your health metrics:
  - Age, Gender, Blood Type
  - Height, Weight, Hemoglobin levels
  - Days since last donation
  - Gender-specific fields (for women)
- Instant AI-powered eligibility result

### 3. **About**
- Information about blood donation importance
- Common eligibility criteria
- Disclaimer and local guidance

## 🛠️ Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io/) - Python web framework
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Machine Learning**: scikit-learn
- **Model Persistence**: joblib

## 📦 Installation

### Prerequisites
- Python 3.8+
- pip or conda

### Local Setup

1. **Clone the repository**
```bash
git clone https://github.com/cymosis/Blood-Donotion-dashboard.git
cd Blood-Donotion-dashboard
```

2. **Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Ensure model files are in place**
```
├── blood_donation_model.pkl
├── scaler.pkl
├── label_encoders.pkl
├── used_features.pkl
└── sampletest_data.csv
```

5. **Run the dashboard**
```bash
streamlit run dashboard.py
```

Visit `http://localhost:8501` in your browser.

## 📋 Requirements

```
streamlit==1.28.1
pandas==2.0.3
numpy==1.24.3
joblib==1.3.1
matplotlib==3.7.2
seaborn==0.12.2
plotly==5.15.0
scikit-learn==1.3.0
scipy==1.11.1
geopy==2.3.0
```

## 🚀 Deployment

### Option 1: Streamlit Cloud (Recommended)
1. Push code to GitHub
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Click "New app" and select this repository
4. Choose `dashboard.py` as the main file
5. Deploy!

**Live URL**: Your app will be available at `https://[your-username]-blood-donation-dashboard.streamlit.app`

### Option 2: Docker + Cloud Hosting

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Deploy to:
- AWS (ECS, App Runner)
- Google Cloud Run
- Azure Container Instances
- Render.com
- Heroku

### Option 3: Traditional Web Server
- Deploy to any Python-capable hosting (PythonAnywhere, Replit, etc.)

## 📊 Data Structure

The dashboard expects the following data columns:
- **Demographics**: Age, Gender, Blood Type, Height, Weight
- **Health Metrics**: Hemoglobin, Days Since Last Donation
- **Women-Specific**: Days Since Last Period (optional)
- **Eligibility Status**: Classification (Eligible, Temporarily Ineligible, Permanently Ineligible)

## 🤖 Machine Learning Model

The eligibility prediction model is trained on:
- **Features**: Age, Gender, Health Metrics, Previous Donation History
- **Target**: Eligibility Classification (3 classes)
- **Algorithm**: Trained classifier (loaded from `blood_donation_model.pkl`)

### Eligibility Categories:
- **Eligible (✅)**: Meets all criteria for blood donation
- **Temporarily Non-eligible (⚠️)**: Can donate after waiting period
- **Permanently Non-eligible (❌)**: Current health conditions prevent donation

## 📝 Data Files

- `Data_Visualization.ipynb`: Jupyter notebook with data exploration and preprocessing
- `dashboard.py`: Main Streamlit application
- `sampletest_data.csv`: Sample dataset for testing
- Model files: `.pkl` files for ML model and preprocessors

## ⚠️ Disclaimer

This dashboard provides an **estimate** based on machine learning predictions. For definitive eligibility determination, please consult with your **local blood donation center** or healthcare provider.

Always follow your country's blood donation guidelines and regulations.

## 📧 Contact & Support

- **Repository**: [Blood-Donotion-dashboard](https://github.com/cymosis/Blood-Donotion-dashboard)
- **Author**: [@cymosis](https://github.com/cymosis)

## 📄 License

This project is open source. Please check the repository for license details.

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 🙏 Acknowledgments

- Blood donation eligibility criteria sourced from international guidelines
- Data visualization powered by Plotly
- Built with Streamlit for rapid development

---

**Made with ❤️ for blood donors and healthcare systems worldwide**
