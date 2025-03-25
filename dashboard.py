import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(
    page_title="Blood Donation Eligibility Dashboard",
    page_icon="🩸",
    layout="wide"
)

# Load the trained model and preprocessors
@st.cache
def load_model_and_preprocessors():
    try:
        model = joblib.load('/home/intouchs/Donate/blood_donation_model.pkl')
        scaler = joblib.load('/home/intouchs/Donate/scaler.pkl')
        label_encoders = joblib.load('/home/intouchs/Donate/label_encoders.pkl')
        used_features = joblib.load('/home/intouchs/Donate/used_features.pkl')
        return model, scaler, label_encoders, used_features
    except FileNotFoundError:
        st.error("Model files not found. Please make sure you have trained the model and saved the required files.")
        # Return dummy values to prevent errors
        return None, None, None, None

# Function to load dataset
@st.cache
def load_data():
    try:
        # Load data from CSV
        df = pd.read_csv('/home/intouchs/Donate/sampletest_data.csv')
        
        # Display column names for debugging
        print("Available columns:", df.columns.tolist())
        
        # Handle Gender translation (French to English)
        if 'Gender' in df.columns:
            # Map French gender terms to English
            gender_map = {'Homme': 'Male', 'Femme': 'Female'}
            df['Gender'] = df['Gender'].map(gender_map).fillna(df['Gender'])
        
        # Add Blood Type if not present (create from other health indicators)
        if 'Blood Type' not in df.columns:
            # Generate blood types based on sample data
            np.random.seed(42)  # For reproducibility
            df['Blood Type'] = np.random.choice(['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-'], len(df))
        
        # Add eligibility classification if not present
        if 'Eligibility' not in df.columns:
            # Use health indicators to determine eligibility
            # Rule-based approach for demonstration
            conditions = [
                # Permanently ineligible
                ((df['Hemoglobin'] < 12) | 
                 (df['Weight'] < 50) | 
                 (df['Transfused less than 3 months ago'] == 'Yes') |
                 (df["Porteur(HIV,hbs,hcv)"] == 'Yes')),
                
                # Temporarily ineligible
                ((df['Days Since Last Donation'] < 56) | 
                 (df['Unavailable_Recent_Donation'] == 'Yes') |
                 (df['Unavailable_Pregnant'] == 'Yes') |
                 (df['Unavailable_Recent_Childbirth'] == 'Yes')),
                
                # Otherwise eligible
            ]
            choices = ['Définitivement non-eligible', 'Temporairement Non-eligible', 'Eligible']
            df['Eligibility'] = np.select(conditions, choices[:2], default=choices[2])
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

# Function to make predictions
def predict_eligibility(data):
    model, scaler, label_encoders, used_features = load_model_and_preprocessors()
    
    # Check if models are loaded
    if model is None:
        return "Error: Model files not found"
    
    # Convert to DataFrame
    df = pd.DataFrame([data])
    
    # Print the input data for debugging
    print("Input data:", df)
    print("Available label encoders:", list(label_encoders.keys()))
    
    # Map Gender values if needed (French to English or vice versa)
    if 'Gender' in df.columns:
        # Check how gender was encoded in the model
        if 'Gender' in label_encoders:
            gender_classes = label_encoders['Gender'].classes_
            print(f"Model's gender classes: {gender_classes}")
            
            # Map between "Homme"/"Femme" and "Male"/"Female" based on what the model expects
            if 'Homme' in gender_classes:
                gender_map = {'Male': 'Homme', 'Female': 'Femme'}
                df['Gender'] = df['Gender'].map(gender_map).fillna(df['Gender'])
                print(f"Mapped to French: {df['Gender'].values}")
    
    # Identify numerical columns
    numerical_columns = ['Height', 'Weight', 'Age', 'Days Since Last Period', 'Days Since Last Donation', 'Hemoglobin']
    
    # Process categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if col in label_encoders:
            # Debug print
            print(f"Encoding {col} with values: {df[col].values}")
            print(f"Encoder classes: {label_encoders[col].classes_}")
            
            # Now transform with the encoder
            try:
                df[col] = label_encoders[col].transform(df[col].astype(str))
            except ValueError as e:
                print(f"Error transforming {col}: {e}")
                # If still failing, use a default encoded value (0)
                df[col] = 0
    
    # Scale numerical columns
    if all(col in df.columns for col in numerical_columns):
        for col in numerical_columns:
            if col in df.columns:
                if pd.isna(df[col]).any():
                    df[col] = df[col].fillna(0)
        columns_to_scale = [col for col in numerical_columns if col in df.columns]
        if columns_to_scale:
            df[columns_to_scale] = scaler.transform(df[columns_to_scale])
    
    # Ensure all required features are present
    df = df.reindex(columns=used_features, fill_value=0)
    
    # Print processed data for debugging
    print("Processed data for prediction:", df)
    
    # Make prediction
    prediction = model.predict(df)
    
    # Map prediction to eligibility status
    if prediction[0] == 1:
        return 'Eligible'
    elif prediction[0] == 2:
        return 'Temporairement Non-eligible'
    elif prediction[0] == 0:
        return 'Définitivement non-eligible'
    else:
        return 'To be determined'

# Main function
def main():
    # Load data
    df = load_data()
    
    # Check if DataFrame is empty
    if df.empty:
        st.error("No data available. Please check your data file.")
        return
        
    # Sidebar with filters and eligibility checker
    st.sidebar.title("Blood Donation Eligibility")
    
    page = st.sidebar.radio("Navigate", ["Dashboard", "Check Your Eligibility", "About"])
    
    if page == "Dashboard":
        st.title("Blood Donation Analytics Dashboard")
        
        # Display available columns
        # st.sidebar.subheader("Available Data Columns")
        # st.sidebar.write(df.columns.tolist())
        
        # Filters in the sidebar
        st.sidebar.subheader("Filter Data")
        
        # Initialize filtered_df
        filtered_df = df.copy()
        
        # Age filter (if exists)
        if 'Age' in df.columns:
            age_min = int(df['Age'].min()) if not df['Age'].empty else 18
            age_max = int(df['Age'].max()) if not df['Age'].empty else 65
            age_range = st.sidebar.slider("Age Range", age_min, age_max, (age_min, age_max))
            filtered_df = filtered_df[(filtered_df['Age'] >= age_range[0]) & (filtered_df['Age'] <= age_range[1])]
        else:
            st.sidebar.warning("Age column not found in data")
        
        # Direct Gender access
        if 'Gender' in df.columns:
            # Directly use the unique gender values
            unique_genders = df['Gender'].unique()
            gender_option = st.sidebar.multiselect("Gender", unique_genders, default=unique_genders)
            if gender_option:
                filtered_df = filtered_df[filtered_df['Gender'].isin(gender_option)]
        else:
            st.sidebar.warning("Gender column not found in data")
        
        # Blood Type filter (if exists)
        if 'Blood Type' in df.columns:
            unique_blood_types = df['Blood Type'].unique()
            blood_type_option = st.sidebar.multiselect("Blood Type", unique_blood_types, default=unique_blood_types)
            if blood_type_option:
                filtered_df = filtered_df[filtered_df['Blood Type'].isin(blood_type_option)]
        else:
            st.sidebar.warning("Blood Type column not found in data")
        
        # Layout for visualizations - using columns that exist
        st.subheader("Data Overview")
        st.dataframe(filtered_df.head(10))
        
        col1, col2 = st.columns(2)
        
        # Dynamically create visualizations based on available columns
        with col1:
            if 'Eligibility' in df.columns:
                st.subheader("Eligibility Distribution")
                fig = px.pie(filtered_df, names='Eligibility', title='Eligibility Distribution',
                            color_discrete_sequence=px.colors.qualitative.Set3)
                st.plotly_chart(fig, use_container_width=True)
            
            if 'Age' in df.columns and 'Hemoglobin' in df.columns:
                st.subheader("Age vs. Hemoglobin")
                color_col = 'Eligibility' if 'Eligibility' in df.columns else None
                fig = px.scatter(filtered_df, x='Age', y='Hemoglobin', color=color_col,
                                title='Age vs Hemoglobin Level', height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'Blood Type' in df.columns:
                st.subheader("Blood Type Distribution")
                color_col = 'Eligibility' if 'Eligibility' in df.columns else None
                fig = px.histogram(filtered_df, x='Blood Type', color=color_col,
                                title='Blood Type Distribution', height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            if 'Gender' in df.columns and 'Eligibility' in df.columns:
                st.subheader("Eligibility by Gender")
                gender_elig = filtered_df.groupby(['Gender', 'Eligibility']).size().reset_index(name='Count')
                fig = px.bar(gender_elig, x='Gender', y='Count', color='Eligibility',
                            title='Eligibility Status by Gender', height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        # Full width chart
        if 'Days Since Last Donation' in df.columns and 'Eligibility' in df.columns:
            st.subheader("Days Since Last Donation vs. Eligibility")
            fig = px.box(filtered_df, x='Eligibility', y='Days Since Last Donation', 
                        color='Eligibility', title='Days Since Last Donation by Eligibility Status')
            st.plotly_chart(fig, use_container_width=True)

    elif page == "Check Your Eligibility":
        st.title("Blood Donation Eligibility Checker")
        
        # Look at the label encoders to see what gender values were used during training
        model, scaler, label_encoders, used_features = load_model_and_preprocessors()
        
        # Get the actual gender values used in training if available
        gender_values = ["Male", "Female"]  # Default English values
        
        # Check if we should use French values based on the data
        sample_gender = df['Gender'].iloc[0] if not df.empty and 'Gender' in df.columns else None
        if sample_gender in ["Homme", "Femme"]:
            gender_values = ["Homme", "Femme"]
        
        if model is not None and 'Gender' in label_encoders:
            try:
                model_gender_values = label_encoders['Gender'].classes_.tolist()
                # Only override if different from defaults
                if not all(g in gender_values for g in model_gender_values):
                    gender_values = model_gender_values
                    # st.info(f"Using gender values from trained model: {gender_values}")
            except:
                st.warning("Could not retrieve gender values from trained model, using defaults")
        
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Age", 18, 100, 30)
            gender = st.selectbox("Gender", gender_values)
            blood_type = st.selectbox("Blood Type", ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"])
            height = st.number_input("Height (cm)", 140, 220, 170)
        
        with col2:
            weight = st.number_input("Weight (kg)", 40, 200, 70)
            hemoglobin = st.number_input("Hemoglobin (g/dL)", 5.0, 25.0, 14.0, step=0.1)
            days_since_donation = st.number_input("Days Since Last Donation", 0, 1000, 100)
            
            # Only show "Days Since Last Period" for females (French or English)
            days_since_period = None
            if gender in ["Female", "F", "Femme"]:
                days_since_period = st.number_input("Days Since Last Period", 0, 100, 30)
        
        # Create data dictionary
        data = {
            'Age': age,
            'Gender': gender,
            'Blood Type': blood_type,
            'Height': height,
            'Weight': weight,
            'Hemoglobin': hemoglobin,
            'Days Since Last Donation': days_since_donation
        }
        
        if gender in ["Female", "F"] and days_since_period is not None:
            data['Days Since Last Period'] = days_since_period
        
        # Predict button
        if st.button("Check Eligibility"):
            with st.spinner('Analyzing your eligibility...'):
                try:
                    result = predict_eligibility(data)
                    
                    # Display result with appropriate styling
                    if result == 'Eligible':
                        st.success(f"Result: {result} ✅")
                        # Replace balloons with snow animation
                        st.snow()
                        # Add a celebratory message with better styling for visibility
                        st.markdown("""
                        <div style="padding: 15px; border-radius: 10px; background-color: #d4edda; border: 2px solid #c3e6cb; text-align: center; margin: 10px 0;">
                            <h3 style="color: #155724; font-weight: bold; font-size: 24px;">🩸 Thank you for being eligible to donate! 🩸</h3>
                            <p style="color: #155724; font-size: 18px; margin-top: 10px;">Your donation can save up to 3 lives!</p>
                        </div>
                        """, unsafe_allow_html=True)
                    elif result == 'Temporairement Non-eligible':
                        st.warning(f"Result: {result} ⚠️")
                        st.write("You may be eligible to donate later. Check with your local blood center for specific waiting periods.")
                    else:
                        st.error(f"Result: {result} ❌")
                        st.write("Based on the provided information, you may not be eligible to donate blood.")
                
                except Exception as e:
                    st.error(f"Error during prediction: {e}")
    
    else:  # About page
        st.title("About Blood Donation Eligibility")
        
        st.markdown("""
        ## Why Blood Donation Matters
        
        Blood donation is a critical component of healthcare systems worldwide. Every day, blood transfusions save lives in emergency situations, surgical procedures, treatment of diseases like cancer, and many other medical contexts.
        
        ## About This Dashboard
        
        This dashboard provides analytics on blood donation patterns and an AI-powered eligibility checker. The eligibility prediction is based on a machine learning model trained on blood donation data.
        
        ## Eligibility Criteria
        
        Common eligibility criteria for blood donation include:
        
        - Age (typically 18-65 years)
        - Weight (minimum 50 kg in most countries)
        - Hemoglobin levels (minimum 12.5 g/dL for women and 13.5 g/dL for men)
        - Time since last donation (usually 56-112 days depending on the donation type)
        - General health status
        - Absence of certain infectious diseases
        - Not being on certain medications
        
        ## Disclaimer
        
        The eligibility checker provides an estimate based on the model's predictions. For definitive eligibility determination, please consult with your local blood donation center.
        """)

# Run the app
if __name__ == "__main__":
    main() 