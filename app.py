import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from datetime import datetime
import matplotlib.pyplot as plt

@st.cache_resource
def load_model_scaler():
    try:
        model = joblib.load('fake_user_rf_model.pkl')
        scaler = joblib.load('fake_user_scaler.pkl')
        
        # --- scikit-learn Version Mismatch Patch ---
        if hasattr(model, 'estimators_'):
            for estimator in model.estimators_:
                if not hasattr(estimator, 'monotonic_cst'):
                    estimator.monotonic_cst = None
        
        return model, scaler
    except FileNotFoundError as e:
        st.error(f"Model files not found: {e}")
        st.error("Please run the training script first to generate the required models.")
        return None, None


def feature_engineering(df_input):
    """Same feature engineering as training - operating on a local copy"""
    df = df_input.copy()  
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    user_groups = df.groupby('user')
    
    features = pd.DataFrame()
    features['total_events'] = user_groups.size()
    features['failed_logins'] = user_groups.apply(lambda x: (x['login_result'] == 'Failure').sum())
    features['success_logins'] = user_groups.apply(lambda x: (x['login_result'] == 'Success').sum())
    features['mfa_used_rate'] = user_groups.apply(lambda x: (x['mfa_used'] == 'Yes').mean())
    features['unique_ips'] = user_groups['source_ip'].nunique()
    features['unique_agents'] = user_groups['user_agent'].nunique()
    features['unique_regions'] = user_groups['region'].nunique()
    features['first_event'] = user_groups['timestamp'].min()
    features['last_event'] = user_groups['timestamp'].max()
    features['active_days'] = (features['last_event'] - features['first_event']).dt.days + 1
    features['events_per_day'] = features['total_events'] / features['active_days']
    
    # Additional features
    features['failure_rate'] = features['failed_logins'] / features['total_events']
    features['success_rate'] = features['success_logins'] / features['total_events']
    features['ip_diversity'] = features['unique_ips'] / features['total_events']
    
    features = features.drop(columns=['first_event', 'last_event']).fillna(0)
    return features

def main():
    st.title("🔍 Fake User Detection in Cloud Activities")
    
    # Load model and scaler
    model, scaler = load_model_scaler()
    if model is None:
        st.stop()
    
    st.success("✅ Model loaded successfully!")
    
    uploaded_file = st.file_uploader("Choose CSV file with cloud activity logs", type=["csv", "txt"])
    
    if uploaded_file is not None:
        try:
            # Defined required columns
            required_cols = ['user', 'timestamp', 'login_result', 'mfa_used', 'source_ip', 'user_agent', 'region']
            
            # Read first line to check headers
            df_check = pd.read_csv(uploaded_file, nrows=2)
            uploaded_file.seek(0) # Reset file pointer
            
            # Smart Header Auto-Fixer Logic
            has_matching_headers = all(col in df_check.columns for col in required_cols)
            
            if not has_matching_headers:
                st.warning("⚠️ Column headers missing or mismatched! Automatically applying required schema layout...")
                # Load with manually assigned names if the file has no valid header row
                df_raw = pd.read_csv(uploaded_file, names=required_cols, header=None)
            else:
                df_raw = pd.read_csv(uploaded_file)
                
            st.session_state.df_data = df_raw.copy()
            df = df_raw.copy()
            
            st.subheader("Data Preview")
            st.dataframe(df.head())
            
            # Data exploration
            st.subheader("Data Exploration")
            st.write(df.describe(include='all'))
            fig_hist = px.histogram(df, x='login_result', title="Login Result Distribution")
            st.plotly_chart(fig_hist)
            
            # Feature engineering and prediction
            with st.spinner("Processing features and making predictions..."):
                features = feature_engineering(df)
                st.session_state.features = features.copy()
                
                X_scaled = scaler.transform(features)
                pred_proba = model.predict_proba(X_scaled)[:, 1]
                predictions = (pred_proba >= 0.5).astype(int)
            
            # Prepare results
            results_df = features.copy()
            results_df['Predicted_Fake_User'] = predictions
            results_df['Fake_Probability'] = pred_proba
            results_df['Predicted_Fake_User'] = results_df['Predicted_Fake_User'].map({0: 'Real', 1: 'Fake'})
            
            st.subheader("🎯 Prediction Results")
            col1, col2 = st.columns(2)
            with col1:
                fake_count = (results_df['Predicted_Fake_User'] == 'Fake').sum()
                st.metric("Fake Users Detected", fake_count)
            with col2:
                real_count = (results_df['Predicted_Fake_User'] == 'Real').sum()
                st.metric("Real Users", real_count)
            
            pred_counts = results_df['Predicted_Fake_User'].value_counts()
            fig_pie = px.pie(values=pred_counts.values, names=pred_counts.index, title="Prediction Distribution")
            st.plotly_chart(fig_pie)
            
            # Feature importance visualization
            st.subheader("🌟 Feature Importance")
            importances = model.feature_importances_
            fig, ax = plt.subplots()
            ax.barh(features.columns, importances)
            ax.set_xlabel("Importance")
            ax.set_title("Feature Importance from Model")
            st.pyplot(fig)
            
            # Global detailed results view
            st.subheader("Detailed Breakdown")
            st.dataframe(results_df.sort_values('Fake_Probability', ascending=False))
            
            # Download results
            csv = results_df.to_csv(index=True)
            st.download_button(
                "📥 Download Results CSV",
                csv,
                f"fake_user_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv"
            )
            
        except Exception as e:
            st.error(f"Error processing file: {e}")


if __name__ == "__main__":
    main()
