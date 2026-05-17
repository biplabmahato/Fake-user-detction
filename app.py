import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

@st.cache_resource
def load_model_scaler():
    try:
        model = joblib.load('fake_user_rf_model.pkl')
        scaler = joblib.load('fake_user_scaler.pkl')
        return model, scaler
    except FileNotFoundError as e:
        st.error(f"Model files not found: {e}")
        st.error("Please run the training script first to generate the required models.")
        return None, None


def feature_engineering(df):
    """Same feature engineering as training"""
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

def generate_example_csv():
    data = {
        "user": ["user1", "user1", "user2", "user2", "user3"],
        "timestamp": [
            "2025-10-01 08:45:00",
            "2025-10-01 09:15:00",
            "2025-10-02 10:00:00",
            "2025-10-02 11:30:00",
            "2025-10-03 09:00:00"
        ],
        "login_result": ["Success", "Failure", "Success", "Failure", "Success"],
        "mfa_used": ["Yes", "No", "Yes", "No", "Yes"],
        "source_ip": ["192.168.1.1", "192.168.1.2", "10.0.0.1", "10.0.0.1", "172.16.0.5"],
        "user_agent": ["Chrome", "Chrome", "Firefox", "Firefox", "Safari"],
        "region": ["US", "US", "EU", "EU", "APAC"]
    }
    df = pd.DataFrame(data)
    return df.to_csv(index=False)

def main():
    st.title("🔍 Fake User Detection in Cloud Activities")
    
    st.markdown("## Example CSV Template")
    st.markdown(
        "Download this example CSV to structure your input data correctly (columns and sample values)."
    )
    st.download_button(
        label="📄 Download Example CSV",
        data=generate_example_csv(),
        file_name="example_fake_user_data.csv",
        mime="text/csv"
    )
    
    # Load model and scaler
    model, scaler = load_model_scaler()
    if model is None:
        st.stop()
    
    st.success("✅ Model loaded successfully!")
    
    uploaded_file = st.file_uploader("Choose CSV file with cloud activity logs", type=["csv"])
    
    if uploaded_file is not None:
        try:
            if "df_data" not in st.session_state:
                df = pd.read_csv(uploaded_file)
                st.session_state.df_data = df
            else:
                df = st.session_state.df_data
            
            st.subheader("Data Preview")
            st.dataframe(df.head())
            
            # Validate required columns
            required_cols = ['user', 'timestamp', 'login_result', 'mfa_used', 'source_ip', 'user_agent', 'region']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                st.error(f"Missing required columns: {missing_cols}")
                st.stop()
            
            # Data exploration
            st.subheader("Data Exploration")
            st.write(df.describe(include='all'))
            fig_hist = px.histogram(df, x='login_result', title="Login Result Distribution")
            st.plotly_chart(fig_hist)
            
            # Feature engineering and prediction
            # with st.spinner("Processing features and making predictions..."):
            #     features = feature_engineering(df)
            #     st.session_state.features = features
                
            #     # Adjustable threshold slider
            #     threshold = st.slider(
            #         "Adjust Fake User Probability Threshold",
            #         min_value=0.0,
            #         max_value=1.0,
            #         value=0.5,
            #         step=0.01,
            #         help="Set threshold to classify fake users (default 0.5)"
            #     )
                
            #     X_scaled = scaler.transform(features)
            #     pred_proba = model.predict_proba(X_scaled)[:, 1]
            #     predictions = (pred_proba >= threshold).astype(int)
            
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
            
            # User filtering for detailed results
            st.subheader("Detailed Results with Filtering")
            
            # Filters
            # user_filter = st.text_input("Filter by User (substring match):", "")
            # prob_filter = st.slider("Minimum Fake Probability:", 0.0, 1.0, 0.0, 0.01)
            # filtered_df = results_df[
            #     (results_df.index.str.contains(user_filter)) &
            #     (results_df['Fake_Probability'] >= prob_filter)
            # ]
            # st.dataframe(filtered_df.sort_values('Fake_Probability', ascending=False))
            
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
