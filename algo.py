import os
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def generate_cloud_user_activity_dataset(n_samples=5000, n_users=1000):
    """Generate synthetic cloud user activity dataset for fake user detection"""
    np.random.seed(42)
    
    users = [f"user_{i:04d}" for i in range(n_users)]
    data = []
    base_time = datetime.now() - timedelta(days=30)
    
    for _ in range(n_samples):
        user = np.random.choice(users)
        user_id = int(user.split('_')[1])
        is_fake = np.random.random() < (0.8 if user_id >= 800 else 0.1)
        
        if is_fake:
            # Fake user patterns
            login_success_prob = 0.3  # Lower success rate
            mfa_usage_prob = 0.1      # Less likely to use MFA
            unique_ip_factor = 3      # More unique IPs
            region_variety = 5        # More regions
        else:
            # Real user patterns
            login_success_prob = 0.85  # Higher success rate
            mfa_usage_prob = 0.7       # More likely to use MFA
            unique_ip_factor = 1       # Fewer unique IPs
            region_variety = 2         # Fewer regions
        
        timestamp = base_time + timedelta(
            days=np.random.randint(0, 30),
            hours=np.random.randint(0, 24),
            minutes=np.random.randint(0, 60)
        )
        
        login_result = "Success" if np.random.random() < login_success_prob else "Failure"
        mfa_used = "Yes" if np.random.random() < mfa_usage_prob else "No"
        ip_base = np.random.choice(['192.168.', '10.0.', '172.16.', '203.0.', '8.8.'])
        source_ip = f"{ip_base}{np.random.randint(1, 255)}.{np.random.randint(1, 255)}"
        
        agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
            "Python-requests/2.25.1",
            "curl/7.68.0"
        ]
        user_agent = np.random.choice(agents)
        
        regions = ['us-east-1', 'us-west-2', 'eu-west-1', 'ap-southeast-1', 'ca-central-1']
        region = np.random.choice(regions[:region_variety])
        
        data.append({
            'user': user,
            'timestamp': timestamp,
            'login_result': login_result,
            'mfa_used': mfa_used,
            'source_ip': source_ip,
            'user_agent': user_agent,
            'region': region,
            'is_fake': 1 if is_fake else 0
        })
    
    return pd.DataFrame(data)

def feature_engineering(df):
    """Feature engineering matching your Streamlit app"""
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
    
    # Additional behavioral features
    features['failure_rate'] = features['failed_logins'] / features['total_events']
    features['success_rate'] = features['success_logins'] / features['total_events']
    features['ip_diversity'] = features['unique_ips'] / features['total_events']
    
    features = features.drop(columns=['first_event', 'last_event']).fillna(0)
    
    # Get labels for each user
    user_labels = df.groupby('user')['is_fake'].first()
    features['is_fake'] = features.index.map(user_labels)
    
    return features

def main():
    print("Generating synthetic cloud user activity dataset...")
    df = generate_cloud_user_activity_dataset(n_samples=5000, n_users=1000)
    
    print(f"Generated dataset with {len(df)} log entries for {df['user'].nunique()} users")
    df.to_csv('cloud_user_activity_logs.csv', index=False)
    
    print("Performing feature engineering...")
    features_df = feature_engineering(df)
    
    X = features_df.drop('is_fake', axis=1)
    y = features_df['is_fake']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Training Random Forest model...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight='balanced'
    )
    rf_model.fit(X_train_scaled, y_train)
    
    test_accuracy = rf_model.score(X_test_scaled, y_test)
    print(f"Testing Accuracy: {test_accuracy:.4f}")
    
    # Save models with EXACT names your Streamlit app expects
    joblib.dump(rf_model, 'fake_user_rf_model.pkl')
    joblib.dump(scaler, 'fake_user_scaler.pkl')
    joblib.dump(X.columns.tolist(), 'feature_names.pkl')
    
    print("\nModels saved as:")
    print("  - fake_user_rf_model.pkl")
    print("  - fake_user_scaler.pkl")
    print("  - feature_names.pkl")
    print("\n✅ Models are now compatible with your Streamlit app!")

if __name__ == "__main__":
    main()
