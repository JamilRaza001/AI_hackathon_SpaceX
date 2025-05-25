import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import requests
import numpy as np
from datetime import datetime
import os

# --- Data Fetching and Preparation Functions (Copied from your original script) ---
def fetch_spacex_launch_data():
    url = "https://api.spacexdata.com/v4/launches"
    response = requests.get(url)
    response.raise_for_status()
    launches = response.json()
    return launches

def process_launch_data(launches):
    data = []
    for launch in launches:
        item = {
            'launch_id': launch.get('id'),
            'name': launch.get('name'),
            'date_utc': launch.get('date_utc'),
            'success': launch.get('success'),
            'rocket_id': launch.get('rocket'),
            'launchpad_id': launch.get('launchpad'),
            'payloads': launch.get('payloads'),
            'cores': launch.get('cores'),
            'details': launch.get('details'),
        }
        data.append(item)
    df = pd.DataFrame(data)
    return df

def fetch_rockets():
    url = "https://api.spacexdata.com/v4/rockets"
    response = requests.get(url)
    response.raise_for_status()
    rockets = response.json()
    return {r['id']: r['name'] for r in rockets}

def fetch_launchpads():
    url = "https://api.spacexdata.com/v4/launchpads"
    response = requests.get(url)
    response.raise_for_status()
    pads = response.json()
    return {p['id']: {'name': p['name'], 'region': p['region'],
                      'latitude': p['latitude'], 'longitude': p['longitude']} for p in pads}

def fetch_weather_data(date_str, location):
    np.random.seed(hash(date_str) % 2**32)
    return {
        'temperature_C': round(15 + 10*np.random.rand(), 2),
        'wind_speed_mps': round(5 + 5*np.random.rand(), 2),
        'weather_condition': np.random.choice(['Clear', 'Cloudy', 'Rain', 'Storm', 'Snow'])
    }

def prepare_dataset():
    launches = fetch_spacex_launch_data()
    df_launches = process_launch_data(launches)

    rockets = fetch_rockets()
    launchpads = fetch_launchpads()

    df_launches['date_utc'] = pd.to_datetime(df_launches['date_utc'])
    df_launches['rocket_name'] = df_launches['rocket_id'].map(rockets)
    
    df_launches['launchpad_name'] = df_launches['launchpad_id'].map(lambda x: launchpads.get(x, {}).get('name'))
    df_launches['launchpad_region'] = df_launches['launchpad_id'].map(lambda x: launchpads.get(x, {}).get('region'))
    df_launches['launchpad_latitude'] = df_launches['launchpad_id'].map(lambda x: launchpads.get(x, {}).get('latitude'))
    df_launches['launchpad_longitude'] = df_launches['launchpad_id'].map(lambda x: launchpads.get(x, {}).get('longitude'))
    df_launches.dropna(subset=['launchpad_latitude', 'launchpad_longitude'], inplace=True)

    weather_data = df_launches.apply(
        lambda row: fetch_weather_data(row['date_utc'].strftime('%Y-%m-%dT%H:%M:%SZ'),
                                       {'lat': row['launchpad_latitude'], 'lon': row['launchpad_longitude']}), axis=1)
    weather_df = pd.DataFrame(list(weather_data))
    df_full = pd.concat([df_launches.reset_index(drop=True), weather_df], axis=1)

    df_full['success'] = df_full['success'].fillna(False)
    df_full['temperature_C'] = df_full['temperature_C'].fillna(df_full['temperature_C'].mean())
    df_full['wind_speed_mps'] = df_full['wind_speed_mps'].fillna(df_full['wind_speed_mps'].mean())
    df_full['weather_condition'] = df_full['weather_condition'].fillna('Unknown')

    le_weather = LabelEncoder()
    df_full['weather_condition_encoded'] = le_weather.fit_transform(df_full['weather_condition'])

    le_rocket = LabelEncoder()
    df_full['rocket_name_encoded'] = le_rocket.fit_transform(df_full['rocket_name'].fillna('Unknown'))

    le_launchpad = LabelEncoder()
    df_full['launchpad_name_encoded'] = le_launchpad.fit_transform(df_full['launchpad_name'].fillna('Unknown'))

    joblib.dump({
        'weather': le_weather,
        'rocket': le_rocket,
        'launchpad': le_launchpad
    }, 'encoders.pkl')

    df_full['year'] = df_full['date_utc'].dt.year
    df_full['month'] = df_full['date_utc'].dt.month
    df_full['day'] = df_full['date_utc'].dt.day

    return df_full[[
        'launch_id', 'name', 'date_utc', 'success', 'rocket_name', 'rocket_name_encoded',
        'launchpad_name', 'launchpad_name_encoded', 'launchpad_region', 'launchpad_latitude', 'launchpad_longitude',
        'temperature_C', 'wind_speed_mps', 'weather_condition', 'weather_condition_encoded',
        'year', 'month', 'day'
    ]], {
        'weather': le_weather,
        'rocket': le_rocket,
        'launchpad': le_launchpad
    }


# --- Model Training and Saving ---
def train_and_save_model():
    print("Preparing dataset...")
    df_prepared, encoders = prepare_dataset()
    df_prepared.to_csv("spacex_launch_data_preprocessed.csv", index=False)
    print("Dataset prepared and saved to spacex_launch_data_preprocessed.csv")
    print("Encoders saved to encoders.pkl")

    # Features and target variable
    feature_cols = [
        'rocket_name_encoded', 'launchpad_name_encoded', 'temperature_C',
        'wind_speed_mps', 'weather_condition_encoded', 'year', 'month', 'day'
    ]

    X = df_prepared[feature_cols]
    y = df_prepared['success'].astype(int)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Initialize Random Forest Classifier
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train model
    print("Training Random Forest model...")
    rf_clf.fit(X_train, y_train)
    print("Model training complete.")

    # Save the trained model
    joblib.dump(rf_clf, "rf_model.pkl")
    print("Model saved to rf_model.pkl")

if __name__ == "__main__":
    train_and_save_model()
# Visualization code remains identical...
import streamlit as st
import os
import joblib
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

# Add folium import here
import folium
from streamlit_folium import st_folium # Also ensure this is here if using st_folium

# Constants for model and encoder paths
MODEL_PATH = "rf_model.pkl"
ENCODER_PATH = "encoders.pkl"
DATA_PATH = "spacex_launch_data_preprocessed.csv"

# ... rest of your code ...

# 1. Fetch SpaceX launch data from SpaceX API (kept for completeness, but moved to train_model.py for actual data prep)
@st.cache_data
def fetch_spacex_launch_data_cached():
    url = "https://api.spacexdata.com/v4/launches"
    response = requests.get(url)
    response.raise_for_status()
    launches = response.json()
    return launches

# (Other data processing and fetching functions are now primarily used by train_model.py)

# 5. Main data preparation pipeline (if you want to regenerate data within Streamlit, though not recommended for production)
# This function is mostly for the `train_model.py` script.
# @st.cache_data
# def prepare_dataset_for_streamlit():
#     # ... (logic for fetching, processing, and encoding as in train_model.py)
#     # This would be used if you wanted Streamlit to *also* generate the data/encoders
#     # but it's better to do it once offline.
#     pass


# Load preprocessed data (adjust path if needed)
@st.cache_data
def load_preprocessed_data():
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH, parse_dates=['date_utc'])
        return df
    else:
        st.error(f"Preprocessed data file '{DATA_PATH}' not found. Please run 'train_model.py' first.")
        return pd.DataFrame()

df = load_preprocessed_data()


# Load model and encoders with caching
@st.cache_resource
def load_ml_assets():
    model, encoders = None, None
    if os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            encoders = joblib.load(ENCODER_PATH)
            st.success("Machine Learning model and encoders loaded successfully!")
        except Exception as e:
            st.error(f"Error loading model or encoders: {e}. Please ensure 'train_model.py' was run correctly.")
    else:
        st.warning("Model or encoders not found! Please run 'train_model.py' first to train and save them.")
    return model, encoders

# Load model and encoders once at the start of the script
model, encoders = load_ml_assets()


st.title("🚀 SpaceX Launch Analysis & Prediction Dashboard")

# Sidebar filters
st.sidebar.header("Filter Launch Data")
if not df.empty:
    years = st.sidebar.multiselect("Select Year(s)", options=sorted(df['year'].unique()), default=sorted(df['year'].unique()))
    sites = st.sidebar.multiselect("Select Launch Site(s)", options=sorted(df['launchpad_name'].unique()), default=sorted(df['launchpad_name'].unique()))

    filtered_df = df[(df['year'].isin(years)) & (df['launchpad_name'].isin(sites))]

    st.subheader(f"Filtered Launch Data ({len(filtered_df)} launches)")
    st.dataframe(filtered_df[['date_utc', 'name', 'rocket_name', 'launchpad_name', 'success', 'temperature_C', 'wind_speed_mps', 'weather_condition']])

    # Map
    st.subheader("Launch Sites Map")
    if not filtered_df.empty:
        avg_lat = filtered_df['launchpad_latitude'].mean()
        avg_lon = filtered_df['launchpad_longitude'].mean()
        m = folium.Map(location=[avg_lat, avg_lon], zoom_start=4)

        site_groups = filtered_df.groupby('launchpad_name')
        for site, group in site_groups:
            lat = group['launchpad_latitude'].iloc[0]
            lon = group['launchpad_longitude'].iloc[0]
            success_rate = group['success'].mean()
            color = 'green' if success_rate > 0.7 else 'orange' if success_rate > 0.4 else 'red'
            popup_text = f"{site}<br>Success Rate: {success_rate:.1%}<br>Total Launches: {len(group)}"
            folium.CircleMarker(location=[lat, lon], radius=10, color=color, fill=True, fill_color=color, popup=popup_text).add_to(m)

        st_data = st_folium(m, width=700, height=450)
    else:
        st.warning("No data to display on the map for the selected filters.")

    # Analytics Visualizations Section
    st.subheader("📊 Analytics Visualizations")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Rocket Performance", "Launch Site Analysis", "Weather Impact",
        "Environmental Factors", "Correlations", "Model Insights"
    ])

    with tab1:
        st.write("### Launch Success Count by Rocket Type")
        if not filtered_df.empty:
            fig, ax = plt.subplots(figsize=(10,6))
            sns.countplot(
                data=filtered_df,
                x='rocket_name',
                hue=filtered_df['success'].map({True: 'Success', False: 'Failure'}),
                ax=ax
            )
            ax.set_title('Launch Outcomes by Rocket Type')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
            ax.set_ylabel('Number of Launches')
            ax.legend(title='Outcome')
            st.pyplot(fig)
        else:
            st.warning("No data available for selected filters.")

    with tab2:
        st.write("### Launch Success Rate by Site")
        if not filtered_df.empty:
            fig, ax = plt.subplots(figsize=(10,6))
            success_by_site = filtered_df.groupby(['launchpad_name', 'success']).size().unstack().fillna(0)
            success_rate = success_by_site[True] / (success_by_site[True] + success_by_site[False])
            success_rate.sort_values(ascending=False).plot(kind='bar', ax=ax, color='skyblue')
            ax.set_title('Success Rate by Launch Site')
            ax.set_ylabel('Success Rate')
            ax.set_ylim(0, 1)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
            st.pyplot(fig)
        else:
            st.warning("No data available for selected filters.")

    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            st.write("#### Success Rate by Weather")
            if not filtered_df.empty:
                fig, ax = plt.subplots(figsize=(8,5))
                weather_success = filtered_df.groupby('weather_condition')['success'].mean().sort_values(ascending=False)
                weather_success.plot(kind='bar', ax=ax, color='coral')
                ax.set_title('Success Rate by Weather Condition')
                ax.set_ylabel('Success Rate')
                ax.set_ylim(0, 1)
                st.pyplot(fig)
            else:
                st.warning("No data available.")

        with col2:
            st.write("#### Weather Condition Distribution")
            if not filtered_df.empty:
                fig, ax = plt.subplots(figsize=(8,5))
                filtered_df['weather_condition'].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax)
                ax.set_title('Weather Condition Distribution')
                ax.set_ylabel('')
                st.pyplot(fig)
            else:
                st.warning("No data available.")

    with tab4:
        col1, col2 = st.columns(2)
        with col1:
            st.write("#### Temperature Distribution")
            if not filtered_df.empty:
                fig, ax = plt.subplots(figsize=(8,5))
                sns.boxplot(x='success', y='temperature_C', data=filtered_df, ax=ax)
                ax.set_title('Temperature vs. Launch Success')
                ax.set_xlabel('Success')
                ax.set_ylabel('Temperature (°C)')
                st.pyplot(fig)
            else:
                st.warning("No data available.")

        with col2:
            st.write("#### Wind Speed Distribution")
            if not filtered_df.empty:
                fig, ax = plt.subplots(figsize=(8,5))
                sns.boxplot(x='success', y='wind_speed_mps', data=filtered_df, ax=ax)
                ax.set_title('Wind Speed vs. Launch Success')
                ax.set_xlabel('Success')
                ax.set_ylabel('Wind Speed (m/s)')
                st.pyplot(fig)
            else:
                st.warning("No data available.")

    with tab5:
        st.write("### Feature Correlation Heatmap")
        if not filtered_df.empty:
            numeric_cols = [
                'success', 'rocket_name_encoded', 'launchpad_name_encoded',
                'temperature_C', 'wind_speed_mps', 'weather_condition_encoded',
                'year', 'month', 'day'
            ]
            fig, ax = plt.subplots(figsize=(10,8))
            sns.heatmap(
                filtered_df[numeric_cols].corr(),
                annot=True, cmap='coolwarm', fmt='.2f', ax=ax
            )
            ax.set_title('Feature Correlation Matrix')
            st.pyplot(fig)
        else:
            st.warning("No data available for correlations.")

    with tab6:
        if model is not None:
            st.write("### Model Feature Importances")
            feature_cols = [
                'rocket_name_encoded', 'launchpad_name_encoded', 'temperature_C',
                'wind_speed_mps', 'weather_condition_encoded', 'year', 'month', 'day'
            ]
            # Ensure feature_cols are in the same order as trained model
            if hasattr(model, 'feature_names_in_'): # scikit-learn >= 1.0
                model_feature_names = model.feature_names_in_
            else: # Fallback for older scikit-learn versions or if not set
                 model_feature_names = feature_cols # Assuming order is consistent

            importances = pd.Series(model.feature_importances_, index=model_feature_names).sort_values(ascending=False)
            fig, ax = plt.subplots(figsize=(8,6))
            sns.barplot(x=importances, y=importances.index, palette="viridis", ax=ax)
            ax.set_title('Feature Importances in Prediction Model')
            ax.set_xlabel('Importance Score')
            st.pyplot(fig)
        else:
            st.warning("Model not loaded - feature importances unavailable.")


    # Predictive tool
    st.subheader("🚀 Predict Launch Success")

    rocket_options = df['rocket_name'].unique()
    launchpad_options = df['launchpad_name'].unique()
    weather_options = df['weather_condition'].unique()

    rocket = st.selectbox("Rocket Type", rocket_options)
    launchpad = st.selectbox("Launch Site", launchpad_options)
    temperature = st.number_input("Temperature (°C)", float(df['temperature_C'].min()), float(df['temperature_C'].max()), float(df['temperature_C'].mean()))
    wind_speed = st.number_input("Wind Speed (m/s)", float(df['wind_speed_mps'].min()), float(df['wind_speed_mps'].max()), float(df['wind_speed_mps'].mean()))
    weather = st.selectbox("Weather Condition", weather_options)
    year = st.number_input("Year", int(df['year'].min()), int(df['year'].max()), int(df['year'].max()))
    month = st.number_input("Month", 1, 12, 1)
    day = st.number_input("Day", 1, 31, 1)


    if model is None or encoders is None:
        st.warning("Prediction model or encoders are not loaded. Please run 'train_model.py' first.")
    else:
        try:
            # Encode categorical inputs
            rocket_encoded = encoders['rocket'].transform([rocket])[0]
            launchpad_encoded = encoders['launchpad'].transform([launchpad])[0]
            weather_encoded = encoders['weather'].transform([weather])[0]

            input_df = pd.DataFrame({
                'rocket_name_encoded': [rocket_encoded],
                'launchpad_name_encoded': [launchpad_encoded],
                'temperature_C': [temperature],
                'wind_speed_mps': [wind_speed],
                'weather_condition_encoded': [weather_encoded],
                'year': [year],
                'month': [month],
                'day': [day]
            })

            if st.button("Predict Launch Success Probability"):
                prob = model.predict_proba(input_df)[0][1]
                st.write(f"🚀 Predicted Probability of Launch Success: **{prob:.2%}**")

        except Exception as e:
            st.error(f"An error occurred during prediction. Please check your inputs and ensure the model is correctly loaded: {e}")

else:
    st.error("Cannot display dashboard. Preprocessed data not found. Please run 'train_model.py' first.")