#SpaceX Launch Analysis & Prediction Platform with Weather Integration and ML
import streamlit as st
import os
import joblib
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import LabelEncoder

# 1. Fetch SpaceX launch data from SpaceX API
def fetch_spacex_launch_data():
    url = "https://api.spacexdata.com/v4/launches"
    response = requests.get(url)
    response.raise_for_status()
    launches = response.json()
    return launches

# 2. Convert launch data into DataFrame with selected features
def process_launch_data(launches):
    data = []
    for launch in launches:
        # extract relevant fields (you can expand this list)
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

# 3. Fetch rockets and launchpads info to enrich dataset (for names, locations)
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
    return {p['id']: {'name': p['name'], 'region': p['region'], 'latitude': p['latitude'], 'longitude': p['longitude']} for p in pads}

# 4. Simulated weather data fetch function (replace with real scraping/API)
def fetch_weather_data(date_str, location):
    """
    Simulates weather data fetching for given date and location.
    date_str: 'YYYY-MM-DDTHH:MM:SSZ' format
    location: dict with lat/lon
    """
    # For demo purposes, create dummy weather data
    # Real implementation would call a weather API with date and coordinates
    np.random.seed(hash(date_str) % 2**32)
    return {
        'temperature_C': round(15 + 10*np.random.rand(), 2),
        'wind_speed_mps': round(5 + 5*np.random.rand(), 2),
        'weather_condition': np.random.choice(['Clear', 'Cloudy', 'Rain', 'Storm', 'Snow'])
    }

# 5. Main data preparation pipeline
def prepare_dataset():
    launches = fetch_spacex_launch_data()
    df_launches = process_launch_data(launches)

    rockets = fetch_rockets()
    launchpads = fetch_launchpads()

    # Convert date string to datetime
    df_launches['date_utc'] = pd.to_datetime(df_launches['date_utc'])

    # Map rocket names
    df_launches['rocket_name'] = df_launches['rocket_id'].map(rockets)

    # Map launchpad info
    df_launches['launchpad_name'] = df_launches['launchpad_id'].map(lambda x: launchpads.get(x, {}).get('name'))
    df_launches['launchpad_region'] = df_launches['launchpad_id'].map(lambda x: launchpads.get(x, {}).get('region'))
    df_launches['launchpad_latitude'] = df_launches['launchpad_id'].map(lambda x: launchpads.get(x, {}).get('latitude'))
    df_launches['launchpad_longitude'] = df_launches['launchpad_id'].map(lambda x: launchpads.get(x, {}).get('longitude'))

    # Drop launches without launchpad info
    df_launches.dropna(subset=['launchpad_latitude', 'launchpad_longitude'], inplace=True)

    # Fetch weather data for each launch (simulate here)
    weather_data = df_launches.apply(
        lambda row: fetch_weather_data(row['date_utc'].strftime('%Y-%m-%dT%H:%M:%SZ'), 
                                       {'lat': row['launchpad_latitude'], 'lon': row['launchpad_longitude']}), axis=1)
    weather_df = pd.DataFrame(list(weather_data))

    df_full = pd.concat([df_launches.reset_index(drop=True), weather_df], axis=1)

    # Handle missing values
    # Success column may have nulls if unknown
    df_full['success'] = df_full['success'].fillna(False)  # assume failed if unknown
    df_full['temperature_C'] = df_full['temperature_C'].fillna(df_full['temperature_C'].mean())
    df_full['wind_speed_mps'] = df_full['wind_speed_mps'].fillna(df_full['wind_speed_mps'].mean())
    df_full['weather_condition'] = df_full['weather_condition'].fillna('Unknown')

    # Encode categorical variables
    le_weather = LabelEncoder()
    df_full['weather_condition_encoded'] = le_weather.fit_transform(df_full['weather_condition'])

    le_rocket = LabelEncoder()
    df_full['rocket_name_encoded'] = le_rocket.fit_transform(df_full['rocket_name'].fillna('Unknown'))

    le_launchpad = LabelEncoder()
    df_full['launchpad_name_encoded'] = le_launchpad.fit_transform(df_full['launchpad_name'].fillna('Unknown'))

    # Extract datetime features
    df_full['year'] = df_full['date_utc'].dt.year
    df_full['month'] = df_full['date_utc'].dt.month
    df_full['day'] = df_full['date_utc'].dt.day

    # Select useful columns for modeling or visualization
    df_final = df_full[[
        'launch_id', 'name', 'date_utc', 'success', 'rocket_name', 'rocket_name_encoded',
        'launchpad_name', 'launchpad_name_encoded', 'launchpad_region', 'launchpad_latitude', 'launchpad_longitude',
        'temperature_C', 'wind_speed_mps', 'weather_condition', 'weather_condition_encoded',
        'year', 'month', 'day'
    ]]

    return df_final

if __name__ == "__main__":
    df_prepared = prepare_dataset()
    print(df_prepared.head(10))
    # Save to CSV for further use
    df_prepared.to_csv("spacex_launch_data_preprocessed.csv", index=False)


import matplotlib.pyplot as plt
import seaborn as sns

# Load preprocessed data (adjust path if needed)
df = pd.read_csv("spacex_launch_data_preprocessed.csv", parse_dates=['date_utc'])

# 1. Overview of dataset
print(df.info())
print(df.describe())
print(df['success'].value_counts(normalize=True))

# Set plot style
sns.set(style="whitegrid")

# 2. Success rate by rocket type
plt.figure(figsize=(10,6))
sns.countplot(data=df, x='rocket_name', hue=df['success'].map({True: 'Success', False: 'Failure'}))  # Convert to string here
plt.title('Launch Success Count by Rocket Type')
plt.xticks(rotation=45)
plt.ylabel('Number of Launches')
plt.legend(title='Success')
plt.tight_layout()
plt.show()

# 3. Success rate by launch site
plt.figure(figsize=(10,6))
success_by_site = df.groupby(['launchpad_name', 'success']).size().unstack()
success_rate_by_site = success_by_site[True] / success_by_site.sum(axis=1)
success_rate_by_site = success_rate_by_site.sort_values(ascending=False)
success_rate_by_site.plot(kind='bar', figsize=(10,6), color='skyblue')
plt.title('Launch Success Rate by Launch Site')
plt.ylabel('Success Rate')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 4. Weather condition influence on success
plt.figure(figsize=(10,6))
weather_success = df.groupby(['weather_condition', 'success']).size().unstack()
weather_success_rate = weather_success[True] / weather_success.sum(axis=1)
weather_success_rate = weather_success_rate.sort_values(ascending=False)
weather_success_rate.plot(kind='bar', figsize=(10,6), color='coral')
plt.title('Launch Success Rate by Weather Condition')
plt.ylabel('Success Rate')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 5. Distribution of temperature by success
plt.figure(figsize=(10,6))
sns.boxplot(x='success', y='temperature_C', data=df)
plt.title('Temperature Distribution by Launch Success')
plt.ylabel('Temperature (°C)')
plt.xlabel('Success')
plt.tight_layout()
plt.show()

# 6. Distribution of wind speed by success
plt.figure(figsize=(10,6))
sns.boxplot(x='success', y='wind_speed_mps', data=df)
plt.title('Wind Speed Distribution by Launch Success')
plt.ylabel('Wind Speed (m/s)')
plt.xlabel('Success')
plt.tight_layout()
plt.show()

# 7. Correlation heatmap of numeric features
plt.figure(figsize=(10,8))
numeric_cols = ['success', 'rocket_name_encoded', 'launchpad_name_encoded', 'temperature_C', 'wind_speed_mps', 'weather_condition_encoded', 'year', 'month', 'day']
corr = df[numeric_cols].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.show()



import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns

# Load preprocessed dataset
df = pd.read_csv("spacex_launch_data_preprocessed.csv")

# Features and target variable
feature_cols = [
    'rocket_name_encoded', 'launchpad_name_encoded', 'temperature_C',
    'wind_speed_mps', 'weather_condition_encoded', 'year', 'month', 'day'
]

X = df[feature_cols]
y = df['success'].astype(int)  # convert bool to int (0/1)

# Train-test split (80% train, 20% test) with stratification on success
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Initialize Random Forest Classifier
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Cross-validation with StratifiedKFold (5 folds)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(rf_clf, X_train, y_train, cv=cv, scoring='accuracy')

print(f"Cross-validation Accuracy Scores: {cv_scores}")
print(f"Mean CV Accuracy: {cv_scores.mean():.4f}")

# Train model on full training set
rf_clf.fit(X_train, y_train)

# Predictions on test set
y_pred = rf_clf.predict(X_test)

# Evaluate performance
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print(f"\nTest Set Performance:")
print(f"Accuracy: {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall: {rec:.4f}")
print(f"F1-score: {f1:.4f}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Failure', 'Success'])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix - Test Set")
plt.show()

# Optional: Feature importance visualization
feature_importances = pd.Series(rf_clf.feature_importances_, index=feature_cols).sort_values(ascending=False)
plt.figure(figsize=(8,6))
sns.barplot(x=feature_importances, y=feature_importances.index, palette="viridis")
plt.title("Feature Importances in Random Forest Model")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()

MODEL_PATH = "rf_model.pkl"
ENCODER_PATH = "encoders.pkl"

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH):
        model = joblib.load(MODEL_PATH)
        encoders = joblib.load(ENCODER_PATH)
        return model, encoders
    return None, None

# Load model here before creating tabs
model, encoders = load_model()

# streamlit code
import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import joblib
import os

# Load data with caching
@st.cache_data
def load_data():
    df = pd.read_csv("spacex_launch_data_preprocessed.csv", parse_dates=['date_utc'])
    return df

df = load_data()

st.title("🚀 SpaceX Launch Analysis & Prediction Dashboard")

# Sidebar filters
st.sidebar.header("Filter Launch Data")
years = st.sidebar.multiselect("Select Year(s)", options=sorted(df['year'].unique()), default=sorted(df['year'].unique()))
sites = st.sidebar.multiselect("Select Launch Site(s)", options=sorted(df['launchpad_name'].unique()), default=sorted(df['launchpad_name'].unique()))

filtered_df = df[(df['year'].isin(years)) & (df['launchpad_name'].isin(sites))]

st.subheader(f"Filtered Launch Data ({len(filtered_df)} launches)")
st.dataframe(filtered_df[['date_utc', 'name', 'rocket_name', 'launchpad_name', 'success', 'temperature_C', 'wind_speed_mps', 'weather_condition']])

# Map
st.subheader("Launch Sites Map")
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


# Analytics Visualizations Section
st.subheader("📊 Analytics Visualizations")

# Use tabs to organize multiple graphs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Rocket Performance", "Launch Site Analysis", "Weather Impact",
    "Environmental Factors", "Correlations", "Model Insights"
])

with tab1:
    # Success Count by Rocket Type
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
    # Success Rate by Launch Site
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
    # Weather Impact Analysis
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
    # Environmental Factors
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
    # Correlation Matrix
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
    # Model Insights
    if model is not None:
        st.write("### Model Feature Importances")
        feature_cols = [
            'rocket_name_encoded', 'launchpad_name_encoded', 'temperature_C',
            'wind_speed_mps', 'weather_condition_encoded', 'year', 'month', 'day'
        ]
        fig, ax = plt.subplots(figsize=(8,6))
        importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
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

MODEL_PATH = "rf_model.pkl"
ENCODER_PATH = "encoders.pkl"

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH):
        model = joblib.load(MODEL_PATH)
        encoders = joblib.load(ENCODER_PATH)
        return model, encoders
    else:
        return None, None

model, encoders = load_model()

if model is None:
    st.warning("Model not found! Please train and save the model as 'rf_model.pkl' and encoders as 'encoders.pkl'.")
else:
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