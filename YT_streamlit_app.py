# Creating Streamlit app showcasing:
            # Revenue prediction from user input
            # Basic visual analytics and model insights


#Import Libararies
import warnings
warnings.filterwarnings("ignore")
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

import os 


 # Force an unshakeable absolute path to the pifolder containing your files
TARGET_DIR = r"D:\Shruti\Guvi\Content Monetization Modeler\venv\Scripts"
# Combine the directory with your filenames using safe separators
model_path = os.path.join(TARGET_DIR, "ridge_model.pkl")
scaler_path = os.path.join(TARGET_DIR, "scaler.pkl")

# Creating Model object

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# Creating User Interface


#------------------------------- Adding Gradient Background--------------------------------

gradient_css = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: linear-gradient(90deg,rgba(182, 175, 189, 1) 0%, rgba(232, 165, 165, 1) 50%);
    background-size: cover;
}
</style>
"""

st.markdown(gradient_css, unsafe_allow_html=True)

sidebar_gradient_css = """
<style>
[data-testid="stSidebar"] {
    background-image: linear-gradient(90deg,rgba(182, 175, 189, 1) 0%, rgba(232, 165, 165, 1) 50%);

    background-size: cover;
}
"""
st.markdown(sidebar_gradient_css, unsafe_allow_html=True)



#----------------------------------- Streamlit Layout-----------------------------------------

# Streamlit App Title
st.set_page_config(page_title="Content Monetization Modeler", layout="wide")

st.sidebar.image("D:/Shruti/Guvi/Content Monetization Modeler/Data/yt_content_monetization.jpg")
st.sidebar.title("Navigation : ")
page = st.sidebar.radio("Go To", ["Project Introduction", "Prediction & Visualization",
                                   "Creator Information"])


#----------------------------------- Project Introduction Page-----------------------------------------

if page == "Project Introduction":
    
    st.image("D:/Shruti/Guvi/Content Monetization Modeler/Data/YouTube-Channel_BlogThumbnail.png")
    st.title("Project Introduction")

    st.subheader("Welcome to the YouTube Content Monetization Modeler")
    st.write("""As video creators and media companies increasingly rely on platforms like YouTube for income,
              predicting potential ad revenue becomes essential for business planning and content strategy.
             On this Platform Linear Regression model is used to accurately estimate YouTube ad revenue for
              individual videos based on various performance and contextual features.""")

    st.divider()
    st.subheader("Business Use Cases:")
    st.write("**Content Strategy Optimization:** Helps creators determine what type of content yields the highest returns.")
    st.write("**Revenue Forecasting:** Media companies can predict expected income from future video uploads.")
    st.write("**Creator Support Tools:** Can be integrated into platforms offering analytics services to YouTubers.")
    st.write("**Ad Campaign Planning:** Enables advertisers to forecast ROI based on content performance metrics.")


#----------------------------------- Predication Page-----------------------------------------

elif page == "Prediction & Visualization":
    st.title("Prediction & Visualization")

    
   


    # ==========================================
# SIDEBAR / INPUT FORM FIELDS
# ==========================================
    
    st.subheader("1. Timing & Context")
    # Raw date and time selectors
    col1,col2 = st.columns(2)
    with col1:
        upload_date = st.date_input("Upload Date")
    with col2:
        upload_time = st.time_input("Upload Time")



    col1, col2, col3 = st.columns(3)
    with col1:
        
        # ==========================================
        # 1. CATEGORY FIELD MAPPING
        # ==========================================
        category_mapping = {
            "Education": "0",
            "Entertainment": "1",  # Maps correctly to '1' despite data typo
            "Gaming": "2",
            "Lifestyle": "3",
            "Music": "4",
            "Tech": "5",
        }
        selected_cat = st.selectbox("Content Category", options=list(category_mapping.keys()))
        category = category_mapping[selected_cat]
       
    with col2:
        
        # ==========================================
        # 2. DEVICE FIELD MAPPING
        # ==========================================
        device_mapping = {
            "Desktop": "0",
            "Mobile": "1",
            "TV": "2",
            "Tablet": "3",
        }
        selected_dev = st.selectbox("Primary Device Type", options=list(device_mapping.keys()))
        device = device_mapping[selected_dev]
        
    with col3:
        
        # ==========================================
        # 3. COUNTRY FIELD MAPPING
        # ==========================================
        country_mapping = {
            "Australia (AU)": "0",
            "Canada (CA)": "1",
            "Germany (DE)": "2",
            "India (IN)": "3",
            "United Kingdom (UK)": "4",
            "United States (US)": "5",
        }
        selected_geo = st.selectbox("Target Country", options=list(country_mapping.keys()))
        country = country_mapping[selected_geo]
        
    st.subheader("2. Video Performance Metrics")
    col4, col5,col6,col7 = st.columns(4)
    with col4:
        views = st.number_input("Expected Views", min_value=0, value=10000)
    with col5:
        likes = st.number_input("Expected Likes", min_value=0.0, value=1000.0)
    with col6:
        subscribers = st.number_input("Channel Subscribers", min_value=0, value=250000)
    with col7:
        comments = st.number_input("Expected Comments", min_value=0.0, value=200.0)

    video_length_minutes = st.slider("Video Length (Minutes)", min_value=0.1, max_value=60.0, value=10.0)
    

#-------------------------------------Input Data Processing ----------------------
    

    try:
        Date_Day = upload_date.day
        Date_Month  = upload_date.month
        Date_Year = upload_date.year

        Time_Hour = upload_time.hour
        Time_Minutes = upload_time.minute
        Time_Seconds = upload_time.second

        watch_time_minutes = views * ( video_length_minutes * 0.45)
    
        # 1. Engagement Rate: (likes + comments) / views
        engagement_rate = (likes + comments)/(views)

        # 2. Average View Duration: watch_time_minutes / views
        avg_view_duration_min = watch_time_minutes/(views)

        # 3. Like to Comment Ratio: likes / comments
        like_to_comment_ratio = likes/(comments)

        # 4. Reach Index: views / subscribers
        view_to_sub_ratio = views/(subscribers)


        # Creating Dataframe

        X = pd.DataFrame([[float(views),float(likes),float(comments),float(watch_time_minutes),float(video_length_minutes),float(subscribers),float(category),float(device),float(country),float(Date_Day),float(Date_Month),float(Date_Year),float(Time_Hour),float(Time_Minutes),float(Time_Seconds),float(engagement_rate),float(avg_view_duration_min),float(like_to_comment_ratio),float(view_to_sub_ratio)]],columns=['views','likes','comments','watch_time_minutes','video_length_minutes','subscribers','category','device','country','Date_Day','Date_Month','Date_Year','Time_Hour','Time_Minutes','Time_Seconds','engagement_rate','avg_view_duration_min','like_to_comment_ratio','view_to_sub_ratio'])
        #st.dataframe(X,hide_index=True)

        # Scaling 

        X = scaler.transform(X)

        #st.dataframe(X,hide_index=True)
       
        # 6. Generate Prediction
        predicted_array = model.predict(X)
        
        predicted_revenue = max(0.0, float(predicted_array[0]))

        st.metric(label="Estimated Ad Revenue", value=f"${predicted_revenue:,.2f}")

        col6,col7,col8,col9 = st.columns(4)
        with col6:
            st.metric(label="Engagement Rate: ", value=f"{engagement_rate:,.2f}")
        with col7:
            st.metric(label="Average View Duration: ", value=f"{avg_view_duration_min:,.2f}")
        with col8:
            st.metric(label="Like to Comment Ratio: ", value=f"{like_to_comment_ratio:,.2f}")
        with col9:
            st.metric(label="Reach Index: ", value=f"{view_to_sub_ratio:,.2f}")


    except Exception as e:
        st.error(f"Processing error: {e}")
    
   
    # 8. COMPACT MODEL DRIVERS CHART (SEABORN VISUALIZATION)
    st.write("---")
    st.header("Visualization")

    col10,col11 = st.columns(2)
    with col10:
        st.subheader("Engagement Breakdown")
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.pie(
            [likes, comments], 
            labels=["Likes", "Comments"], 
            colors=["#1f77b4", "#aec7e8"], 
            autopct="%1.1f%%", 
            startangle=90
        )
        ax.set_title("Engagement Distribution")
        st.pyplot(fig)
        plt.close()

    with col11:
        st.subheader("Subscriber vs. Target Views")
        reach_data = pd.DataFrame({
        "Metric": ["Total Subscribers", "Expected Views"],
        "Count": [subscribers, views]
            })

        fig, ax = plt.subplots(figsize=(4, 4))
        sn.barplot(data=reach_data, y="Count", x="Metric", ax=ax, palette="Blues_r")
        ax.set_title("Audience Penetration Scale")
        ax.set_xlabel("") # Removes redundant label clutter
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
#----------------------------------- Creator InformATION Page-----------------------------------------

elif page == "Creator Information":
    st.title("Creator Information")
    st.subheader("Name : Shruti Kinalekar")
    st.write("""Software Engineer with total 8 years of experience in Software Testing.
             \nCurrently working on Data Science Projects.
             \nTools : Python, MySQl, Streamlit,vc++,Oracle""")