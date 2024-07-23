# Core packages
import streamlit as st
import os
import joblib

# EDA packages
import pandas as pd
import numpy as np

# Data visualization packages
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use('Agg')

@st.cache
def load_data(dataset):
    df = pd.read_csv(dataset)
    return df

# Load model
def load_prediction_model(model_file):
    loaded_model = joblib.load(open(os.path.join(model_file), "rb"))
    return loaded_model

def main():
    st.title("Flood Prediction ML App")
    # Menu
    menu = ["EDA", "Prediction", "Train and Save Models"]
    choices = st.sidebar.selectbox("Select Activities", menu)
    
    if choices == "EDA":
        st.subheader("EDA")
        data = load_data("floods.csv")
        st.dataframe(data.head(5))
        
        if st.checkbox("Show Summary Of Dataset"):
            st.write(data.describe())
        if st.checkbox("Show Shape"):
            st.write(data.shape)
        # Show plots
        if st.checkbox("Value Count Plot"):
            st.write(sns.countplot(data["FloodProbability"]))
            st.pyplot()
        # Show columns by selection
        if st.checkbox("Select Columns To Show"):
            all_columns = data.columns.tolist()
            selected_columns = st.multiselect("Select", all_columns)
            new_df = data[selected_columns]
            st.dataframe(new_df)
                
        if st.checkbox("Pie Chart"):
            all_columns_names = data.columns.tolist()
            if st.button("Generate Pie Plot"):
                st.write(data.iloc[:, -1].value_counts().plot.pie(autopct="%1.1f%%"))
                st.pyplot()                
                        
                    
    if choices == "Prediction":
        st.subheader("Prediction")
        MonsoonIntensity = st.number_input("Select Monsoon Intensity", 1, 20)
        TopographyDrainage = st.number_input("Select Topography Drainage", 1, 20)
        RiverManagement = st.number_input("Select River Management", 1, 20)
        Deforestation = st.number_input("Select Deforestation", 1, 20)
        Urbanization = st.number_input("Select Urbanization", 1, 20)
        ClimateChange = st.number_input("Select Climate Change", 1, 20)
        DamsQuality = st.number_input("Select Dams Quality", 1, 20)
        Siltation = st.number_input("Select Siltation", 1, 20)
        AgriculturalPractices = st.number_input("Select Agricultural Practices", 1, 20)
        Encroachments = st.number_input("Select Encroachments", 1, 20)
        IneffectiveDisasterPreparedness = st.number_input("Select Ineffective Disaster Preparedness", 1, 20)
        DrainageSystems = st.number_input("Select Drainage Systems", 1, 20)
        CoastalVulnerability = st.number_input("Select Coastal Vulnerability", 1, 20)
        Landslides = st.number_input("Select Landslides", 1, 20)
        Watersheds = st.number_input("Select Watersheds", 1, 20)
        DeterioratingInfrastructure = st.number_input("Select Deteriorating Infrastructure", 1, 20)
        PopulationScore = st.number_input("Select Population Score", 1, 20)
        WetlandLoss = st.number_input("Select Wetland Loss", 1, 20)
        InadequatePlanning = st.number_input("Select Inadequate Planning", 1, 20)
        PoliticalFactors = st.number_input("Select Political Factors", 1, 20)        
        
        pretty_data = {
            "MonsoonIntensity": MonsoonIntensity, "TopographyDrainage": TopographyDrainage, "RiverManagement": RiverManagement,
            "Deforestation": Deforestation, "Urbanization": Urbanization, "ClimateChange": ClimateChange, "DamsQuality": DamsQuality,
            "Siltation": Siltation, "AgriculturalPractices": AgriculturalPractices, "Encroachments": Encroachments,
            "IneffectiveDisasterPreparedness": IneffectiveDisasterPreparedness, "DrainageSystems": DrainageSystems,
            "CoastalVulnerability": CoastalVulnerability, "Landslides": Landslides, "Watersheds": Watersheds,
            "DeterioratingInfrastructure": DeterioratingInfrastructure, "PopulationScore": PopulationScore, "WetlandLoss": WetlandLoss,
            "InadequatePlanning": InadequatePlanning, "PoliticalFactors": PoliticalFactors
        }
        
        st.subheader("Options Selected")
        st.json(pretty_data)
        
        st.subheader("Data Encoded As")
        sample_data = [
            MonsoonIntensity, TopographyDrainage, RiverManagement, Deforestation, Urbanization, ClimateChange,
            DamsQuality, Siltation, AgriculturalPractices, Encroachments, IneffectiveDisasterPreparedness,
            DrainageSystems, CoastalVulnerability, Landslides, Watersheds, DeterioratingInfrastructure,
            PopulationScore, WetlandLoss, InadequatePlanning, PoliticalFactors
        ]
        st.write(sample_data)
        # Reshape sample data to 2D array
        prep_data = np.array(sample_data).reshape(1, -1)
        
        model_choice = st.selectbox("Model Choice", ["LinearRegression", "RandomForestRegressor", "SVR", "MLPRegressor", "ElasticNet"])
        
        if st.button("Evaluate"):
            if model_choice == "LinearRegression":
                predictor = load_prediction_model("lrmodel1.pkl")
                prediction = predictor.predict(prep_data)
            elif model_choice == "RandomForestRegressor":
                predictor = load_prediction_model("rfmodel1.pkl")
                prediction = predictor.predict(prep_data)
            elif model_choice == "SVR":
                predictor = load_prediction_model("svmodel1.pkl")
                prediction = predictor.predict(prep_data)
            elif model_choice == "MLPRegressor":
                predictor = load_prediction_model("mlpmodel1.pkl")
                prediction = predictor.predict(prep_data)
            elif model_choice == "ElasticNet":
                predictor = load_prediction_model("enmodel1.pkl")
                prediction = predictor.predict(prep_data)
            st.write("Prediction:", prediction)
                                        
    if choices == "About":
        st.subheader("About")

if __name__ == '__main__':
    main()
    