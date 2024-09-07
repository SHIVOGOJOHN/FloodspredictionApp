import streamlit as st
import pandas as pd
from bokeh.plotting import figure
from bokeh.io import show
from bokeh.models import ColumnDataSource
from bokeh.layouts import column
import numpy as np
from bokeh.palettes import Viridis256

# Load the flood data
@st.cache_data
def load_data():
    df = pd.read_csv("floods.csv")  # Ensure this path is correct or use an uploader
    return df

df = load_data()

# Sidebar for user inputs
st.sidebar.title("Flood Prediction Dashboard")
st.sidebar.markdown("Use the filters below to adjust the metrics:")

# Filter options
monsoon_intensity = st.sidebar.slider("Monsoon Intensity", min_value=int(df['MonsoonIntensity'].min()), max_value=int(df['MonsoonIntensity'].max()), value=int(df['MonsoonIntensity'].mean()))
urbanization = st.sidebar.slider("Urbanization Level", min_value=int(df['Urbanization'].min()), max_value=int(df['Urbanization'].max()), value=int(df['Urbanization'].mean()))
population_score = st.sidebar.slider("Population Score", min_value=int(df['PopulationScore'].min()), max_value=int(df['PopulationScore'].max()), value=int(df['PopulationScore'].mean()))

# Filtered DataFrame
filtered_df = df[(df['MonsoonIntensity'] >= monsoon_intensity) & (df['Urbanization'] >= urbanization) & (df['PopulationScore'] >= population_score)]

# Main dashboard
st.title("Floods Prediction Dashboard")
st.markdown("This dashboard helps in visualizing and predicting flood probabilities based on various environmental and human factors.")

# Flood Probability Distribution
st.subheader("Flood Probability Distribution")
hist, edges = np.histogram(filtered_df['FloodProbability'], bins=10)
p_hist = figure(title="Flood Probability Distribution", x_axis_label="Flood Probability", y_axis_label="Frequency")
p_hist.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], fill_color="navy", line_color="white", alpha=0.5)
st.bokeh_chart(p_hist, use_container_width=True)

# Factors affecting Flood Probability
st.subheader("Factors Affecting Flood Probability")

# Monsoon Intensity vs Flood Probability
st.markdown("### Monsoon Intensity vs Flood Probability")
p_monsoon = figure(title="Monsoon Intensity vs Flood Probability", x_axis_label="Monsoon Intensity", y_axis_label="Flood Probability")
p_monsoon.scatter(filtered_df['MonsoonIntensity'], filtered_df['FloodProbability'], size=8, color="green", alpha=0.5)
st.bokeh_chart(p_monsoon, use_container_width=True)

# Urbanization vs Flood Probability
st.markdown("### Urbanization vs Flood Probability")
p_urbanization = figure(title="Urbanization vs Flood Probability", x_axis_label="Urbanization", y_axis_label="Flood Probability")
p_urbanization.scatter(filtered_df['Urbanization'], filtered_df['FloodProbability'], size=8, color="blue", alpha=0.5)
st.bokeh_chart(p_urbanization, use_container_width=True)

# Population Score vs Flood Probability
st.markdown("### Population Score vs Flood Probability")
p_population = figure(title="Population Score vs Flood Probability", x_axis_label="Population Score", y_axis_label="Flood Probability")
p_population.scatter(filtered_df['PopulationScore'], filtered_df['FloodProbability'], size=8, color="red", alpha=0.5)
st.bokeh_chart(p_population, use_container_width=True)

# Top Contributors to Flood Probability
st.subheader("Top Contributors to Flood Probability")
top_contributors = filtered_df.corr()['FloodProbability'].sort_values(ascending=False).head(10)
top_contributors_df = top_contributors.reset_index()
top_contributors_df.columns = ['Factor', 'Correlation']
p_contributors = figure(y_range=top_contributors_df['Factor'].tolist(), height=300, title="Top 10 Contributors to Flood Probability", x_axis_label="Correlation", y_axis_label="Factors")
p_contributors.hbar(y=top_contributors_df['Factor'], right=top_contributors_df['Correlation'], height=0.4, color=Viridis256[10])
st.bokeh_chart(p_contributors, use_container_width=True)

# Summary Statistics
st.subheader("Summary Statistics")
st.write(filtered_df.describe())

# Closing remarks
st.markdown("This dashboard aims to support decision-making in flood risk management by analyzing and visualizing critical factors that contribute to flooding.")

# Footer
st.markdown("""
    <style>
    footer {
        visibility: hidden;
    }
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #f1f1f1;
        color: black;
        text-align: center;
        padding: 10px;
    }
    </style>
    <div class="footer">
    <p>Developed with ❤️ by [JOHN.S]</p>
    </div>
""", unsafe_allow_html=True)
