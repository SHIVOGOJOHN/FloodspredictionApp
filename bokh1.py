import streamlit as st
import pandas as pd
import pandas_bokeh
import warnings

from bokeh.plotting import figure
from bokeh.tile_providers import get_provider, Vendors

# Create a figure with mercator projection
p = figure(x_range=(-2000000, 6000000), y_range=(-1000000, 7000000),
           x_axis_type="mercator", y_axis_type="mercator")

# Add the CARTODBPOSITRON tile
tile_provider = get_provider(Vendors.CARTODBPOSITRON)
p.add_tile(tile_provider)

# Set Bokeh output to display within Streamlit
pandas_bokeh.output_notebook()

# Load the flood data
@st.cache_data
def load_data():
    df = pd.read_csv("floods.csv")
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
probability_chart = filtered_df['FloodProbability'].plot_bokeh(kind='hist', bins=10, title="Flood Probability Distribution", xlabel="Flood Probability", ylabel="Frequency")
st.bokeh_chart(probability_chart, use_container_width=True)

# Factors affecting Flood Probability
st.subheader("Factors Affecting Flood Probability")

# Monsoon Intensity vs Flood Probability
st.markdown("### Monsoon Intensity vs Flood Probability")
monsoon_chart = filtered_df.plot_bokeh(x='MonsoonIntensity', y='FloodProbability', kind='scatter', title="Monsoon Intensity vs Flood Probability", xlabel="Monsoon Intensity", ylabel="Flood Probability")
st.bokeh_chart(monsoon_chart, use_container_width=True)

# Urbanization vs Flood Probability
st.markdown("### Urbanization vs Flood Probability")
urbanization_chart = filtered_df.plot_bokeh(x='Urbanization', y='FloodProbability', kind='scatter', title="Urbanization vs Flood Probability", xlabel="Urbanization", ylabel="Flood Probability")
st.bokeh_chart(urbanization_chart, use_container_width=True)

# Population Score vs Flood Probability
st.markdown("### Population Score vs Flood Probability")
population_chart = filtered_df.plot_bokeh(x='PopulationScore', y='FloodProbability', kind='scatter', title="Population Score vs Flood Probability", xlabel="Population Score", ylabel="Flood Probability")
st.bokeh_chart(population_chart, use_container_width=True)

# Top Contributors to Flood Probability
st.subheader("Top Contributors to Flood Probability")
top_contributors = filtered_df.corr()['FloodProbability'].sort_values(ascending=False).head(10)
top_contributors_chart = top_contributors.plot_bokeh(kind='barh', title="Top 10 Contributors to Flood Probability", xlabel="Correlation", ylabel="Factors")
st.bokeh_chart(top_contributors_chart, use_container_width=True)

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
    <p>Developed with ❤️ by [Your Name]</p>
    </div>
""", unsafe_allow_html=True)
