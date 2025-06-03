import streamlit as st
from pathlib import Path
import pandas as pd
import plotly.express as px
import sys
from pathlib import Path
import leafmap.foliumap as leafmap
import pandas as pd
from pages.Species_Composition import app as species_composition_app
from pages.Model_Development import app as model_development_app
from pages.Analysis import app as analysis_app
from pages.Observations import app as observations_app
from pages.Image_viewer import app as image_viewer_app
from pages.Bulk_Labeling import app as bulk_labeling_app

st.set_page_config(
    page_title="Bureau of Ocean Energy Management - Gulf of Mexico Biodiversity Survey",
    page_icon="🦅",
    layout="wide"
)

# Original home page content
def create_label_count_plots(label_counts_df):
    """Create plots showing label distributions over time"""
    if label_counts_df is None:
        return None, None

    # Create histogram of most recent model's counts
    counts_df = label_counts_df.groupby(['set', 'cropmodel_label'
                                            ]).size().reset_index(name='count')
    counts_df.columns = ['set', 'label', 'count']

    # remove FalsePositive and '0' from the label column
    counts_df = counts_df[counts_df['label'] != 'FalsePositive']
    counts_df = counts_df[counts_df['label'] != '0']

    fig_hist = px.bar(counts_df,
                        x='label',
                        y='count',
                        color='set',
                        title=f'Label Distribution in Latest Prediction',
                        labels={
                            'label': 'Label Type',
                            'count': 'Number of Instances',
                            'set': 'Dataset'
                        },
                        barmode='group')

    return fig_hist

# Read the data
data_path = Path(__file__).parent / "data" / "most_recent_all_flight_predictions.csv"

if not data_path.exists():
    st.error(f"File not found: {data_path}")
    st.stop()

# Add the app directory to Python path
app_dir = Path(__file__).parent
if str(app_dir) not in sys.path:
    sys.path.append(str(app_dir))

df = pd.read_csv(data_path)
df["timestamp"] = pd.to_datetime(df["timestamp"])

st.title("Bureau of Ocean Energy Management - Gulf of Mexico Biodiversity Survey")
st.text("This application provides tools for visualizing and analyzing biodiversity data collected during aerial surveys of offshore energy development areas. The tool uses AI to detect and classify marine wildlife species in aerial images. These data are used to inform the development of offshore projects using rapid and cost-effective surveys.")
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
        ### Overview
        
        This project processes aerial survey data to:
        - Detect and classify marine wildlife species
        - Generate distribution maps and abundance estimates
        - Analyze temporal and spatial patterns
        - Support environmental impact assessments

        The data viewer includes:
        - Interactive maps for viewing survey tracks and observations
        - Analysis tools for exploring species distributions
        - Image galleries of detected species
    """)

with col2:
    # Display basic statistics
    st.header("Statistics")
    st.write(f"Current Flight: {df['flight_name'].unique()[-1]}")
    st.write(f"Total Observations: {len(df)}")
    st.write(f"Number of Species: {df['cropmodel_label'].nunique()}")
    st.write(f"Total Flights: {df['flight_name'].nunique()}")
    st.write(f"Date Range: {df['timestamp'].min().strftime('%Y-%m-%d')} to {df['timestamp'].max().strftime('%Y-%m-%d')}")

# Create and display label distribution plots
st.subheader("Predicted Species")
hist_plot = create_label_count_plots(df)
st.plotly_chart(hist_plot, use_container_width=True)

# Rare species plot
species_counts = df['cropmodel_label'].value_counts()
if not species_counts.empty:
    max_count = species_counts.iloc[0]
    rare_threshold = max_count * 0.10
    rare_species = species_counts[species_counts < rare_threshold]
    if not rare_species.empty:
        rare_df = df[df['cropmodel_label'].isin(rare_species.index)]
        rare_counts = rare_df['cropmodel_label'].value_counts().reset_index()
        rare_counts.columns = ['label', 'count']
        rare_fig = px.bar(
            rare_counts,
            x='label',
            y='count',
            title='Predicted Rare Species (<10% of Most Common)',
            labels={'label': 'Species', 'count': 'Number of Instances'}
        )
        st.plotly_chart(rare_fig, use_container_width=True)
    else:
        st.info('No rare species (less than 10% of the most common) found in this dataset.')
else:
    st.info('No species data available.')

# Place the data below the plot
st.write(df.groupby('cropmodel_label').size().sort_values(ascending=False).reset_index(name='count'))