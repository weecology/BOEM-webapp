import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
from utils.styling import load_css
from utils.annotations import load_annotations, apply_annotations
import geopandas as gpd
from utils.auth import require_login

def create_label_count_plots(label_counts_df):
    """Create plots showing label distributions over time"""
    if label_counts_df is None:
        return None, None
    counts_df = label_counts_df.groupby(['set', 'cropmodel_label']).size().reset_index(name='count')
    counts_df.columns = ['set', 'label', 'count']
    counts_df = counts_df[counts_df['label'] != 'FalsePositive']
    counts_df = counts_df[counts_df['label'] != '0']
    label_order = counts_df.groupby('label')['count'].sum().sort_values(ascending=False).index
    fig_hist = px.bar(counts_df,
                      x='label',
                      y='count',
                      color='set',
                      title='Label Distribution in Latest Prediction',
                      labels={
                          'label': 'Label Type',
                          'count': 'Number of Instances',
                          'set': 'Dataset'
                      },
                      category_orders={"label": label_order},
                      barmode='group')
    return fig_hist

def app():
    require_login()
    st.title("Species Analysis")
    app_data_dir = Path(__file__).parents[1] / "data"
    default_file = app_data_dir / "most_recent_all_flight_predictions.csv"
    df = pd.read_csv(default_file)
    
    # Apply annotation overrides to analysis dataset
    annotations_df = load_annotations("app/data/annotations.csv")
    df = apply_annotations(df, annotations_df, id_col="crop_image_id", label_col="cropmodel_label", set_col="set")
    df = df[df["cropmodel_label"].str.count(" ") == 1]
    date_component = df["flight_name"].str.split("_").str[1]
    datetime_values = pd.to_datetime(date_component, format='%Y%m%d', errors='coerce')
    df["timestamp"] = datetime_values.astype(str)
    gdf = gpd.read_file(default_file.parent / "all_predictions.shp")
    gdf['date'] = pd.to_datetime(gdf['date'], errors='coerce')

    st.subheader("Predicted Species")
    hist_plot = create_label_count_plots(df)
    st.plotly_chart(hist_plot, use_container_width=True)
    st.write(df.groupby('cropmodel_label').size().sort_values(ascending=False).reset_index(name='count'))

    # Rare species plot
    species_counts = df["cropmodel_label"].value_counts()
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
                title='Predicted Rare Species (<10% of Most Common - Human Reviewed)',
                labels={'label': 'Species', 'count': 'Number of Instances'}
            )
            st.plotly_chart(rare_fig, use_container_width=True)
        else:
            st.info('No rare species (less than 10% of the most common) found in this dataset.')
    else:
        st.info('No species data available.')

    st.subheader("Reviewed Observations")
    reviewed_species_counts = df[df['set'].isin(['train', 'validation', "review"] )]['cropmodel_label'].value_counts()
    if not reviewed_species_counts.empty:
        reviewed_counts = reviewed_species_counts.reset_index()
        reviewed_counts.columns = ['label', 'count']
        reviewed_fig = px.bar(
            reviewed_counts,
            x='label',
            y='count',
            labels={'label': 'Species', 'count': 'Number of Instances'}
        )
        st.plotly_chart(reviewed_fig, use_container_width=True)
        max_count = reviewed_species_counts.iloc[0]
        rare_threshold = max_count * 0.10
        rare_species = reviewed_species_counts[reviewed_species_counts < rare_threshold]
        if not rare_species.empty:
            rare_df = df[df['cropmodel_label'].isin(rare_species.index)]
            rare_counts = rare_df['cropmodel_label'].value_counts().reset_index()
            rare_counts.columns = ['label', 'count']
            rare_fig = px.bar(
                rare_counts,
                x='label',
                y='count',
                title='Predicted Rare Species (<10% of Most Common - Human Reviewed)',
                labels={'label': 'Species', 'count': 'Number of Instances'}
            )
            st.plotly_chart(rare_fig, use_container_width=True)
        else:
            st.info('No rare species (less than 10% of the most common) found in this dataset.')
    else:
        st.info('No species data available.')

if __name__ == "__main__":
    load_css()
    app() 