import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
from utils.styling import load_css

def app():
    st.title("Species Analysis")
    # Get the app's data directory
    app_data_dir = Path(__file__).parents[1] / "data"
    
    # Always use the default data file
    default_file = app_data_dir / "most_recent_all_flight_predictions.csv"
    df = pd.read_csv(default_file)

    df["date"] = pd.to_datetime(df["timestamp"])
    df["count"] = 1
    
    # Add tabs for different visualizations
    tab1, tab2 = st.tabs([
        "Temporal Analysis",
        "Flight Analysis"
    ])
    
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    df["monthyear"] = df["date"].dt.strftime('%Y-%m')
    label_counts_df = df.groupby(['monthyear']).size().reset_index(name='count')

    # Create time series plot of label counts
    fig_timeseries = px.line(
        label_counts_df,
        x='monthyear',
        y='count',
        title='Label Counts Over Time',
        labels={'count': 'Number of Instances', 'monthyear': 'Month-Year'}
    )
    
    st.plotly_chart(fig_timeseries)

    with st.expander("View Raw Label Counts Data"):
        st.dataframe(label_counts_df)

    with tab1:
        st.subheader("Temporal Analysis")
        
        # Add month and year columns
        df['Month'] = df['date'].dt.month
        df['Year'] = df['date'].dt.year
        df['MonthYear'] = df['date'].dt.strftime('%Y-%m')
        
        # Monthly trends
        monthly_counts = df.groupby('MonthYear').size().reset_index(name='count')
        fig_monthly = px.line(
            monthly_counts,
            x='MonthYear',
            y='count',
            title="Monthly Bird Counts",
            labels={'count': 'Total Count', 'MonthYear': 'Month-Year'}
        )
        fig_monthly.update_layout(
            height=500,
            width=800,
            xaxis_tickangle=-45
        )
        st.plotly_chart(fig_monthly)
        
        # Species by month heatmap
        monthly_species = df.pivot_table(
            index='MonthYear',
            columns='label',
            values='count',
            aggfunc='sum',
            fill_value=0
        )
        
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(monthly_species, cmap='YlOrRd', ax=ax)
        plt.title("Species Distribution by Month")
        plt.xticks(rotation=45)
        st.pyplot(fig)
    
    with tab2:
        st.subheader("Flight Analysis")
        
        # Site distribution
        site_counts = df.groupby('flight_name').size().sort_values(ascending=False)
        
        fig_sites = px.bar(
            x=site_counts.index,
            y=site_counts.values,
            title="Bird Counts by Flight",
            labels={'x': 'Flight', 'y': 'Total Count'}
        )
        fig_sites.update_layout(
            xaxis_tickangle=-45,
            height=500,
            width=800
        )
        st.plotly_chart(fig_sites)
        
        # Species distribution by flight
        df["count"] = 1
        flight_species = df.pivot_table(
            index='flight_name',
            columns='label',
            values='count',
            aggfunc='sum',
            fill_value=0
        )
        
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(flight_species, cmap='YlOrRd', ax=ax)
        plt.title("Species Distribution by Flight")
        plt.xticks(rotation=45)
        st.pyplot(fig)

if __name__ == "__main__":
    load_css()
    app() 