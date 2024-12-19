import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt

def app():
    st.title("Bird Species Analysis")
    
    # Read the data
    data_path = Path(__file__).parents[1] / "data" / "PredictedBirds.csv"
    
    if not data_path.exists():
        st.error(f"File not found: {data_path}")
        return
        
    df = pd.read_csv(data_path)
    
    # Convert date with specific format
    df['Date'] = pd.to_datetime(df['Date'], format='%m_%d_%Y')
    
    # Display basic statistics
    st.header("Dataset Overview")
    st.write(f"Total Records: {len(df)}")
    st.write(f"Number of Species: {df['label'].nunique()}")
    st.write(f"Total Sites: {df['Site'].nunique()}")
    st.write(f"Date Range: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
    
    # Add tabs for different visualizations
    tab1, tab2, tab3 = st.tabs([
        "Species Distribution", 
        "Temporal Analysis",
        "Site Analysis"
    ])
    
    with tab1:
        st.subheader("Species Distribution")
        
        # Top N species by count
        n_species = st.slider("Select number of top species to display", 5, 20, 10)
        
        # Create species count plot
        species_counts = df.groupby('label')['count'].sum().sort_values(ascending=False).head(n_species)
        
        fig = px.bar(
            x=species_counts.index,
            y=species_counts.values,
            title=f"Top {n_species} Most Common Bird Species",
            labels={'x': 'Species', 'y': 'Total Count'}
        )
        fig.update_layout(
            xaxis_tickangle=-45,
            height=500,
            width=800
        )
        st.plotly_chart(fig)
        
        # Pie chart
        fig_pie = px.pie(
            values=species_counts.values,
            names=species_counts.index,
            title=f"Distribution of Top {n_species} Species"
        )
        fig_pie.update_layout(height=500, width=800)
        st.plotly_chart(fig_pie)
    
    with tab2:
        st.subheader("Temporal Analysis")
        
        # Add month and year columns
        df['Month'] = df['Date'].dt.month
        df['Year'] = df['Date'].dt.year
        df['MonthYear'] = df['Date'].dt.strftime('%Y-%m')
        
        # Monthly trends
        monthly_counts = df.groupby('MonthYear')['count'].sum().reset_index()
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
    
    with tab3:
        st.subheader("Site Analysis")
        
        # Site distribution
        site_counts = df.groupby('Site')['count'].sum().sort_values(ascending=False)
        
        fig_sites = px.bar(
            x=site_counts.index,
            y=site_counts.values,
            title="Bird Counts by Site",
            labels={'x': 'Site', 'y': 'Total Count'}
        )
        fig_sites.update_layout(
            xaxis_tickangle=-45,
            height=500,
            width=800
        )
        st.plotly_chart(fig_sites)
        
        # Species distribution by site
        site_species = df.pivot_table(
            index='Site',
            columns='label',
            values='count',
            aggfunc='sum',
            fill_value=0
        )
        
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(site_species, cmap='YlOrRd', ax=ax)
        plt.title("Species Distribution by Site")
        plt.xticks(rotation=45)
        st.pyplot(fig)

if __name__ == "__main__":
    app() 