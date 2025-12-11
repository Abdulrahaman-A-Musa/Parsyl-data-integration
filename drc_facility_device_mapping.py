# ================================
# DRC FACILITY-DEVICE MAPPING DASHBOARD
# Machine Learning-Enhanced Geographic Analysis
# ================================

import streamlit as st
import pandas as pd
import numpy as np
# import plotly.express as px
# import plotly.graph_objects as go
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import folium
from streamlit_folium import folium_static
from io import BytesIO
import base64

# ---------------- PAGE CONFIGURATION ----------------
st.set_page_config(
    page_title="DRC Facility-Device Mapping Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🗺️"
)

# ---------------- CUSTOM CSS STYLING ----------------
st.markdown(
    """
    <style>
    /* Simple Clean Theme */
    :root {
        --primary-color: #1e88e5;
        --secondary-color: #43a047;
        --text-dark: #2c3e50;
        --text-light: #7f8c8d;
        --border-color: #e0e0e0;
        --bg-light: #f5f7fa;
    }
    
    /* Dashboard Header */
    .dashboard-header {
        background: #1e88e5;
        color: white;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    .dashboard-title {
        font-size: 2.5em;
        font-weight: 700;
        margin: 0;
    }
    
    .dashboard-subtitle {
        font-size: 1.1em;
        margin-top: 0.5rem;
        opacity: 0.9;
    }
    
    /* Metric Cards */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 8px;
        border: 2px solid #e0e0e0;
        margin-bottom: 1rem;
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e88e5;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #7f8c8d;
        font-weight: 600;
        text-transform: uppercase;
    }
    
    /* Section Headers */
    .section-header {
        background: white;
        padding: 1rem 1.5rem;
        border-left: 4px solid #1e88e5;
        border-radius: 5px;
        margin: 2rem 0 1rem 0;
        border: 1px solid #e0e0e0;
    }
    
    .section-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2c3e50;
        margin: 0;
    }
    
    /* Alert Boxes */
    .alert-success {
        background: #e8f5e9;
        border-left: 4px solid #43a047;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
        color: #2e7d32;
    }
    
    .alert-warning {
        background: #fff3e0;
        border-left: 4px solid #ff9800;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
        color: #e65100;
    }
    
    .alert-info {
        background: #e3f2fd;
        border-left: 4px solid #1e88e5;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
        color: #1565c0;
    }
    
    /* Buttons */
    .stButton > button {
        background: #1e88e5;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 2rem;
        font-weight: 600;
    }
    
    .stButton > button:hover {
        background: #1565c0;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------- HELPER FUNCTIONS ----------------
@st.cache_data
def load_sample_data():
    """Generate sample DRC facility data for demonstration"""
    np.random.seed(42)
    
    # DRC provinces and major cities
    provinces = {
        'Kinshasa': {'lat': -4.3276, 'lon': 15.3136},
        'Katanga': {'lat': -11.6795, 'lon': 27.4795},
        'Nord-Kivu': {'lat': -0.7167, 'lon': 29.2333},
        'Sud-Kivu': {'lat': -2.5000, 'lon': 28.8333},
        'Kasai-Oriental': {'lat': -4.3333, 'lon': 23.5833},
        'Equateur': {'lat': 0.0500, 'lon': 18.2833},
    }
    
    facilities = []
    device_id = 1000
    
    for province, coords in provinces.items():
        # Generate 5-15 facilities per province
        n_facilities = np.random.randint(5, 16)
        for i in range(n_facilities):
            # Add some random variation to coordinates
            lat = coords['lat'] + np.random.uniform(-0.5, 0.5)
            lon = coords['lon'] + np.random.uniform(-0.5, 0.5)
            
            # Random number of devices per facility (1-20)
            device_count = np.random.randint(1, 21)
            
            for _ in range(device_count):
                facilities.append({
                    'device_id': device_id,
                    'latitude': lat + np.random.uniform(-0.01, 0.01),
                    'longitude': lon + np.random.uniform(-0.01, 0.01),
                    'country': 'DRC',
                    'Province': province,
                    'facility_name': f'{province}_Facility_{i+1}'
                })
                device_id += 1
    
    return pd.DataFrame(facilities)

def cluster_facilities(df, n_clusters=5):
    """Apply K-Means clustering to identify facility groups"""
    # Prepare features for clustering
    features = df[['latitude', 'longitude', 'device_count']].copy()
    
    # Normalize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Apply K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(features_scaled)
    
    return df, kmeans

def predict_facility_priority(df):
    """Predict facility priority based on device count and location"""
    # Create priority categories based on device count
    df['priority_actual'] = pd.cut(
        df['device_count'], 
        bins=[0, 5, 10, 20, float('inf')],
        labels=['Low', 'Medium', 'High', 'Critical']
    )
    
    return df

def create_interactive_map(df):
    """Create interactive Folium map with facility markers"""
    # Calculate center of map
    center_lat = df['latitude'].mean()
    center_lon = df['longitude'].mean()
    
    # Create map with professional CartoDB tiles
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=8,  # Increased zoom for closer default view
        tiles='CartoDB positron',
        control_scale=True,
        prefer_canvas=True
    )
    
    # Add alternative tile layers
    folium.TileLayer('OpenStreetMap', name='Street Map').add_to(m)
    
    # Add Google Hybrid (Satellite + Labels)
    folium.TileLayer(
        tiles='https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}',
        attr='Google',
        name='Google Hybrid (Satellite)',
        overlay=False,
        control=True
    ).add_to(m)
    
    # Add Google Satellite (no labels)
    folium.TileLayer(
        tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
        attr='Google',
        name='Google Satellite',
        overlay=False,
        control=True
    ).add_to(m)
    
    # Create marker clusters by priority
    from folium.plugins import MarkerCluster
    
    # Add cluster circles first (background layer)
    for cluster_id in sorted(df['cluster'].unique()):
        cluster_data = df[df['cluster'] == cluster_id]
        center_lat = cluster_data['latitude'].mean()
        center_lon = cluster_data['longitude'].mean()
        total_devices = cluster_data['device_count'].sum()
        
        # Color gradient for clusters
        cluster_colors = ['#9c27b0', '#673ab7', '#3f51b5', '#2196f3', '#00bcd4', 
                         '#009688', '#4caf50', '#8bc34a', '#cddc39', '#ffeb3b']
        cluster_color = cluster_colors[cluster_id % len(cluster_colors)]
        
        folium.Circle(
            location=[center_lat, center_lon],
            radius=50000,  # 50km radius
            color=cluster_color,
            fill=True,
            fillColor=cluster_color,
            fillOpacity=0.15,
            weight=2,
            opacity=0.6,
            popup=folium.Popup(f"""
                <div style="font-family: Arial; font-size: 13px; min-width: 150px;">
                    <h4 style="margin: 5px 0; color: {cluster_color}; text-align: center;">
                        🎯 Cluster {cluster_id}
                    </h4>
                    <hr style="margin: 5px 0;">
                    <b>Facilities:</b> {len(cluster_data)}<br>
                    <b>Total Devices:</b> {total_devices}<br>
                    <b>Avg Devices:</b> {cluster_data['device_count'].mean():.1f}
                </div>
            """, max_width=200),
            tooltip=f'Cluster {cluster_id}: {len(cluster_data)} facilities'
        ).add_to(m)
    
    # Add markers for each facility with custom icons
    for _, row in df.iterrows():
        # Color and icon based on device count
        if row['device_count'] <= 5:
            color = '#4caf50'  # Green
            icon_name = 'glyphicon-ok-circle'
            priority_emoji = '🟢'
        elif row['device_count'] <= 10:
            color = '#2196f3'  # Blue
            icon_name = 'glyphicon-plus-sign'
            priority_emoji = '🔵'
        elif row['device_count'] <= 15:
            color = '#ff9800'  # Orange
            icon_name = 'glyphicon-warning-sign'
            priority_emoji = '🟠'
        else:
            color = '#f44336'  # Red
            icon_name = 'glyphicon-exclamation-sign'
            priority_emoji = '🔴'
        
        # Enhanced popup with professional styling
        popup_html = f"""
        <div style="font-family: 'Segoe UI', Arial, sans-serif; font-size: 13px; min-width: 250px;">
            <div style="background: {color}; color: white; padding: 10px; margin: -10px -10px 10px -10px; border-radius: 5px 5px 0 0;">
                <h3 style="margin: 0; font-size: 16px; font-weight: 600;">
                    {priority_emoji} {row['facility_name']}
                </h3>
            </div>
            <table style="width: 100%; border-collapse: collapse;">
                <tr style="border-bottom: 1px solid #e0e0e0;">
                    <td style="padding: 5px; font-weight: 600; color: #555;">📍 Province:</td>
                    <td style="padding: 5px; text-align: right;">{row['Province']}</td>
                </tr>
                <tr style="border-bottom: 1px solid #e0e0e0; background: #f9f9f9;">
                    <td style="padding: 5px; font-weight: 600; color: #555;">📱 Devices:</td>
                    <td style="padding: 5px; text-align: right; font-weight: 700; color: {color};">{row['device_count']}</td>
                </tr>
                <tr style="border-bottom: 1px solid #e0e0e0;">
                    <td style="padding: 5px; font-weight: 600; color: #555;">🎯 Cluster:</td>
                    <td style="padding: 5px; text-align: right;">Cluster {row['cluster']}</td>
                </tr>
                <tr style="border-bottom: 1px solid #e0e0e0; background: #f9f9f9;">
                    <td style="padding: 5px; font-weight: 600; color: #555;">⚡ Priority:</td>
                    <td style="padding: 5px; text-align: right;">
                        <span style="background: {color}; color: white; padding: 2px 8px; border-radius: 3px; font-size: 11px; font-weight: 600;">
                            {row['priority_actual']}
                        </span>
                    </td>
                </tr>
                <tr>
                    <td colspan="2" style="padding: 5px; font-size: 11px; color: #888; text-align: center;">
                        📍 {row['latitude']:.4f}, {row['longitude']:.4f}
                    </td>
                </tr>
            </table>
        </div>
        """
        
        # Add custom marker
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=8 + (row['device_count'] / 3),  # Size scales with device count
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"{row['facility_name']} ({row['device_count']} devices)",
            color=color,
            fillColor=color,
            fillOpacity=0.8,
            weight=2,
            opacity=1
        ).add_to(m)
    
    # Add fullscreen button
    from folium.plugins import Fullscreen
    Fullscreen(position='topleft').add_to(m)
    
    # Add layer control
    folium.LayerControl(position='topright').add_to(m)
    
    # Add legend
    legend_html = f"""
    <div style="position: fixed; bottom: 50px; right: 50px; z-index: 1000; 
                background: white; padding: 15px; border-radius: 10px;
                box-shadow: 0 4px 15px rgba(0,0,0,0.2); font-family: Arial; font-size: 13px;
                border: 2px solid #1e88e5;">
        <h4 style="margin: 0 0 10px 0; color: #1e88e5; font-size: 14px; font-weight: 700;">
            📊 Device Count Legend
        </h4>
        <div style="margin: 5px 0;">
            <span style="background: #4caf50; width: 15px; height: 15px; display: inline-block; 
                         border-radius: 50%; margin-right: 8px; border: 2px solid white; box-shadow: 0 2px 4px rgba(0,0,0,0.2);"></span>
            <span style="color: #555;">1-5 devices (Low)</span>
        </div>
        <div style="margin: 5px 0;">
            <span style="background: #2196f3; width: 15px; height: 15px; display: inline-block; 
                         border-radius: 50%; margin-right: 8px; border: 2px solid white; box-shadow: 0 2px 4px rgba(0,0,0,0.2);"></span>
            <span style="color: #555;">6-10 devices (Medium)</span>
        </div>
        <div style="margin: 5px 0;">
            <span style="background: #ff9800; width: 15px; height: 15px; display: inline-block; 
                         border-radius: 50%; margin-right: 8px; border: 2px solid white; box-shadow: 0 2px 4px rgba(0,0,0,0.2);"></span>
            <span style="color: #555;">11-15 devices (High)</span>
        </div>
        <div style="margin: 5px 0;">
            <span style="background: #f44336; width: 15px; height: 15px; display: inline-block; 
                         border-radius: 50%; margin-right: 8px; border: 2px solid white; box-shadow: 0 2px 4px rgba(0,0,0,0.2);"></span>
            <span style="color: #555;">16+ devices (Critical)</span>
        </div>
        <hr style="margin: 10px 0; border: none; border-top: 1px solid #e0e0e0;">
        <div style="font-size: 11px; color: #888; text-align: center;">
            💡 Click markers for details
        </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    
    return m

def generate_ml_insights(df):
    """Generate ML-based insights about the facility-device distribution"""
    insights = []
    
    # Insight 1: Device concentration
    high_device_facilities = df[df['device_count'] > 15]
    if len(high_device_facilities) > 0:
        insights.append({
            'type': 'warning',
            'title': 'High Device Concentration',
            'text': f'{len(high_device_facilities)} facilities have >15 devices. Consider redistribution for optimal coverage.'
        })
    
    # Insight 2: Provincial distribution
    province_stats = df.groupby('Province')['device_count'].agg(['sum', 'mean']).round(2)
    max_province = province_stats['sum'].idxmax()
    min_province = province_stats['sum'].idxmin()
    
    insights.append({
        'type': 'info',
        'title': 'Provincial Distribution',
        'text': f'{max_province} has the most devices ({province_stats.loc[max_province, "sum"]:.0f}), while {min_province} has the least ({province_stats.loc[min_province, "sum"]:.0f}).'
    })
    
    # Insight 3: Cluster analysis
    cluster_stats = df.groupby('cluster').size()
    largest_cluster = cluster_stats.idxmax()
    
    insights.append({
        'type': 'success',
        'title': 'Geographic Clustering',
        'text': f'ML identified {df["cluster"].nunique()} distinct facility clusters. Cluster {largest_cluster} contains {cluster_stats[largest_cluster]} facilities.'
    })
    
    return insights

# ---------------- MAIN APPLICATION ----------------
def main():
    # Header
    st.markdown(
        """
        <div class="dashboard-header">
            <h1 class="dashboard-title">🗺️ DRC Facility-Device Mapping</h1>
            <p class="dashboard-subtitle">ML-Enhanced Geographic Analysis & Optimization Dashboard</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🎛️ Dashboard Controls")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload Your Data (CSV/Excel)",
            type=['csv', 'xlsx'],
            help="Upload a file with columns: latitude, longitude, device_id, country, Province"
        )
        
        st.markdown("---")
        
        # ML Settings
        st.markdown("### 🤖 ML Settings")
        n_clusters = st.slider("Number of Clusters", 3, 10, 5)
        
        st.markdown("---")
        
        # Filters
        st.markdown("### 🔍 Data Filters")
        
    # Load data
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df_raw = pd.read_csv(uploaded_file)
            else:
                df_raw = pd.read_excel(uploaded_file)
            st.success("✅ Data loaded successfully!")
            
            # Show detected columns for debugging
            with st.expander("📋 View Detected Columns"):
                st.write("**Columns in your data:**")
                st.write(list(df_raw.columns))
                st.write(f"\n**Total rows:** {len(df_raw):,}")
                st.dataframe(df_raw.head(3))
            
        except Exception as e:
            st.error(f"❌ Error loading file: {e}")
            return
    else:
        st.info("ℹ️ Using sample DRC data for demonstration. Upload your own data in the sidebar.")
        df_raw = load_sample_data()
    
    # Data preprocessing - Handle both sample and uploaded data
    required_cols = ['latitude', 'longitude', 'country', 'Province']
    
    # Check if required columns exist (case-insensitive)
    df_raw.columns = df_raw.columns.str.strip()  # Remove whitespace
    col_mapping = {}
    for req_col in required_cols:
        found = False
        for actual_col in df_raw.columns:
            if actual_col.lower() == req_col.lower():
                col_mapping[actual_col] = req_col
                found = True
                break
        if not found and req_col not in ['country']:  # country is optional
            st.error(f"❌ Required column '{req_col}' not found in your data!")
            st.info("Your data must have columns: latitude, longitude, Province (country is optional)")
            return
    
    # Rename columns to standard names
    df_raw = df_raw.rename(columns=col_mapping)
    
    # Add country column if missing
    if 'country' not in df_raw.columns:
        df_raw['country'] = 'DRC'
    
    # Group by facility location and count devices
    # Each row in your data represents a device at a facility
    df_facilities = df_raw.groupby(['latitude', 'longitude', 'country', 'Province']).size().reset_index(name='device_count')
    df_facilities['facility_name'] = [f"Facility_{i+1}" for i in range(len(df_facilities))]
    
    # Apply ML clustering
    with st.spinner('🤖 Running ML analysis...'):
        df_facilities, kmeans = cluster_facilities(df_facilities, n_clusters)
        df_facilities = predict_facility_priority(df_facilities)
    
    # Sidebar filters (continued)
    with st.sidebar:
        provinces = ['All'] + sorted(df_facilities['Province'].unique().tolist())
        selected_province = st.selectbox("Filter by Province", provinces)
        
        if selected_province != 'All':
            df_filtered = df_facilities[df_facilities['Province'] == selected_province]
        else:
            df_filtered = df_facilities.copy()
        
        st.markdown("---")
        st.markdown(f"**Facilities Displayed:** {len(df_filtered):,}")
        st.markdown(f"**Total Devices:** {df_filtered['device_count'].sum():,}")
    
    # KPIs
    st.markdown('<div class="section-header"><h2 class="section-title">📊 Key Performance Indicators</h2></div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">🏥 Total Facilities</div>
                <div class="metric-value">{len(df_filtered):,}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">📱 Total Devices</div>
                <div class="metric-value">{df_filtered['device_count'].sum():,}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">📈 Avg Devices/Facility</div>
                <div class="metric-value">{df_filtered['device_count'].mean():.1f}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">🗺️ Provinces Covered</div>
                <div class="metric-value">{df_filtered['Province'].nunique()}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    # Interactive Map (Moved to top)
    st.markdown('<div class="section-header"><h2 class="section-title">🗺️ Interactive Facility Map</h2></div>', unsafe_allow_html=True)
    
    # Map instructions
    st.info("💡 **Map Controls:** Click markers for facility details | Use mouse wheel to zoom | Switch map styles in top-right corner | Fullscreen available")
    
    # Create and display map
    facility_map = create_interactive_map(df_filtered)
    folium_static(facility_map, width=1400, height=700)
    
    # Device Distribution Table (Below Map)
    st.markdown('<div class="section-header"><h2 class="section-title">📋 Device Distribution by Facility</h2></div>', unsafe_allow_html=True)
    
    # Prepare table data
    table_data = df_filtered[['facility_name', 'Province', 'device_count', 'latitude', 'longitude', 'cluster', 'priority_actual']].copy()
    table_data = table_data.sort_values('device_count', ascending=False).reset_index(drop=True)
    table_data.index = table_data.index + 1  # Start index from 1
    
    # Rename columns for display
    table_data = table_data.rename(columns={
        'facility_name': 'Facility Name',
        'Province': 'Province',
        'device_count': 'Total Devices',
        'latitude': 'Latitude',
        'longitude': 'Longitude',
        'cluster': 'Cluster ID',
        'priority_actual': 'Priority Level'
    })
    
    # Add percentage column
    total_devices = table_data['Total Devices'].sum()
    table_data['% of Total'] = (table_data['Total Devices'] / total_devices * 100).round(2)
    
    # Display summary statistics above table
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📍 Total Facilities", len(table_data))
    with col2:
        st.metric("📱 Total Devices", f"{total_devices:,}")
    with col3:
        st.metric("📊 Average per Facility", f"{table_data['Total Devices'].mean():.1f}")
    with col4:
        highest_facility = table_data.iloc[0]
        st.metric("🏆 Highest Allocation", f"{highest_facility['Facility Name']} ({highest_facility['Total Devices']})")
    
    st.markdown("---")
    
    # Display the table with styling
    def highlight_priority(row):
        """Apply color coding based on priority"""
        if row['Priority Level'] == 'Critical':
            return ['background-color: #ffebee'] * len(row)
        elif row['Priority Level'] == 'High':
            return ['background-color: #fff3e0'] * len(row)
        elif row['Priority Level'] == 'Medium':
            return ['background-color: #e3f2fd'] * len(row)
        else:
            return ['background-color: #e8f5e9'] * len(row)
    
    # Apply styling and display
    styled_table = table_data.style.apply(highlight_priority, axis=1)\
        .format({
            'Total Devices': '{:,.0f}',
            '% of Total': '{:.2f}%',
            'Latitude': '{:.4f}',
            'Longitude': '{:.4f}'
        })\
        .set_properties(**{
            'text-align': 'center',
            'font-size': '13px',
            'border': '1px solid #e0e0e0'
        })\
        .set_table_styles([
            {'selector': 'th', 'props': [
                ('background-color', '#1e88e5'),
                ('color', 'white'),
                ('font-weight', 'bold'),
                ('text-align', 'center'),
                ('padding', '10px'),
                ('font-size', '14px')
            ]},
            {'selector': 'td', 'props': [
                ('padding', '8px')
            ]},
            {'selector': 'tr:hover', 'props': [
                ('background-color', '#f5f5f5')
            ]}
        ])
    
    st.dataframe(styled_table, use_container_width=True, height=400)
    
    # Download button for the table
    csv = table_data.to_csv(index=True, index_label='Rank')
    st.download_button(
        label="📥 Download Device Distribution Table (CSV)",
        data=csv,
        file_name=f"device_distribution_{df_filtered['Province'].iloc[0] if len(df_filtered['Province'].unique()) == 1 else 'all_provinces'}.csv",
        mime="text/csv",
        use_container_width=True
    )
    
    # Summary by Province
    st.markdown("### 🗺️ Provincial Summary")
    province_summary = df_filtered.groupby('Province').agg({
        'facility_name': 'count',
        'device_count': ['sum', 'mean', 'min', 'max']
    }).round(2)
    province_summary.columns = ['Facilities', 'Total Devices', 'Avg Devices', 'Min Devices', 'Max Devices']
    province_summary = province_summary.sort_values('Total Devices', ascending=False)
    
    st.dataframe(
        province_summary.style
        .format({
            'Total Devices': '{:,.0f}',
            'Avg Devices': '{:.1f}',
            'Min Devices': '{:.0f}',
            'Max Devices': '{:.0f}'
        })
        .set_properties(**{
            'text-align': 'center',
            'font-size': '13px',
            'border': '1px solid #e0e0e0'
        })
        .set_table_styles([
            {'selector': 'th', 'props': [
                ('background-color', '#1e88e5'),
                ('color', 'white'),
                ('font-weight', 'bold'),
                ('text-align', 'center'),
                ('padding', '10px')
            ]}
        ]),
        use_container_width=True
    )
    
    # ML Insights (Moved below map)
    st.markdown('<div class="section-header"><h2 class="section-title">🤖 ML-Generated Insights</h2></div>', unsafe_allow_html=True)
    
    insights = generate_ml_insights(df_filtered)
    
    for insight in insights:
        alert_class = f"alert-{insight['type']}"
        st.markdown(
            f"""
            <div class="{alert_class}">
                <strong>{insight['title']}</strong><br>
                {insight['text']}
            </div>
            """,
            unsafe_allow_html=True
        )
    
    # Charts
    st.markdown('<div class="section-header"><h2 class="section-title">📈 Data Analysis & Visualizations</h2></div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Provincial Analysis", "🎯 Cluster Analysis", "📈 Device Distribution", "🗂️ Priority Facilities"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            # Provincial device count
            province_stats = df_filtered.groupby('Province')['device_count'].sum().reset_index()
            province_stats = province_stats.sort_values('device_count', ascending=False)
            
            fig = px.bar(
                province_stats,
                x='Province',
                y='device_count',
                title='Total Devices by Province',
                color='device_count',
                color_continuous_scale='Blues',
                labels={'device_count': 'Total Devices'}
            )
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Provincial facility count
            province_facilities = df_filtered.groupby('Province').size().reset_index(name='facility_count')
            
            fig = px.pie(
                province_facilities,
                values='facility_count',
                names='Province',
                title='Facility Distribution by Province',
                hole=0.4
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            # Cluster sizes
            cluster_stats = df_filtered.groupby('cluster').agg({
                'device_count': 'sum',
                'facility_name': 'count'
            }).reset_index()
            cluster_stats.columns = ['Cluster', 'Total Devices', 'Facilities']
            
            fig = px.bar(
                cluster_stats,
                x='Cluster',
                y=['Total Devices', 'Facilities'],
                title='ML Cluster Analysis',
                barmode='group',
                color_discrete_sequence=['#2563eb', '#7c3aed']
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Scatter plot with clusters
            fig = px.scatter(
                df_filtered,
                x='longitude',
                y='latitude',
                color='cluster',
                size='device_count',
                hover_data=['facility_name', 'Province', 'device_count'],
                title='Geographic Clusters (ML-Generated)',
                color_continuous_scale='Viridis'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        col1, col2 = st.columns(2)
        
        with col1:
            # Device count distribution
            fig = px.histogram(
                df_filtered,
                x='device_count',
                nbins=20,
                title='Device Count Distribution',
                color_discrete_sequence=['#2563eb']
            )
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Box plot by province
            fig = px.box(
                df_filtered,
                x='Province',
                y='device_count',
                title='Device Count Range by Province',
                color='Province'
            )
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        # Priority facilities table
        priority_facilities = df_filtered.sort_values('device_count', ascending=False).head(20)
        
        st.markdown("### Top 20 Facilities by Device Count")
        
        display_df = priority_facilities[[
            'facility_name', 'Province', 'device_count', 'cluster', 'priority_actual', 'latitude', 'longitude'
        ]].rename(columns={
            'facility_name': 'Facility',
            'device_count': 'Devices',
            'cluster': 'Cluster',
            'priority_actual': 'Priority',
            'latitude': 'Latitude',
            'longitude': 'Longitude'
        })
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        # Download button
        csv = display_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Priority Facilities (CSV)",
            data=csv,
            file_name="priority_facilities.csv",
            mime="text/csv"
        )
    
    # Summary Statistics
    st.markdown('<div class="section-header"><h2 class="section-title">📋 Summary Statistics</h2></div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### Device Statistics")
        stats_df = pd.DataFrame({
            'Metric': ['Total', 'Mean', 'Median', 'Std Dev', 'Min', 'Max'],
            'Value': [
                f"{df_filtered['device_count'].sum():,}",
                f"{df_filtered['device_count'].mean():.2f}",
                f"{df_filtered['device_count'].median():.1f}",
                f"{df_filtered['device_count'].std():.2f}",
                f"{df_filtered['device_count'].min()}",
                f"{df_filtered['device_count'].max()}"
            ]
        })
        st.dataframe(stats_df, use_container_width=True, hide_index=True)
    
    with col2:
        st.markdown("### Provincial Summary")
        province_summary = df_filtered.groupby('Province').agg({
            'facility_name': 'count',
            'device_count': 'sum'
        }).rename(columns={
            'facility_name': 'Facilities',
            'device_count': 'Devices'
        }).reset_index()
        st.dataframe(province_summary, use_container_width=True, hide_index=True)
    
    with col3:
        st.markdown("### Priority Distribution")
        priority_dist = df_filtered['priority_actual'].value_counts().reset_index()
        priority_dist.columns = ['Priority', 'Count']
        st.dataframe(priority_dist, use_container_width=True, hide_index=True)
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #64748b; padding: 2rem;">
            <p><strong>DRC Facility-Device Mapping Dashboard</strong></p>
            <p>ML-Enhanced Geographic Analysis | Powered by Streamlit, Plotly & Scikit-learn</p>
            <p style="font-size: 0.9rem; margin-top: 1rem;">Dashboard Version 1.0 | December 2025</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
