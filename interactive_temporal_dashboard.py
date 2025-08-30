#!/usr/bin/env python3
"""
Interactive Temporal Music Dashboard

Real-time interactive exploration of 4+ years of listening patterns:
1. 🕐 Interactive temporal dashboard with all time patterns
2. 🎵 Genre intelligence explorer with playlist mapping
3. 📈 Comprehensive Musical Life timeline with interactive features

71,051 streaming records + 634 playlist tracks analyzed interactively.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="🕐 Interactive Temporal Music Dashboard",
    page_icon="🕐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .insight-box {
        border-left: 4px solid #1f77b4;
        padding: 1rem;
        margin: 0.5rem 0;
        background-color: #fafafa;
    }
    .timeline-event {
        border-left: 3px solid #ff7f0e;
        padding: 0.5rem;
        margin: 0.3rem 0;
        background-color: #fff5ee;
    }
</style>
""", unsafe_allow_html=True)

class InteractiveTemporalDashboard:
    """Interactive dashboard for comprehensive temporal music analysis"""
    
    def __init__(self):
        self.streaming_data = None
        self.playlist_data = None
        self.temporal_insights = {}
        self.genre_intelligence = {}
        self.musical_timeline = {}
        
    @st.cache_data
    def load_data(_self):
        """Load and cache streaming and playlist data"""
        try:
            # Load streaming data (71K records)
            streaming_data = pd.read_parquet('data/processed/streaming_data_processed.parquet')
            streaming_data['played_at'] = pd.to_datetime(streaming_data['played_at'])
            
            # Load playlist data (634 tracks)
            playlist_dir = Path('/Users/quangnguyen/Downloads/spotify_playlists')
            playlists = []
            
            if playlist_dir.exists():
                for csv_file in playlist_dir.glob('*.csv'):
                    df = pd.read_csv(csv_file)
                    df['playlist_name'] = csv_file.stem
                    df['Added At'] = pd.to_datetime(df['Added At'])
                    playlists.append(df)
                
                playlist_data = pd.concat(playlists, ignore_index=True) if playlists else pd.DataFrame()
            else:
                playlist_data = pd.DataFrame()
            
            return streaming_data, playlist_data
            
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return None, None
    
    def process_temporal_data(self, streaming_data):
        """Process temporal components of streaming data"""
        data = streaming_data.copy()
        
        # Extract time components
        data['hour'] = data['played_at'].dt.hour
        data['day_of_week'] = data['played_at'].dt.day_name()
        data['month'] = data['played_at'].dt.month_name()
        data['year'] = data['played_at'].dt.year
        data['date'] = data['played_at'].dt.date
        data['is_weekend'] = data['played_at'].dt.weekday >= 5
        data['season'] = data['month'].map({
            'December': 'Winter', 'January': 'Winter', 'February': 'Winter',
            'March': 'Spring', 'April': 'Spring', 'May': 'Spring',
            'June': 'Summer', 'July': 'Summer', 'August': 'Summer',
            'September': 'Fall', 'October': 'Fall', 'November': 'Fall'
        })
        
        # Session analysis
        data = data.sort_values('played_at')
        data['time_gap'] = data['played_at'].diff().dt.total_seconds() / 60  # minutes
        data['new_session'] = (data['time_gap'] > 30) | (data['time_gap'].isna())
        data['session_id'] = data['new_session'].cumsum()
        
        return data
    
    def show_temporal_patterns_dashboard(self, data):
        """Interactive temporal patterns dashboard"""
        
        st.header("🕐 Interactive Temporal Patterns Dashboard")
        
        # Sidebar filters
        st.sidebar.subheader("🎛️ Temporal Filters")
        
        # Year filter
        available_years = sorted(data['year'].unique())
        selected_years = st.sidebar.multiselect(
            "Select Years", 
            available_years, 
            default=available_years,
            help="Filter data by specific years"
        )
        
        # Season filter
        available_seasons = data['season'].unique()
        selected_seasons = st.sidebar.multiselect(
            "Select Seasons",
            available_seasons,
            default=available_seasons,
            help="Filter by seasons"
        )
        
        # Filter data
        filtered_data = data[
            (data['year'].isin(selected_years)) & 
            (data['season'].isin(selected_seasons))
        ]
        
        if len(filtered_data) == 0:
            st.warning("No data matches the selected filters!")
            return
        
        # Key metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Plays", f"{len(filtered_data):,}")
        with col2:
            st.metric("Unique Artists", f"{filtered_data['artist_name'].nunique():,}")
        with col3:
            st.metric("Total Hours", f"{filtered_data['minutes_played'].sum()/60:.0f}")
        with col4:
            avg_daily = len(filtered_data) / ((filtered_data['played_at'].max() - filtered_data['played_at'].min()).days + 1)
            st.metric("Daily Avg", f"{avg_daily:.1f} plays")
        
        # Row 1: Hourly and Daily patterns
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("⏰ Hourly Listening Patterns")
            
            hourly_data = filtered_data.groupby('hour').size().reset_index(name='plays')
            
            fig = px.bar(
                hourly_data, 
                x='hour', 
                y='plays',
                title="Plays by Hour of Day",
                color='plays',
                color_continuous_scale='viridis'
            )
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Peak hour insight
            peak_hour = hourly_data.loc[hourly_data['plays'].idxmax(), 'hour']
            peak_plays = hourly_data['plays'].max()
            
            st.markdown(f"""
            <div class="insight-box">
                <strong>🎯 Peak Listening Hour:</strong> {peak_hour}:00 
                ({peak_plays:,} plays - {peak_plays/hourly_data['plays'].sum()*100:.1f}% of all listening)
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.subheader("📅 Daily Listening Patterns")
            
            # Daily patterns with proper ordering
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            daily_data = filtered_data.groupby('day_of_week').size().reindex(day_order).reset_index(name='plays')
            
            fig = px.bar(
                daily_data,
                x='day_of_week',
                y='plays', 
                title="Plays by Day of Week",
                color='plays',
                color_continuous_scale='plasma'
            )
            fig.update_layout(showlegend=False, height=400)
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
            
            # Most active day insight
            most_active_day = daily_data.loc[daily_data['plays'].idxmax(), 'day_of_week']
            most_active_plays = daily_data['plays'].max()
            
            st.markdown(f"""
            <div class="insight-box">
                <strong>📈 Most Active Day:</strong> {most_active_day}
                ({most_active_plays:,} plays)
            </div>
            """, unsafe_allow_html=True)
        
        # Row 2: Heatmap and Weekend Analysis
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("🔥 Listening Activity Heatmap")
            
            # Create pivot for heatmap
            pivot_data = filtered_data.groupby(['day_of_week', 'hour']).size().reset_index(name='plays')
            pivot_table = pivot_data.pivot(index='day_of_week', columns='hour', values='plays').fillna(0)
            pivot_table = pivot_table.reindex(day_order)
            
            fig = px.imshow(
                pivot_table.values,
                labels=dict(x="Hour of Day", y="Day of Week", color="Plays"),
                x=list(range(24)),
                y=day_order,
                color_continuous_scale='YlOrRd',
                title="Listening Intensity: Hour × Day Heatmap"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("🎉 Weekend vs Weekday")
            
            weekend_data = filtered_data.groupby('is_weekend').agg({
                'track_id': 'count',
                'minutes_played': 'mean'
            }).round(2)
            weekend_data.index = ['Weekday', 'Weekend']
            
            # Weekend comparison chart
            fig = px.bar(
                x=weekend_data.index,
                y=weekend_data['track_id'],
                title="Total Plays: Weekday vs Weekend",
                color=['#1f77b4', '#ff7f0e']
            )
            fig.update_layout(showlegend=False, height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            # Weekend boost calculation
            weekend_boost = weekend_data.loc['Weekend', 'track_id'] / weekend_data.loc['Weekday', 'track_id']
            
            st.markdown(f"""
            <div class="insight-box">
                <strong>🚀 Weekend Boost:</strong> {weekend_boost:.1f}x more active on weekends
            </div>
            """, unsafe_allow_html=True)
        
        # Row 3: Seasonal and Session Analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🍂 Seasonal Patterns")
            
            seasonal_data = filtered_data.groupby('season').size().reset_index(name='plays')
            
            fig = px.pie(
                seasonal_data,
                values='plays',
                names='season',
                title="Listening by Season",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("🎧 Listening Sessions")
            
            # Session analysis
            session_stats = filtered_data.groupby('session_id').agg({
                'track_id': 'count',
                'minutes_played': 'sum'
            })
            
            session_distribution = session_stats['track_id'].value_counts().head(10).reset_index()
            session_distribution.columns = ['tracks_per_session', 'session_count']
            
            fig = px.bar(
                session_distribution,
                x='tracks_per_session',
                y='session_count',
                title="Session Length Distribution",
                color='session_count',
                color_continuous_scale='viridis'
            )
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Session insights
            avg_session_length = session_stats['track_id'].mean()
            total_sessions = len(session_stats)
            
            st.markdown(f"""
            <div class="insight-box">
                <strong>📊 Session Stats:</strong><br/>
                • Total Sessions: {total_sessions:,}<br/>
                • Avg Session: {avg_session_length:.1f} tracks<br/>
                • Longest Session: {session_stats['track_id'].max()} tracks
            </div>
            """, unsafe_allow_html=True)
    
    def show_genre_intelligence_explorer(self, streaming_data, playlist_data):
        """Interactive genre intelligence explorer"""
        
        st.header("🎵 Genre Intelligence Explorer")
        
        # Cultural classification function
        def classify_culture(artist_name):
            if pd.isna(artist_name):
                return 'unknown'
            artist_lower = str(artist_name).lower()
            vietnamese_indicators = ['buitruonglinh', 'vsoul', 'khói', 'đen', 'mck', 'obito']
            if any(ind in artist_lower for ind in vietnamese_indicators) or \
               any(char in artist_lower for char in 'àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ'):
                return 'vietnamese'
            else:
                return 'western'
        
        # Add cultural classification
        streaming_data['cultural_class'] = streaming_data['artist_name'].apply(classify_culture)
        
        # Sidebar controls
        st.sidebar.subheader("🎭 Genre Filters")
        
        # Cultural focus
        cultural_focus = st.sidebar.selectbox(
            "Cultural Focus",
            ["All Cultures", "Vietnamese Only", "Western Only", "Cross-Cultural Analysis"]
        )
        
        # Artist exploration
        top_artists = streaming_data['artist_name'].value_counts().head(50).index.tolist()
        selected_artists = st.sidebar.multiselect(
            "Focus on Specific Artists",
            top_artists,
            help="Select artists to analyze in detail"
        )
        
        # Apply filters
        if cultural_focus == "Vietnamese Only":
            filtered_data = streaming_data[streaming_data['cultural_class'] == 'vietnamese']
        elif cultural_focus == "Western Only":
            filtered_data = streaming_data[streaming_data['cultural_class'] == 'western']
        else:
            filtered_data = streaming_data
        
        if selected_artists:
            filtered_data = filtered_data[filtered_data['artist_name'].isin(selected_artists)]
        
        # Row 1: Cultural Overview
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🌍 Cultural Distribution")
            
            cultural_stats = filtered_data['cultural_class'].value_counts().reset_index()
            cultural_stats.columns = ['culture', 'plays']
            
            fig = px.pie(
                cultural_stats,
                values='plays',
                names='culture',
                title="Listening by Culture",
                color_discrete_map={
                    'vietnamese': '#ff6b6b',
                    'western': '#4ecdc4',
                    'unknown': '#95a5a6'
                }
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("🎤 Top Artists Analysis")
            
            artist_stats = filtered_data.groupby(['artist_name', 'cultural_class']).size().reset_index(name='plays')
            top_artist_stats = artist_stats.nlargest(15, 'plays')
            
            fig = px.bar(
                top_artist_stats,
                x='plays',
                y='artist_name',
                color='cultural_class',
                title="Top 15 Artists by Culture",
                orientation='h',
                color_discrete_map={
                    'vietnamese': '#ff6b6b',
                    'western': '#4ecdc4'
                }
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        # Row 2: Cultural Evolution Timeline
        st.subheader("📈 Cultural Evolution Timeline")
        
        # Calculate cultural ratios by year and month
        cultural_timeline = filtered_data.groupby([
            filtered_data['played_at'].dt.to_period('M'), 
            'cultural_class'
        ]).size().unstack(fill_value=0)
        
        # Calculate percentages
        cultural_timeline_pct = cultural_timeline.div(cultural_timeline.sum(axis=1), axis=0) * 100
        
        # Create interactive timeline
        fig = go.Figure()
        
        for culture in cultural_timeline_pct.columns:
            fig.add_trace(go.Scatter(
                x=cultural_timeline_pct.index.astype(str),
                y=cultural_timeline_pct[culture],
                mode='lines+markers',
                name=culture.title(),
                line=dict(width=3),
                marker=dict(size=6)
            ))
        
        fig.update_layout(
            title="Cultural Music Evolution Over Time",
            xaxis_title="Time Period",
            yaxis_title="Percentage of Listening",
            height=500,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Row 3: Playlist Analysis (if available)
        if not playlist_data.empty:
            st.subheader("📝 Playlist Intelligence")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Playlist categories
                playlist_categories = {
                    'Mood - Positive': ['gleeful', 'something_cute_and_dynamic', 'sheer_love'],
                    'Mood - Reflective': ['heartbreaking', 'such_a_bad_day', 'tempalative_mood'],
                    'Mood - Chill': ['dreamy', 'chillie'],
                    'Cultural - Vietnamese': ['vpop'],
                    'Cultural - Western': ['us-uk'],
                    'Memory & Lyrics': ['memory_brings_back', 'lyrics_nail_ur_heart'],
                    'Daily Favorites': ['daily_music', 'on_repeat', 'liked_songs']
                }
                
                category_stats = []
                for category, playlists in playlist_categories.items():
                    matching_playlists = playlist_data[playlist_data['playlist_name'].isin(playlists)]
                    if not matching_playlists.empty:
                        category_stats.append({
                            'category': category,
                            'tracks': len(matching_playlists),
                            'unique_artists': matching_playlists['Artist Name(s)'].nunique()
                        })
                
                if category_stats:
                    category_df = pd.DataFrame(category_stats)
                    
                    fig = px.bar(
                        category_df,
                        x='tracks',
                        y='category',
                        title="Playlist Categories Analysis",
                        orientation='h',
                        color='tracks',
                        color_continuous_scale='viridis'
                    )
                    fig.update_layout(showlegend=False, height=400)
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Playlist creation timeline
                if 'Added At' in playlist_data.columns:
                    playlist_timeline = playlist_data.groupby(
                        playlist_data['Added At'].dt.to_period('M')
                    ).size().reset_index(name='tracks_added')
                    playlist_timeline['period'] = playlist_timeline['Added At'].astype(str)
                    
                    fig = px.line(
                        playlist_timeline,
                        x='period',
                        y='tracks_added',
                        title="Playlist Curation Timeline",
                        markers=True
                    )
                    fig.update_xaxes(tickangle=45)
                    st.plotly_chart(fig, use_container_width=True)
        
        # Artist Deep Dive (if specific artists selected)
        if selected_artists:
            st.subheader("🔍 Artist Deep Dive")
            
            for artist in selected_artists[:3]:  # Limit to top 3 for performance
                artist_data = filtered_data[filtered_data['artist_name'] == artist]
                
                with st.expander(f"🎤 {artist} Analysis ({len(artist_data)} plays)"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        # Hourly pattern for this artist
                        hourly_pattern = artist_data.groupby('hour').size()
                        peak_hour = hourly_pattern.idxmax()
                        st.metric("Peak Hour", f"{peak_hour}:00")
                    
                    with col2:
                        # Favorite day
                        daily_pattern = artist_data.groupby('day_of_week').size()
                        favorite_day = daily_pattern.idxmax()
                        st.metric("Favorite Day", favorite_day)
                    
                    with col3:
                        # Cultural classification
                        culture = artist_data['cultural_class'].iloc[0]
                        st.metric("Culture", culture.title())
                    
                    # Timeline for this artist
                    artist_timeline = artist_data.groupby(
                        artist_data['played_at'].dt.to_period('M')
                    ).size().reset_index(name='plays')
                    artist_timeline['period'] = artist_timeline['played_at'].astype(str)
                    
                    fig = px.line(
                        artist_timeline,
                        x='period',
                        y='plays',
                        title=f"{artist} - Listening Timeline",
                        markers=True
                    )
                    fig.update_xaxes(tickangle=45)
                    st.plotly_chart(fig, use_container_width=True)
    
    def show_musical_timeline(self, data):
        """Interactive musical life timeline"""
        
        st.header("📈 Comprehensive Musical Life Timeline")
        
        # Timeline overview metrics
        total_days = (data['played_at'].max() - data['played_at'].min()).days
        total_hours = data['minutes_played'].sum() / 60
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Days", f"{total_days:,}")
        with col2:
            st.metric("Total Hours", f"{total_hours:.0f}")
        with col3:
            st.metric("Unique Tracks", f"{data['track_id'].nunique():,}")
        with col4:
            st.metric("Unique Artists", f"{data['artist_name'].nunique():,}")
        
        # Interactive timeline selector
        st.subheader("🕐 Timeline Explorer")
        
        timeline_view = st.selectbox(
            "Select Timeline View",
            ["Monthly Activity", "Yearly Overview", "Daily Intensity", "Artist Discovery Timeline"]
        )
        
        if timeline_view == "Monthly Activity":
            # Monthly activity timeline
            monthly_data = data.groupby(data['played_at'].dt.to_period('M')).agg({
                'track_id': 'count',
                'artist_name': 'nunique',
                'minutes_played': 'sum'
            }).reset_index()
            monthly_data['period'] = monthly_data['played_at'].astype(str)
            monthly_data['hours_listened'] = monthly_data['minutes_played'] / 60
            
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Monthly Plays', 'Monthly Listening Hours'),
                vertical_spacing=0.12
            )
            
            fig.add_trace(
                go.Scatter(
                    x=monthly_data['period'],
                    y=monthly_data['track_id'],
                    mode='lines+markers',
                    name='Plays',
                    line=dict(color='#1f77b4', width=3)
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=monthly_data['period'],
                    y=monthly_data['hours_listened'],
                    mode='lines+markers',
                    name='Hours',
                    line=dict(color='#ff7f0e', width=3)
                ),
                row=2, col=1
            )
            
            fig.update_layout(height=600, showlegend=False)
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
            
            # Activity peaks
            peak_month = monthly_data.loc[monthly_data['track_id'].idxmax()]
            st.markdown(f"""
            <div class="insight-box">
                <strong>📊 Peak Activity:</strong> {peak_month['period']} 
                ({peak_month['track_id']:,} plays, {peak_month['hours_listened']:.0f} hours)
            </div>
            """, unsafe_allow_html=True)
        
        elif timeline_view == "Yearly Overview":
            # Yearly comparison
            yearly_data = data.groupby('year').agg({
                'track_id': 'count',
                'artist_name': 'nunique',
                'minutes_played': 'sum'
            }).reset_index()
            yearly_data['hours_listened'] = yearly_data['minutes_played'] / 60
            yearly_data['daily_avg'] = yearly_data['track_id'] / 365
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(
                    yearly_data,
                    x='year',
                    y='track_id',
                    title="Annual Listening Activity",
                    color='track_id',
                    color_continuous_scale='viridis'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.bar(
                    yearly_data,
                    x='year',
                    y='artist_name',
                    title="Artist Discovery by Year",
                    color='artist_name',
                    color_continuous_scale='plasma'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Year comparison table
            st.subheader("📊 Year-by-Year Comparison")
            yearly_display = yearly_data[['year', 'track_id', 'artist_name', 'hours_listened', 'daily_avg']].copy()
            yearly_display.columns = ['Year', 'Total Plays', 'Unique Artists', 'Hours Listened', 'Daily Average']
            yearly_display['Hours Listened'] = yearly_display['Hours Listened'].round(0)
            yearly_display['Daily Average'] = yearly_display['Daily Average'].round(1)
            
            st.dataframe(yearly_display, use_container_width=True)
        
        elif timeline_view == "Daily Intensity":
            # Daily listening intensity heatmap
            data['date'] = data['played_at'].dt.date
            daily_intensity = data.groupby('date').size().reset_index(name='plays')
            daily_intensity['date'] = pd.to_datetime(daily_intensity['date'])
            daily_intensity['year'] = daily_intensity['date'].dt.year
            daily_intensity['month'] = daily_intensity['date'].dt.month
            daily_intensity['day'] = daily_intensity['date'].dt.day
            
            # Create calendar heatmap
            pivot_calendar = daily_intensity.pivot_table(
                values='plays', 
                index='month', 
                columns='day', 
                aggfunc='mean'
            ).fillna(0)
            
            fig = px.imshow(
                pivot_calendar.values,
                labels=dict(x="Day of Month", y="Month", color="Avg Daily Plays"),
                x=list(range(1, 32)),
                y=list(range(1, 13)),
                color_continuous_scale='YlOrRd',
                title="Daily Listening Intensity Calendar View"
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        elif timeline_view == "Artist Discovery Timeline":
            # Artist discovery over time
            st.subheader("🎤 Artist Discovery Journey")
            
            # Get first play date for each artist
            artist_discovery = data.groupby('artist_name')['played_at'].min().reset_index()
            artist_discovery['discovery_month'] = artist_discovery['played_at'].dt.to_period('M')
            
            # Count new artists discovered each month
            monthly_discoveries = artist_discovery.groupby('discovery_month').size().reset_index(name='new_artists')
            monthly_discoveries['period'] = monthly_discoveries['discovery_month'].astype(str)
            
            fig = px.line(
                monthly_discoveries,
                x='period',
                y='new_artists',
                title="New Artist Discoveries Over Time",
                markers=True
            )
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
            
            # Top discovery months
            top_discovery_months = monthly_discoveries.nlargest(5, 'new_artists')
            
            st.subheader("🏆 Top Discovery Months")
            for _, month_data in top_discovery_months.iterrows():
                period = month_data['period']
                new_artists = month_data['new_artists']
                
                # Get artists discovered in this month
                month_artists = artist_discovery[
                    artist_discovery['discovery_month'].astype(str) == period
                ]['artist_name'].tolist()
                
                st.markdown(f"""
                <div class="timeline-event">
                    <strong>{period}</strong>: {new_artists} new artists discovered<br/>
                    <em>Top discoveries: {', '.join(month_artists[:5])}</em>
                </div>
                """, unsafe_allow_html=True)

def main():
    """Main Streamlit application"""
    
    st.title("🕐 Interactive Temporal Music Dashboard")
    st.markdown("**Comprehensive exploration of 4+ years of listening patterns**")
    st.markdown("71,051 streaming records + 634 playlist tracks analyzed interactively")
    
    # Initialize dashboard
    dashboard = InteractiveTemporalDashboard()
    
    # Load data
    with st.spinner("Loading comprehensive music data..."):
        streaming_data, playlist_data = dashboard.load_data()
    
    if streaming_data is None:
        st.error("Failed to load data. Please check file paths.")
        return
    
    # Process temporal data
    streaming_data = dashboard.process_temporal_data(streaming_data)
    
    st.success(f"✅ Loaded {len(streaming_data):,} streaming records and {len(playlist_data):,} playlist tracks")
    
    # Sidebar navigation
    st.sidebar.title("🎛️ Dashboard Navigation")
    
    page = st.sidebar.selectbox(
        "Select Analysis View",
        ["🕐 Temporal Patterns", "🎵 Genre Intelligence", "📈 Musical Timeline", "📊 Complete Overview"]
    )
    
    # Show selected analysis
    if page == "🕐 Temporal Patterns":
        dashboard.show_temporal_patterns_dashboard(streaming_data)
    elif page == "🎵 Genre Intelligence":
        dashboard.show_genre_intelligence_explorer(streaming_data, playlist_data)
    elif page == "📈 Musical Timeline":
        dashboard.show_musical_timeline(streaming_data)
    elif page == "📊 Complete Overview":
        # Show all analyses
        dashboard.show_temporal_patterns_dashboard(streaming_data)
        st.markdown("---")
        dashboard.show_genre_intelligence_explorer(streaming_data, playlist_data)
        st.markdown("---")
        dashboard.show_musical_timeline(streaming_data)
    
    # Footer
    st.markdown("---")
    st.markdown("*Interactive Temporal Music Dashboard - Powered by Streamlit & Plotly*")

if __name__ == "__main__":
    main()
