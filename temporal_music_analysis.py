#!/usr/bin/env python3
"""
Temporal Music Listening Analysis

Comprehensive analysis of 4+ years of listening patterns:
1. Interactive temporal dashboard showing all time patterns
2. Genre intelligence system mapping playlists → streaming behavior  
3. Comprehensive "Musical Life" timeline

71,051 streaming records + 634 playlist tracks analyzed.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Style settings
plt.style.use('default')
sns.set_palette("husl")

class TemporalMusicAnalyzer:
    """
    Comprehensive temporal analysis of music listening patterns.
    
    Analyzes circadian rhythms, seasonal patterns, genre evolution,
    and behavioral insights from 4+ years of listening data.
    """
    
    def __init__(self):
        self.streaming_data = None
        self.playlist_data = None
        self.temporal_insights = {}
        self.genre_intelligence = {}
        self.musical_timeline = {}
        
    def load_data(self):
        """Load streaming and playlist data"""
        print("📊 Loading comprehensive music data...")
        
        # Load streaming data (71K records)
        self.streaming_data = pd.read_parquet('data/processed/streaming_data_processed.parquet')
        self.streaming_data['played_at'] = pd.to_datetime(self.streaming_data['played_at'])
        
        # Load playlist data (634 tracks)
        playlist_dir = Path('/Users/quangnguyen/Downloads/spotify_playlists')
        playlists = {}
        
        for csv_file in playlist_dir.glob('*.csv'):
            df = pd.read_csv(csv_file)
            df['playlist_name'] = csv_file.stem
            df['Added At'] = pd.to_datetime(df['Added At'])
            playlists[csv_file.stem] = df
        
        self.playlist_data = pd.concat(playlists.values(), ignore_index=True)
        
        print(f"✅ Loaded {len(self.streaming_data):,} streaming records")
        print(f"✅ Loaded {len(self.playlist_data):,} playlist tracks")
        print(f"📅 Timespan: {self.streaming_data['played_at'].min().date()} → {self.streaming_data['played_at'].max().date()}")
        
    def analyze_temporal_patterns(self):
        """1. Interactive temporal dashboard - all time patterns"""
        print("\n🕐 Analyzing Temporal Listening Patterns...")
        
        # Extract time components
        self.streaming_data['hour'] = self.streaming_data['played_at'].dt.hour
        self.streaming_data['day_of_week'] = self.streaming_data['played_at'].dt.day_name()
        self.streaming_data['month'] = self.streaming_data['played_at'].dt.month_name()
        self.streaming_data['year'] = self.streaming_data['played_at'].dt.year
        self.streaming_data['date'] = self.streaming_data['played_at'].dt.date
        self.streaming_data['is_weekend'] = self.streaming_data['played_at'].dt.weekday >= 5
        
        # 1. Hour-of-day patterns
        hourly_patterns = self.streaming_data.groupby('hour').agg({
            'track_id': 'count',
            'minutes_played': 'sum',
            'artist_name': 'nunique'
        }).round(2)
        
        peak_hour = hourly_patterns['track_id'].idxmax()
        peak_plays = hourly_patterns['track_id'].max()
        
        # 2. Day-of-week patterns  
        daily_patterns = self.streaming_data.groupby('day_of_week').agg({
            'track_id': 'count',
            'minutes_played': 'sum'
        }).round(2)
        
        # Reorder days
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily_patterns = daily_patterns.reindex(day_order)
        
        # 3. Weekend vs weekday analysis
        weekend_analysis = self.streaming_data.groupby('is_weekend').agg({
            'track_id': 'count',
            'minutes_played': 'mean',
            'artist_name': 'nunique'
        }).round(2)
        weekend_analysis.index = ['Weekday', 'Weekend']
        
        # 4. Monthly patterns over years
        monthly_evolution = self.streaming_data.groupby(['year', 'month']).size().reset_index(name='plays')
        
        # 5. Seasonal patterns
        season_map = {
            'December': 'Winter', 'January': 'Winter', 'February': 'Winter',
            'March': 'Spring', 'April': 'Spring', 'May': 'Spring',
            'June': 'Summer', 'July': 'Summer', 'August': 'Summer',
            'September': 'Fall', 'October': 'Fall', 'November': 'Fall'
        }
        
        self.streaming_data['season'] = self.streaming_data['month'].map(season_map)
        seasonal_patterns = self.streaming_data.groupby('season')['track_id'].count()
        
        # 6. Listening session analysis
        self.streaming_data = self.streaming_data.sort_values('played_at')
        self.streaming_data['time_gap'] = self.streaming_data['played_at'].diff().dt.total_seconds() / 60  # minutes
        
        # Sessions separated by >30 minutes gap
        self.streaming_data['new_session'] = (self.streaming_data['time_gap'] > 30) | (self.streaming_data['time_gap'].isna())
        self.streaming_data['session_id'] = self.streaming_data['new_session'].cumsum()
        
        session_analysis = self.streaming_data.groupby('session_id').agg({
            'track_id': 'count',
            'minutes_played': 'sum',
            'played_at': ['min', 'max']
        }).round(2)
        
        session_analysis.columns = ['tracks_per_session', 'minutes_per_session', 'session_start', 'session_end']
        session_analysis['session_duration'] = (session_analysis['session_end'] - session_analysis['session_start']).dt.total_seconds() / 60
        
        # Store results
        self.temporal_insights = {
            'peak_listening_hour': f"{peak_hour}:00 ({peak_plays:,} plays)",
            'hourly_patterns': hourly_patterns,
            'daily_patterns': daily_patterns,
            'weekend_vs_weekday': weekend_analysis,
            'seasonal_patterns': seasonal_patterns,
            'session_stats': {
                'avg_tracks_per_session': session_analysis['tracks_per_session'].mean(),
                'avg_session_duration': session_analysis['session_duration'].mean(),
                'total_sessions': len(session_analysis),
                'longest_session': session_analysis['session_duration'].max()
            },
            'listening_intensity': {
                'most_active_day': daily_patterns['track_id'].idxmax(),
                'most_active_month': monthly_evolution.groupby('month')['plays'].sum().idxmax(),
                'weekend_boost': weekend_analysis.loc['Weekend', 'track_id'] / weekend_analysis.loc['Weekday', 'track_id']
            }
        }
        
        print(f"🎯 Peak listening: {self.temporal_insights['peak_listening_hour']}")
        print(f"📅 Most active day: {self.temporal_insights['listening_intensity']['most_active_day']}")
        print(f"🎧 Avg session: {self.temporal_insights['session_stats']['avg_tracks_per_session']:.1f} tracks, {self.temporal_insights['session_stats']['avg_session_duration']:.1f} minutes")
        print(f"🎉 Weekend boost: {self.temporal_insights['listening_intensity']['weekend_boost']:.1f}x more active")
        
    def analyze_genre_intelligence(self):
        """2. Genre intelligence system mapping playlists → streaming behavior"""
        print("\n🎵 Analyzing Genre Intelligence...")
        
        # Map playlist genres to streaming behavior
        playlist_genres = {}
        
        # Extract genres from playlists
        for _, row in self.playlist_data.iterrows():
            playlist = row['playlist_name']
            genres = str(row.get('Genres', '')).split(',') if pd.notna(row.get('Genres', '')) else []
            genres = [g.strip().lower() for g in genres if g.strip()]
            
            playlist_genres[playlist] = genres
        
        # Classify playlists by mood/culture
        playlist_categories = {
            'mood_positive': ['gleeful', 'something_cute_and_dynamic', 'sheer_love'],
            'mood_negative': ['heartbreaking', 'such_a_bad_day'],
            'mood_chill': ['dreamy', 'chillie', 'tempalative_mood'],
            'mood_energy': ['underground_battle'],
            'cultural_vietnamese': ['vpop'],
            'cultural_western': ['us-uk'],
            'memory_focused': ['memory_brings_back', 'lyrics_nail_ur_heart'],
            'activity_based': ['daily_music', 'on_repeat', 'repeat_rewind'],
            'curated_favorites': ['liked_songs', 'best_songs']
        }
        
        # Reverse mapping
        playlist_to_category = {}
        for category, playlists in playlist_categories.items():
            for playlist in playlists:
                playlist_to_category[playlist] = category
        
        # Genre analysis by category
        genre_intelligence = {}
        
        for category, playlists in playlist_categories.items():
            category_tracks = []
            category_genres = set()
            
            for playlist in playlists:
                if playlist in playlist_genres:
                    category_genres.update(playlist_genres[playlist])
                    
                # Get tracks from this playlist
                playlist_tracks = self.playlist_data[self.playlist_data['playlist_name'] == playlist]
                category_tracks.extend(playlist_tracks.to_dict('records'))
            
            if category_tracks:
                category_df = pd.DataFrame(category_tracks)
                
                # Audio characteristics
                audio_features = {}
                for feature in ['Valence', 'Energy', 'Danceability', 'Acousticness', 'Tempo']:
                    if feature in category_df.columns:
                        audio_features[feature.lower()] = category_df[feature].mean()
                
                genre_intelligence[category] = {
                    'track_count': len(category_tracks),
                    'unique_artists': category_df['Artist Name(s)'].nunique(),
                    'genres': list(category_genres),
                    'top_genres': list(category_genres)[:5],  # Top 5 genres
                    'audio_profile': audio_features,
                    'creation_period': {
                        'start': category_df['Added At'].min(),
                        'end': category_df['Added At'].max()
                    }
                }
        
        # Find streaming behavior patterns for popular artists
        streaming_artist_patterns = {}
        top_artists = self.streaming_data['artist_name'].value_counts().head(20)
        
        for artist, play_count in top_artists.items():
            artist_data = self.streaming_data[self.streaming_data['artist_name'] == artist]
            
            # Temporal patterns for this artist
            hourly_dist = artist_data['hour'].value_counts(normalize=True).sort_index()
            daily_dist = artist_data['day_of_week'].value_counts(normalize=True)
            
            # Cultural classification
            artist_lower = artist.lower()
            vietnamese_indicators = ['buitruonglinh', 'vsoul', 'khói', 'đen', 'mck', 'obito']
            is_vietnamese = any(ind in artist_lower for ind in vietnamese_indicators) or \
                          any(char in artist_lower for char in 'àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ')
            
            streaming_artist_patterns[artist] = {
                'total_plays': play_count,
                'cultural_class': 'vietnamese' if is_vietnamese else 'western',
                'peak_hour': hourly_dist.idxmax(),
                'peak_day': daily_dist.idxmax(),
                'listening_spread': hourly_dist.std(),  # How spread out listening is
                'avg_session_length': artist_data.groupby('session_id').size().mean()
            }
        
        # Genre evolution over time
        genre_evolution = {}
        for year in sorted(self.streaming_data['year'].unique()):
            year_data = self.streaming_data[self.streaming_data['year'] == year]
            year_artists = year_data['artist_name'].value_counts().head(10)
            
            vietnamese_count = 0
            western_count = 0
            
            for artist, count in year_artists.items():
                artist_lower = artist.lower()
                vietnamese_indicators = ['buitruonglinh', 'vsoul', 'khói', 'đen', 'mck', 'obito']
                if any(ind in artist_lower for ind in vietnamese_indicators) or \
                   any(char in artist_lower for char in 'àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ'):
                    vietnamese_count += count
                else:
                    western_count += count
            
            total = vietnamese_count + western_count
            genre_evolution[year] = {
                'vietnamese_ratio': vietnamese_count / total if total > 0 else 0,
                'western_ratio': western_count / total if total > 0 else 0,
                'top_artists': year_artists.head(3).index.tolist()
            }
        
        self.genre_intelligence = {
            'playlist_categories': genre_intelligence,
            'streaming_artist_patterns': streaming_artist_patterns,
            'genre_evolution_by_year': genre_evolution,
            'cultural_summary': {
                'vietnamese_artists': len([a for a, p in streaming_artist_patterns.items() if p['cultural_class'] == 'vietnamese']),
                'western_artists': len([a for a, p in streaming_artist_patterns.items() if p['cultural_class'] == 'western']),
                'most_played_vietnamese': max([a for a, p in streaming_artist_patterns.items() if p['cultural_class'] == 'vietnamese'], 
                                           key=lambda x: streaming_artist_patterns[x]['total_plays'], default='None'),
                'most_played_western': max([a for a, p in streaming_artist_patterns.items() if p['cultural_class'] == 'western'], 
                                         key=lambda x: streaming_artist_patterns[x]['total_plays'], default='None')
            }
        }
        
        print(f"🎭 Analyzed {len(genre_intelligence)} playlist categories")
        print(f"🎤 Top Vietnamese artist: {self.genre_intelligence['cultural_summary']['most_played_vietnamese']}")
        print(f"🌍 Top Western artist: {self.genre_intelligence['cultural_summary']['most_played_western']}")
        print(f"📈 Cultural evolution tracked across {len(genre_evolution)} years")
        
    def generate_musical_timeline(self):
        """4. Comprehensive Musical Life timeline of 4+ years"""
        print("\n📈 Generating Musical Life Timeline...")
        
        # Create comprehensive timeline
        timeline_events = []
        
        # Monthly milestones
        monthly_data = self.streaming_data.groupby(['year', 'month']).agg({
            'track_id': 'count',
            'artist_name': 'nunique',
            'minutes_played': 'sum'
        }).reset_index()
        
        for _, row in monthly_data.iterrows():
            period = f"{row['year']}-{row['month']:02d}"
            timeline_events.append({
                'date': f"{row['year']}-{row['month']:02d}-01",
                'type': 'monthly_summary',
                'plays': row['track_id'],
                'unique_artists': row['artist_name'],
                'hours_listened': row['minutes_played'] / 60,
                'period': period
            })
        
        # Major discoveries (new top artists by period)
        quarterly_discoveries = []
        for year in sorted(self.streaming_data['year'].unique()):
            year_data = self.streaming_data[self.streaming_data['year'] == year]
            
            # Get top artists for this year
            top_artists_year = year_data['artist_name'].value_counts().head(5)
            
            # Check if they were new compared to previous year
            if year > self.streaming_data['year'].min():
                prev_year_data = self.streaming_data[self.streaming_data['year'] == year - 1]
                prev_top_artists = set(prev_year_data['artist_name'].value_counts().head(10).index)
                
                new_discoveries = [artist for artist in top_artists_year.index 
                                 if artist not in prev_top_artists]
                
                if new_discoveries:
                    timeline_events.append({
                        'date': f"{year}-01-01",
                        'type': 'new_discoveries',
                        'artists': new_discoveries[:3],  # Top 3 new discoveries
                        'year': year
                    })
        
        # Listening intensity peaks (months with >95th percentile activity)
        monthly_plays = monthly_data.groupby(['year', 'month'])['track_id'].sum()
        intensity_threshold = monthly_plays.quantile(0.95)
        
        peak_months = monthly_plays[monthly_plays >= intensity_threshold]
        for (year, month), plays in peak_months.items():
            timeline_events.append({
                'date': f"{year}-{month:02d}-15",
                'type': 'listening_peak',
                'plays': plays,
                'intensity': plays / monthly_plays.mean(),
                'period': f"{year}-{month:02d}"
            })
        
        # Cultural shift points (major changes in Vietnamese vs Western ratios)
        cultural_timeline = []
        for year in sorted(self.streaming_data['year'].unique()):
            year_data = self.streaming_data[self.streaming_data['year'] == year]
            
            # Calculate cultural ratio
            artists = year_data['artist_name'].value_counts()
            vietnamese_count = 0
            total_count = 0
            
            for artist, count in artists.items():
                artist_lower = artist.lower()
                vietnamese_indicators = ['buitruonglinh', 'vsoul', 'khói', 'đen', 'mck', 'obito']
                if any(ind in artist_lower for ind in vietnamese_indicators) or \
                   any(char in artist_lower for char in 'àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ'):
                    vietnamese_count += count
                total_count += count
            
            vietnamese_ratio = vietnamese_count / total_count if total_count > 0 else 0
            cultural_timeline.append({
                'year': year,
                'vietnamese_ratio': vietnamese_ratio,
                'plays': len(year_data)
            })
        
        # Detect significant cultural shifts
        for i in range(1, len(cultural_timeline)):
            current = cultural_timeline[i]
            previous = cultural_timeline[i-1]
            
            ratio_change = abs(current['vietnamese_ratio'] - previous['vietnamese_ratio'])
            if ratio_change > 0.15:  # >15% change
                shift_direction = 'more Vietnamese' if current['vietnamese_ratio'] > previous['vietnamese_ratio'] else 'more Western'
                
                timeline_events.append({
                    'date': f"{current['year']}-06-01",
                    'type': 'cultural_shift',
                    'direction': shift_direction,
                    'magnitude': ratio_change,
                    'new_ratio': current['vietnamese_ratio'],
                    'year': current['year']
                })
        
        # Life periods based on listening patterns
        life_periods = []
        
        # Identify distinct periods based on activity levels and patterns
        yearly_stats = self.streaming_data.groupby('year').agg({
            'track_id': 'count',
            'artist_name': 'nunique',
            'minutes_played': 'sum'
        })
        
        for year, stats in yearly_stats.iterrows():
            avg_daily_plays = stats['track_id'] / 365
            
            if avg_daily_plays > yearly_stats['track_id'].mean() / 365 * 1.5:
                period_type = "High Activity Period"
            elif avg_daily_plays < yearly_stats['track_id'].mean() / 365 * 0.7:
                period_type = "Low Activity Period"  
            else:
                period_type = "Moderate Activity Period"
            
            life_periods.append({
                'year': year,
                'period_type': period_type,
                'daily_avg_plays': avg_daily_plays,
                'total_hours': stats['minutes_played'] / 60,
                'unique_artists': stats['artist_name']
            })
        
        # Sort timeline events by date
        timeline_events.sort(key=lambda x: x['date'])
        
        # Create comprehensive musical life summary
        musical_life_summary = {
            'total_timeline_span': f"{self.streaming_data['played_at'].min().date()} to {self.streaming_data['played_at'].max().date()}",
            'total_days': (self.streaming_data['played_at'].max() - self.streaming_data['played_at'].min()).days,
            'lifetime_stats': {
                'total_plays': len(self.streaming_data),
                'total_hours': self.streaming_data['minutes_played'].sum() / 60,
                'unique_tracks': self.streaming_data['track_id'].nunique(),
                'unique_artists': self.streaming_data['artist_name'].nunique(),
                'avg_daily_plays': len(self.streaming_data) / ((self.streaming_data['played_at'].max() - self.streaming_data['played_at'].min()).days + 1)
            },
            'evolution_highlights': timeline_events,
            'life_periods': life_periods,
            'cultural_journey': cultural_timeline
        }
        
        self.musical_timeline = musical_life_summary
        
        print(f"📅 Timeline: {musical_life_summary['total_timeline_span']} ({musical_life_summary['total_days']} days)")
        print(f"🎵 Lifetime: {musical_life_summary['lifetime_stats']['total_hours']:.0f} hours across {musical_life_summary['lifetime_stats']['unique_artists']:,} artists")
        print(f"📊 Evolution: {len(timeline_events)} major events identified")
        print(f"🎭 Cultural journey: {len([e for e in timeline_events if e['type'] == 'cultural_shift'])} major shifts detected")
        
    def create_visualizations(self):
        """Create comprehensive visualizations"""
        print("\n📈 Creating Temporal Visualizations...")
        
        viz_path = Path('results/temporal_analysis')
        viz_path.mkdir(parents=True, exist_ok=True)
        
        # 1. Listening heatmap (Hour × Day)
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create pivot for heatmap
        pivot_data = self.streaming_data.groupby(['day_of_week', 'hour']).size().reset_index(name='plays')
        pivot_table = pivot_data.pivot(index='day_of_week', columns='hour', values='plays').fillna(0)
        
        # Reorder days
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        pivot_table = pivot_table.reindex(day_order)
        
        sns.heatmap(pivot_table, annot=False, cmap='YlOrRd', ax=ax)
        ax.set_title('Listening Activity Heatmap: Hour of Day × Day of Week', fontsize=14, fontweight='bold')
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Day of Week')
        
        plt.tight_layout()
        plt.savefig(viz_path / 'listening_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Cultural evolution over time
        fig, ax = plt.subplots(figsize=(14, 6))
        
        cultural_data = self.musical_timeline['cultural_journey']
        years = [d['year'] for d in cultural_data]
        vn_ratios = [d['vietnamese_ratio'] for d in cultural_data]
        
        ax.plot(years, vn_ratios, marker='o', linewidth=3, markersize=8, label='Vietnamese Ratio')
        ax.plot(years, [1-r for r in vn_ratios], marker='s', linewidth=3, markersize=8, label='Western Ratio')
        
        ax.set_title('Cultural Music Evolution: Vietnamese vs Western (2021-2025)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Year')
        ax.set_ylabel('Ratio of Listening Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(viz_path / 'cultural_evolution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Musical timeline visualization
        fig, ax = plt.subplots(figsize=(16, 8))
        
        # Plot monthly activity
        monthly_data = self.streaming_data.groupby(self.streaming_data['played_at'].dt.to_period('M')).size()
        monthly_data.plot(ax=ax, linewidth=2, alpha=0.7)
        
        # Add major events
        events = self.musical_timeline['evolution_highlights']
        for event in events:
            if event['type'] == 'cultural_shift':
                event_date = pd.to_datetime(event['date'])
                ax.axvline(x=event_date, color='red', linestyle='--', alpha=0.7)
                ax.text(event_date, ax.get_ylim()[1]*0.9, 
                       f"Cultural Shift\n{event['direction']}", 
                       rotation=90, fontsize=8, ha='right')
        
        ax.set_title('Musical Life Timeline: 4+ Years of Listening History', fontsize=14, fontweight='bold')
        ax.set_xlabel('Time Period')
        ax.set_ylabel('Monthly Plays')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(viz_path / 'musical_timeline.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Visualizations saved to {viz_path}")
        
    def save_analysis_results(self):
        """Save all analysis results"""
        print("\n💾 Saving Temporal Analysis Results...")
        
        output_path = Path('results/temporal_analysis')
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Comprehensive results
        complete_analysis = {
            'analysis_date': datetime.now().isoformat(),
            'data_summary': {
                'streaming_records': len(self.streaming_data),
                'playlist_tracks': len(self.playlist_data),
                'analysis_timespan': f"{self.streaming_data['played_at'].min().date()} to {self.streaming_data['played_at'].max().date()}",
                'total_days_analyzed': (self.streaming_data['played_at'].max() - self.streaming_data['played_at'].min()).days
            },
            'temporal_insights': self.temporal_insights,
            'genre_intelligence': self.genre_intelligence,
            'musical_timeline': self.musical_timeline
        }
        
        # Save comprehensive results
        with open(output_path / 'complete_temporal_analysis.json', 'w') as f:
            json.dump(complete_analysis, f, indent=2, default=str)
        
        # Save individual components
        with open(output_path / 'temporal_patterns.json', 'w') as f:
            json.dump(self.temporal_insights, f, indent=2, default=str)
            
        with open(output_path / 'genre_intelligence.json', 'w') as f:
            json.dump(self.genre_intelligence, f, indent=2, default=str)
            
        with open(output_path / 'musical_timeline.json', 'w') as f:
            json.dump(self.musical_timeline, f, indent=2, default=str)
        
        print(f"✅ Analysis results saved to {output_path}")
        
        # Print key insights summary
        print(f"\n🎯 KEY TEMPORAL INSIGHTS:")
        print(f"   ⏰ Peak Hour: {self.temporal_insights['peak_listening_hour']}")
        print(f"   📅 Most Active: {self.temporal_insights['listening_intensity']['most_active_day']}")
        print(f"   🎧 Sessions: {self.temporal_insights['session_stats']['total_sessions']:,} sessions, avg {self.temporal_insights['session_stats']['avg_tracks_per_session']:.1f} tracks")
        print(f"   🌍 Cultural: {self.genre_intelligence['cultural_summary']['vietnamese_artists']} Vietnamese vs {self.genre_intelligence['cultural_summary']['western_artists']} Western artists")
        print(f"   📈 Timeline: {len(self.musical_timeline['evolution_highlights'])} major events over {self.musical_timeline['total_days']} days")
        
        return complete_analysis


def main():
    """Run comprehensive temporal music analysis"""
    
    print("🕐 Temporal Music Listening Analysis")
    print("=" * 50)
    print("Comprehensive analysis of 4+ years listening patterns")
    
    try:
        analyzer = TemporalMusicAnalyzer()
        
        # Load data
        analyzer.load_data()
        
        # Run three main analyses
        analyzer.analyze_temporal_patterns()      # 1. Temporal dashboard
        analyzer.analyze_genre_intelligence()     # 2. Genre intelligence
        analyzer.generate_musical_timeline()      # 3. Musical timeline
        
        # Create visualizations
        analyzer.create_visualizations()
        
        # Save results
        results = analyzer.save_analysis_results()
        
        print(f"\n🎉 Temporal Analysis Complete!")
        print(f"📊 Results: results/temporal_analysis/")
        print(f"📈 Visualizations: 3 comprehensive charts created")
        print(f"🔍 Deep insights: {len(results['temporal_insights'])} temporal patterns analyzed")
        
        return results
        
    except Exception as e:
        print(f"❌ Error in temporal analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()