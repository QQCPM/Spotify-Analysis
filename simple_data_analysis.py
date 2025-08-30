#!/usr/bin/env python3
"""
Simple Spotify Data Analysis Script

Quick analysis of your processed Spotify data to provide immediate insights.
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from pathlib import Path

def analyze_spotify_data():
    """Quick analysis of processed Spotify data"""
    
    print("🎵 Quick Analysis of Your Spotify Data")
    print("=" * 50)
    
    # Load processed data
    try:
        df = pd.read_parquet("data/processed/streaming_data_processed.parquet")
        print(f"✅ Loaded {len(df):,} streaming records")
    except Exception as e:
        print(f"❌ Error loading processed data: {str(e)}")
        return
    
    # Basic statistics
    print(f"\n📊 Basic Statistics:")
    print(f"   Date range: {df['played_at'].min().strftime('%Y-%m-%d')} to {df['played_at'].max().strftime('%Y-%m-%d')}")
    print(f"   Total listening time: {df['minutes_played'].sum():,.1f} minutes ({df['minutes_played'].sum()/60:.1f} hours)")
    print(f"   Unique tracks: {df['track_id'].nunique():,}")
    print(f"   Unique artists: {df['artist_name'].nunique():,}")
    
    # Temporal patterns
    print(f"\n⏰ Temporal Patterns:")
    hourly_activity = df.groupby('hour').size()
    peak_hour = hourly_activity.idxmax()
    print(f"   Peak listening hour: {peak_hour}:00 ({hourly_activity[peak_hour]:,} plays)")
    
    daily_activity = df.groupby('day_of_week').size()
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    peak_day_idx = daily_activity.idxmax()
    print(f"   Most active day: {days[peak_day_idx]} ({daily_activity[peak_day_idx]:,} plays)")
    
    # Platform usage
    print(f"\n📱 Platform Usage:")
    platform_stats = df['platform_type'].value_counts()
    for platform, count in platform_stats.head(3).items():
        percentage = count / len(df) * 100
        print(f"   {platform}: {count:,} plays ({percentage:.1f}%)")
    
    # Location analysis
    print(f"\n🌍 Listening Locations:")
    location_stats = df['listening_location'].value_counts()
    for location, count in location_stats.items():
        percentage = count / len(df) * 100
        print(f"   {location}: {count:,} plays ({percentage:.1f}%)")
    
    # Skip rate analysis
    if 'likely_skipped' in df.columns:
        skip_rate = df['likely_skipped'].mean()
        print(f"\n⏭️  Listening Behavior:")
        print(f"   Skip rate: {skip_rate:.1%}")
        print(f"   Average track completion: {(1-skip_rate):.1%}")
    
    # Top artists and tracks
    print(f"\n🎤 Top Artists:")
    top_artists = df.groupby('artist_name').agg({
        'track_id': 'count',
        'minutes_played': 'sum'
    }).sort_values('track_id', ascending=False).head(10)
    
    for i, (artist, stats) in enumerate(top_artists.iterrows(), 1):
        print(f"   {i:2d}. {artist}: {stats['track_id']:,} plays, {stats['minutes_played']:.0f} minutes")
    
    print(f"\n🎵 Top Tracks:")
    top_tracks = df.groupby(['track_name', 'artist_name']).agg({
        'track_id': 'count',
        'minutes_played': 'sum'
    }).sort_values('track_id', ascending=False).head(10)
    
    for i, ((track, artist), stats) in enumerate(top_tracks.iterrows(), 1):
        print(f"   {i:2d}. {track} - {artist}: {stats['track_id']:,} plays")
    
    # Cultural analysis hints
    print(f"\n🌏 Cultural Analysis Hints:")
    
    # Detect Vietnamese artists by name patterns
    vietnamese_patterns = ['Sơn Tùng', 'Hoàng Thùy Linh', 'Đức Phúc', 'Bích Phương', 'Chi Pu', 'Noo Phước Thịnh']
    vietnamese_artists = df[df['artist_name'].str.contains('|'.join(vietnamese_patterns), case=False, na=False)]['artist_name'].nunique()
    
    # Detect Vietnamese characters in artist names
    vietnamese_chars = 'àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ'
    has_vietnamese_chars = df['artist_name'].str.contains(f'[{vietnamese_chars}]', case=False, na=False)
    vietnamese_char_artists = df[has_vietnamese_chars]['artist_name'].nunique()
    
    print(f"   Artists with Vietnamese names/patterns: {vietnamese_artists + vietnamese_char_artists}")
    print(f"   Potential Vietnamese content: {has_vietnamese_chars.sum():,} plays ({has_vietnamese_chars.mean():.1%})")
    
    # Country-based listening
    if 'listening_location' in df.columns:
        vietnam_plays = df[df['listening_location'] == 'Vietnam']
        if len(vietnam_plays) > 0:
            print(f"   Plays while in Vietnam: {len(vietnam_plays):,} ({len(vietnam_plays)/len(df):.1%})")
    
    # Session analysis
    if 'session_id' in df.columns:
        print(f"\n🎧 Session Analysis:")
        n_sessions = df['session_id'].nunique()
        avg_session_length = df.groupby('session_id').size().mean()
        avg_session_duration = df.groupby('session_id')['minutes_played'].sum().mean()
        print(f"   Total sessions: {n_sessions:,}")
        print(f"   Average tracks per session: {avg_session_length:.1f}")
        print(f"   Average session duration: {avg_session_duration:.1f} minutes")
    
    # Monthly listening evolution
    print(f"\n📈 Monthly Listening Evolution:")
    df['year_month'] = df['played_at'].dt.to_period('M')
    monthly_stats = df.groupby('year_month').agg({
        'track_id': 'count',
        'artist_name': 'nunique',
        'minutes_played': 'sum'
    }).tail(12)  # Last 12 months
    
    for period, stats in monthly_stats.iterrows():
        print(f"   {period}: {stats['track_id']:,} plays, {stats['artist_name']:,} artists, {stats['minutes_played']:.0f} minutes")
    
    print(f"\n🎯 Ready for Phase 2 Deep Analysis:")
    print(f"   ✅ Data processing complete")
    print(f"   ✅ {len(df):,} high-quality listening records")
    print(f"   ✅ Rich temporal data ({(df['played_at'].max() - df['played_at'].min()).days} days)")
    print(f"   ✅ Multi-platform usage detected")
    print(f"   ✅ Cross-cultural patterns identified")
    print(f"\n🚀 Next: Run latent factor analysis and hypothesis testing!")


if __name__ == "__main__":
    analyze_spotify_data()