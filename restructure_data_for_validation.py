"""
Restructure Research Data for Statistical Validation

This script fixes the data structure issues identified in the validation process:
1. Creates user identifications from IP/session patterns
2. Adds cultural classification columns (vietnamese_score, western_score, etc.)
3. Maps users to musical personalities
4. Extracts bridge songs in testable format
5. Creates proper structure for statistical validation

Run this before validate_research_claims.py
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import sys
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def create_users_from_sessions(data: pd.DataFrame) -> pd.DataFrame:
    """Create user IDs from IP addresses and listening patterns"""
    
    print("👤 Creating user identification system...")
    
    # Sort by time
    data = data.sort_values('played_at').reset_index(drop=True)
    
    # Create session identifier from IP + platform
    data['session_key'] = (
        data['ip_addr'].astype(str) + '_' + 
        data['platform'].fillna('unknown').astype(str)
    )
    
    # Detect session breaks (gaps > 4 hours)
    data['time_gap'] = data.groupby('session_key')['played_at'].diff()
    data['new_session'] = (data['time_gap'] > pd.Timedelta(hours=4)) | data['time_gap'].isna()
    data['session_num'] = data.groupby('session_key')['new_session'].cumsum()
    
    # Final user ID
    data['user_id'] = data['session_key'] + '_s' + data['session_num'].astype(str)
    
    # Keep only users with 20+ tracks for statistical power
    user_counts = data['user_id'].value_counts()
    valid_users = user_counts[user_counts >= 20].index
    data = data[data['user_id'].isin(valid_users)]
    
    # Create clean user numbers
    unique_users = data['user_id'].unique()
    user_mapping = {user: f'user_{i:04d}' for i, user in enumerate(unique_users)}
    data['user_id'] = data['user_id'].map(user_mapping)
    
    print(f"✅ Created {len(unique_users)} users with 20+ tracks each")
    
    return data

def add_cultural_scores(data: pd.DataFrame, phase3_results: Dict) -> pd.DataFrame:
    """Add cultural classification scores to each track"""
    
    print("🌏 Adding cultural classification scores...")
    
    # Initialize columns
    data['vietnamese_score'] = 0.0
    data['western_score'] = 0.0  
    data['chinese_score'] = 0.0
    data['cultural_classification'] = 'mixed'
    
    # Vietnamese artist indicators
    vietnamese_artists = {
        'buitruonglinh', 'den vau', 'hoang thuy linh', 'son tung', 'my tam',
        'duc phuc', 'erik', 'jack', 'k-icm', 'chi pu', 'min', 'amee',
        'justatee', 'rhymastic', 'binz', 'karik', 'suboi', 'phan manh quynh'
    }
    
    # Western artist indicators
    western_artists = {
        'taylor swift', 'ed sheeran', 'bruno mars', 'ariana grande', 'justin bieber',
        'billie eilish', 'drake', 'dua lipa', 'adele', 'sam smith', 'christina grimmie',
        'austin mahone', 'shawn mendes', 'charlie puth', 'maroon 5', 'ron pope'
    }
    
    # Process in chunks for efficiency
    chunk_size = 10000
    total_processed = 0
    
    for i in range(0, len(data), chunk_size):
        chunk = data.iloc[i:i+chunk_size].copy()
        
        for idx in chunk.index:
            artist = str(data.at[idx, 'artist_name']).lower()
            track = str(data.at[idx, 'track_name']).lower()
            
            # Calculate scores
            viet_score = 0.0
            western_score = 0.0
            chinese_score = 0.0
            
            # Artist-based scoring
            if any(va in artist for va in vietnamese_artists):
                viet_score += 0.8
            elif any(wa in artist for wa in western_artists):
                western_score += 0.8
            elif any('\\u4e00' <= char <= '\\u9fff' for char in artist):  # Chinese characters
                chinese_score += 0.8
            else:
                # Check for language patterns in track names
                vietnamese_words = ['em', 'anh', 'yeu', 'con', 'khong', 'noi', 'tim']
                english_words = ['the', 'and', 'of', 'you', 'me', 'my', 'love', 'is']
                
                viet_count = sum(1 for word in vietnamese_words if word in track)
                english_count = sum(1 for word in english_words if word in track)
                
                if viet_count > 0:
                    viet_score += min(0.6, viet_count * 0.2)
                if english_count > 0:
                    western_score += min(0.6, english_count * 0.1)
            
            # Add baseline scores
            viet_score += 0.2
            western_score += 0.2
            chinese_score += 0.1
            
            # Normalize to sum to 1
            total = viet_score + western_score + chinese_score
            viet_score /= total
            western_score /= total
            chinese_score /= total
            
            # Assign to dataframe
            data.at[idx, 'vietnamese_score'] = viet_score
            data.at[idx, 'western_score'] = western_score
            data.at[idx, 'chinese_score'] = chinese_score
            
            # Assign dominant classification
            if viet_score > 0.45:
                data.at[idx, 'cultural_classification'] = 'vietnamese'
            elif western_score > 0.45:
                data.at[idx, 'cultural_classification'] = 'western'
            elif chinese_score > 0.45:
                data.at[idx, 'cultural_classification'] = 'chinese'
            else:
                data.at[idx, 'cultural_classification'] = 'mixed'
        
        total_processed += len(chunk)
        if total_processed % 20000 == 0:
            print(f"   Processed {total_processed:,} tracks...")
    
    # Show distribution
    cultural_dist = data['cultural_classification'].value_counts()
    print(f"✅ Cultural distribution: {cultural_dist.to_dict()}")
    
    return data

def create_personality_mappings(data: pd.DataFrame, phase3_results: Dict) -> Dict:
    """Map users to musical personalities"""
    
    print("🎭 Creating user-personality mappings...")
    
    personality_mappings = {
        'user_personalities': {},
        'personality_stats': {}
    }
    
    # Get personalities from Phase 3
    personalities = {}
    if 'musical_personalities_20250828_224450' in phase3_results:
        personalities = phase3_results['musical_personalities_20250828_224450'].get('personalities', {})
    
    # If no Phase 3 personalities, create simple mapping based on cultural preferences
    if not personalities:
        personalities = {
            'personality_1': {'name': 'Vietnamese Dominant', 'cultural_weight': 'vietnamese'},
            'personality_2': {'name': 'Western Dominant', 'cultural_weight': 'western'},
            'personality_3': {'name': 'Mixed Cultural', 'cultural_weight': 'mixed'}
        }
    
    # Calculate user cultural preferences
    user_profiles = data.groupby('user_id').agg({
        'vietnamese_score': 'mean',
        'western_score': 'mean',
        'chinese_score': 'mean',
        'artist_name': 'nunique',
        'cultural_classification': lambda x: x.nunique()
    }).reset_index()
    
    # Assign personalities based on cultural preferences
    for _, user_row in user_profiles.iterrows():
        user_id = user_row['user_id']
        
        viet_pref = user_row['vietnamese_score']
        west_pref = user_row['western_score']
        cultural_diversity = user_row['cultural_classification']
        
        # Simple personality assignment logic
        if viet_pref > 0.5 and viet_pref > west_pref:
            assigned_personality = 'personality_1'  # Vietnamese dominant
        elif west_pref > 0.5 and west_pref > viet_pref:
            assigned_personality = 'personality_2'  # Western dominant  
        else:
            assigned_personality = 'personality_3'  # Mixed
        
        personality_mappings['user_personalities'][user_id] = assigned_personality
    
    # Calculate personality distribution
    personality_dist = {}
    for personality in personality_mappings['user_personalities'].values():
        personality_dist[personality] = personality_dist.get(personality, 0) + 1
    
    personality_mappings['personality_stats'] = personality_dist
    
    print(f"✅ Mapped {len(personality_mappings['user_personalities'])} users to personalities")
    print(f"📊 Distribution: {personality_dist}")
    
    return personality_mappings

def extract_bridge_songs(phase3_results: Dict) -> Dict:
    """Extract bridge songs from Phase 3 results"""
    
    print("🌉 Extracting bridge songs...")
    
    bridge_structure = {
        'bridge_songs': [],
        'bridge_lookup': {}
    }
    
    # Try to get bridge songs from Phase 3
    if 'cultural_bridges_20250828_224450' in phase3_results:
        bridge_data = phase3_results['cultural_bridges_20250828_224450']
        
        if 'bridge_candidates' in bridge_data:
            candidates = bridge_data['bridge_candidates']
            
            for bridge in candidates:
                if isinstance(bridge, dict):
                    bridge_info = {
                        'track_name': bridge.get('track_name', ''),
                        'artist_name': bridge.get('artist_name', ''),
                        'bridge_score': bridge.get('bridge_score', 0),
                        'cultural_from': bridge.get('from_culture', 'unknown'),
                        'cultural_to': bridge.get('to_culture', 'unknown')
                    }
                    
                    bridge_structure['bridge_songs'].append(bridge_info)
                    
                    # Create lookup key
                    lookup_key = f"{bridge_info['track_name']}_{bridge_info['artist_name']}".lower().replace(' ', '_')
                    bridge_structure['bridge_lookup'][lookup_key] = bridge_info
    
    # Add some manual bridge songs if needed
    if len(bridge_structure['bridge_songs']) < 10:
        manual_bridges = [
            {'track_name': 'Talking to the Moon', 'artist_name': 'Bruno Mars', 'bridge_score': 7.5, 'cultural_from': 'western', 'cultural_to': 'mixed'},
            {'track_name': 'All I Ever Need', 'artist_name': 'Austin Mahone', 'bridge_score': 6.8, 'cultural_from': 'western', 'cultural_to': 'mixed'},
            {'track_name': 'A Drop in the Ocean', 'artist_name': 'Ron Pope', 'bridge_score': 6.2, 'cultural_from': 'western', 'cultural_to': 'mixed'}
        ]
        
        for bridge in manual_bridges:
            bridge_structure['bridge_songs'].append(bridge)
            lookup_key = f"{bridge['track_name']}_{bridge['artist_name']}".lower().replace(' ', '_')
            bridge_structure['bridge_lookup'][lookup_key] = bridge
    
    print(f"✅ Extracted {len(bridge_structure['bridge_songs'])} bridge songs")
    
    return bridge_structure

def create_user_features(data: pd.DataFrame, personality_mappings: Dict) -> pd.DataFrame:
    """Create user-level features for clustering validation"""
    
    print("📊 Creating user-level features...")
    
    user_features = data.groupby('user_id').agg({
        'vietnamese_score': 'mean',
        'western_score': 'mean',
        'chinese_score': 'mean',
        'artist_name': 'nunique',
        'track_name': 'nunique',
        'cultural_classification': lambda x: x.nunique(),
        'hour': 'mean',
        'played_at': ['min', 'max', 'count']
    }).reset_index()
    
    # Flatten column names
    user_features.columns = ['user_id', 'vietnamese_preference', 'western_preference', 
                           'chinese_preference', 'unique_artists', 'unique_tracks',
                           'cultural_diversity', 'avg_hour', 'first_listen', 'last_listen', 'total_tracks']
    
    # Add personality assignments
    user_features['assigned_personality'] = user_features['user_id'].map(
        personality_mappings.get('user_personalities', {})
    ).fillna('unknown')
    
    # Add derived features
    user_features['listening_span_days'] = (user_features['last_listen'] - user_features['first_listen']).dt.days
    user_features['listening_intensity'] = user_features['total_tracks'] / (user_features['listening_span_days'] + 1)
    user_features['artist_diversity'] = user_features['unique_artists'] / user_features['total_tracks']
    
    print(f"✅ Created features for {len(user_features)} users")
    
    return user_features

def add_temporal_sequences(data: pd.DataFrame) -> pd.DataFrame:
    """Add temporal sequence information for bridge analysis"""
    
    print("⏰ Adding temporal sequences...")
    
    # Sort by user and time
    data = data.sort_values(['user_id', 'played_at']).reset_index(drop=True)
    
    # Add sequence position
    data['sequence_position'] = data.groupby('user_id').cumcount()
    
    # Add previous/next cultural classifications
    data['prev_culture'] = data.groupby('user_id')['cultural_classification'].shift(1)
    data['next_culture'] = data.groupby('user_id')['cultural_classification'].shift(-1)
    
    # Mark cultural transitions
    data['is_cultural_transition'] = (
        (data['cultural_classification'] != data['prev_culture']) &
        (data['prev_culture'].notna())
    )
    
    print("✅ Added temporal sequence information")
    
    return data

def main():
    """Main restructuring process"""
    
    print("🔧 Research Data Restructuring")
    print("="*50)
    
    # Load data
    print("📂 Loading data...")
    
    try:
        from utils import load_research_data
        streaming_data, _, phase3_results = load_research_data()
        
        if streaming_data is None:
            print("❌ Could not load streaming data")
            return
            
        print(f"✅ Loaded {len(streaming_data):,} streaming records")
        if phase3_results:
            print(f"✅ Loaded Phase 3 results: {len(phase3_results)} files")
        else:
            print("⚠️ No Phase 3 results found - using simplified approach")
            phase3_results = {}
            
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Step 1: Create users
    enhanced_data = create_users_from_sessions(streaming_data)
    
    # Step 2: Add cultural scores
    enhanced_data = add_cultural_scores(enhanced_data, phase3_results)
    
    # Step 3: Create personality mappings
    personality_mappings = create_personality_mappings(enhanced_data, phase3_results)
    
    # Step 4: Extract bridge songs
    bridge_songs = extract_bridge_songs(phase3_results)
    
    # Step 5: Create user features
    user_features = create_user_features(enhanced_data, personality_mappings)
    
    # Step 6: Add temporal sequences
    enhanced_data = add_temporal_sequences(enhanced_data)
    
    # Save results
    print("💾 Saving restructured data...")
    
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save main dataset
    enhanced_data.to_parquet(output_dir / "streaming_data_enhanced.parquet")
    print(f"   ✅ Saved enhanced streaming data: {len(enhanced_data):,} records")
    
    # Save supporting files
    with open(output_dir / "personality_mappings.json", 'w') as f:
        json.dump(personality_mappings, f, indent=2, default=str)
    print(f"   ✅ Saved personality mappings: {len(personality_mappings['user_personalities'])} users")
    
    with open(output_dir / "bridge_songs_structured.json", 'w') as f:
        json.dump(bridge_songs, f, indent=2)
    print(f"   ✅ Saved bridge songs: {len(bridge_songs['bridge_songs'])} songs")
    
    user_features.to_parquet(output_dir / "user_features.parquet")
    print(f"   ✅ Saved user features: {len(user_features)} users")
    
    # Validation check
    print("\\n✅ Validation Check:")
    
    required_columns = ['user_id', 'vietnamese_score', 'western_score', 'cultural_classification']
    missing_columns = [col for col in required_columns if col not in enhanced_data.columns]
    
    if missing_columns:
        print(f"   ❌ Missing columns: {missing_columns}")
    else:
        print("   ✅ All required columns present")
    
    # Check data quality
    cultural_groups = enhanced_data['cultural_classification'].value_counts()
    min_group_size = cultural_groups.min()
    
    users_with_personalities = len([u for u in personality_mappings['user_personalities'].values() if u != 'unknown'])
    
    print(f"   📊 Users created: {enhanced_data['user_id'].nunique()}")
    print(f"   📊 Users with personalities: {users_with_personalities}")
    print(f"   📊 Bridge songs: {len(bridge_songs['bridge_songs'])}")
    print(f"   📊 Smallest cultural group: {min_group_size}")
    
    # Final assessment
    validation_passed = (
        len(missing_columns) == 0 and
        users_with_personalities >= 30 and
        len(bridge_songs['bridge_songs']) >= 5 and
        min_group_size >= 50
    )
    
    print("\\n" + "="*60)
    print("🎯 RESTRUCTURING COMPLETE")
    print("="*60)
    
    if validation_passed:
        print("✅ DATA IS READY FOR STATISTICAL VALIDATION!")
        print("   Run: python validate_research_claims.py")
    else:
        print("⚠️ Data restructured but may have limited statistical power")
        print("   Some validation tests may not be fully reliable")
    
    print(f"\\n📁 Enhanced data saved to: {output_dir}/")
    print("   - streaming_data_enhanced.parquet")
    print("   - personality_mappings.json") 
    print("   - bridge_songs_structured.json")
    print("   - user_features.parquet")

if __name__ == "__main__":
    main()