#!/usr/bin/env python3
"""
Comprehensive Analysis of Spotify Playlist Collection

Analyzes the rich playlist data from Downloads/spotify_playlists/
to understand mood-based and cultural listening patterns.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class PlaylistCollectionAnalyzer:
    """
    Comprehensive analyzer for the Spotify playlist collection.
    
    Analyzes 19 playlists with 671+ tracks, focusing on:
    - Mood-based listening patterns
    - Cultural distribution
    - Audio characteristic profiles
    - Temporal evolution of playlist creation
    """
    
    def __init__(self, playlist_dir: str = "/Users/quangnguyen/Downloads/spotify_playlists"):
        self.playlist_dir = Path(playlist_dir)
        self.playlists = {}
        self.combined_data = None
        self.results = {}
        
    def load_all_playlists(self):
        """Load all playlist CSV files"""
        print("🎵 Loading Spotify Playlist Collection...")
        
        csv_files = list(self.playlist_dir.glob("*.csv"))
        print(f"Found {len(csv_files)} playlists")
        
        for csv_file in csv_files:
            playlist_name = csv_file.stem
            try:
                df = pd.read_csv(csv_file)
                df['playlist_name'] = playlist_name
                df['Added At'] = pd.to_datetime(df['Added At'])
                self.playlists[playlist_name] = df
                print(f"  ✅ {playlist_name}: {len(df)} tracks")
            except Exception as e:
                print(f"  ❌ {playlist_name}: {str(e)}")
        
        # Combine all data
        self.combined_data = pd.concat(self.playlists.values(), ignore_index=True)
        print(f"\n📊 Total: {len(self.combined_data)} tracks across {len(self.playlists)} playlists")
        
    def analyze_playlist_categories(self):
        """Analyze playlist categories and patterns"""
        print("\n🗂️  Analyzing Playlist Categories...")
        
        # Categorize playlists
        categories = {
            'cultural': ['vpop', 'us-uk'],
            'mood_positive': ['gleeful', 'something_cute_and_dynamic', 'sheer_love'],
            'mood_negative': ['heartbreaking', 'such_a_bad_day'],
            'mood_chill': ['dreamy', 'chillie', 'tempalative_mood'],
            'activity': ['daily_music', 'on_repeat', 'repeat_rewind'],
            'memory': ['memory_brings_back', 'lyrics_nail_ur_heart'],
            'energy': ['underground_battle'],
            'favorites': ['liked_songs', 'best_songs'],
            'special': ['i_concert_lnh_chi']
        }
        
        # Reverse mapping
        playlist_to_category = {}
        for category, playlists in categories.items():
            for playlist in playlists:
                playlist_to_category[playlist] = category
        
        # Add category to data
        self.combined_data['category'] = self.combined_data['playlist_name'].map(playlist_to_category)
        
        # Category statistics
        category_stats = {}
        for category, playlists in categories.items():
            category_tracks = []
            for playlist in playlists:
                if playlist in self.playlists:
                    category_tracks.extend(self.playlists[playlist].to_dict('records'))
            
            if category_tracks:
                category_df = pd.DataFrame(category_tracks)
                category_stats[category] = {
                    'playlists': playlists,
                    'total_tracks': len(category_tracks),
                    'avg_valence': category_df['Valence'].mean(),
                    'avg_energy': category_df['Energy'].mean(),
                    'avg_danceability': category_df['Danceability'].mean(),
                    'avg_acousticness': category_df['Acousticness'].mean(),
                    'unique_artists': category_df['Artist Name(s)'].nunique()
                }
        
        self.results['category_analysis'] = category_stats
        
        # Print insights
        print(f"\n📈 Category Insights:")
        for category, stats in category_stats.items():
            print(f"  {category.upper()}: {stats['total_tracks']} tracks")
            print(f"    Valence: {stats['avg_valence']:.3f}, Energy: {stats['avg_energy']:.3f}")
            print(f"    Unique artists: {stats['unique_artists']}")
        
    def analyze_cultural_patterns(self):
        """Deep analysis of cultural distribution across playlists"""
        print("\n🌏 Analyzing Cultural Patterns...")
        
        # Define cultural indicators
        vietnamese_patterns = [
            'v-pop', 'vietnamese', 'vietnam indie', 'vinahouse', 'vietnamese lo-fi',
            'vietnamese hip hop', 'vietnamese bolero'
        ]
        
        western_patterns = [
            'soft pop', 'pop', 'hip hop', 'rap', 'rock', 'country', 'r&b',
            'electronic', 'dance', 'house', 'edm', 'dubstep'
        ]
        
        chinese_patterns = [
            'c-pop', 'mandopop', 'chinese r&b', 'cantopop'
        ]
        
        # Classify tracks culturally
        def classify_culture(genres_str):
            if pd.isna(genres_str):
                return 'unknown'
            
            genres_lower = str(genres_str).lower()
            
            vn_score = sum(1 for pattern in vietnamese_patterns if pattern in genres_lower)
            western_score = sum(1 for pattern in western_patterns if pattern in genres_lower)
            chinese_score = sum(1 for pattern in chinese_patterns if pattern in genres_lower)
            
            if vn_score > max(western_score, chinese_score):
                return 'vietnamese'
            elif western_score > max(vn_score, chinese_score):
                return 'western'
            elif chinese_score > 0:
                return 'chinese'
            else:
                return 'other'
        
        self.combined_data['cultural_classification'] = self.combined_data['Genres'].apply(classify_culture)
        
        # Cultural distribution by playlist
        cultural_analysis = {}
        for playlist_name, df in self.playlists.items():
            df['cultural_classification'] = df['Genres'].apply(classify_culture)
            cultural_dist = df['cultural_classification'].value_counts()
            
            cultural_analysis[playlist_name] = {
                'total_tracks': len(df),
                'cultural_distribution': cultural_dist.to_dict(),
                'dominant_culture': cultural_dist.index[0] if len(cultural_dist) > 0 else 'unknown',
                'cultural_diversity': len(cultural_dist),
                'vietnamese_ratio': cultural_dist.get('vietnamese', 0) / len(df),
                'western_ratio': cultural_dist.get('western', 0) / len(df),
                'chinese_ratio': cultural_dist.get('chinese', 0) / len(df)
            }
        
        self.results['cultural_analysis'] = cultural_analysis
        
        # Overall cultural statistics
        overall_cultural = self.combined_data['cultural_classification'].value_counts()
        print(f"\n🎭 Overall Cultural Distribution:")
        for culture, count in overall_cultural.items():
            percentage = count / len(self.combined_data) * 100
            print(f"  {culture.title()}: {count} tracks ({percentage:.1f}%)")
        
        # Most culturally diverse playlists
        diverse_playlists = sorted(cultural_analysis.items(), 
                                 key=lambda x: x[1]['cultural_diversity'], reverse=True)
        print(f"\n🌍 Most Culturally Diverse Playlists:")
        for playlist, data in diverse_playlists[:5]:
            print(f"  {playlist}: {data['cultural_diversity']} cultures, {data['total_tracks']} tracks")
    
    def analyze_audio_characteristics(self):
        """Analyze audio characteristics across playlists"""
        print("\n🎵 Analyzing Audio Characteristics...")
        
        # Key audio features
        audio_features = ['Valence', 'Energy', 'Danceability', 'Acousticness', 
                         'Speechiness', 'Liveness', 'Loudness', 'Tempo']
        
        # Playlist audio profiles
        playlist_profiles = {}
        for playlist_name, df in self.playlists.items():
            profile = {}
            for feature in audio_features:
                if feature in df.columns:
                    profile[f'avg_{feature.lower()}'] = df[feature].mean()
                    profile[f'std_{feature.lower()}'] = df[feature].std()
            
            playlist_profiles[playlist_name] = profile
        
        self.results['audio_profiles'] = playlist_profiles
        
        # Find playlists with extreme characteristics
        extremes = {
            'most_energetic': ('', 0),
            'most_chill': ('', 1),
            'most_happy': ('', 0),
            'most_sad': ('', 1),
            'most_danceable': ('', 0),
            'most_acoustic': ('', 0)
        }
        
        for playlist, profile in playlist_profiles.items():
            energy = profile.get('avg_energy', 0)
            valence = profile.get('avg_valence', 0)
            danceability = profile.get('avg_danceability', 0)
            acousticness = profile.get('avg_acousticness', 0)
            
            if energy > extremes['most_energetic'][1]:
                extremes['most_energetic'] = (playlist, energy)
            if energy < extremes['most_chill'][1]:
                extremes['most_chill'] = (playlist, energy)
            if valence > extremes['most_happy'][1]:
                extremes['most_happy'] = (playlist, valence)
            if valence < extremes['most_sad'][1]:
                extremes['most_sad'] = (playlist, valence)
            if danceability > extremes['most_danceable'][1]:
                extremes['most_danceable'] = (playlist, danceability)
            if acousticness > extremes['most_acoustic'][1]:
                extremes['most_acoustic'] = (playlist, acousticness)
        
        print(f"\n🎯 Audio Characteristic Extremes:")
        for characteristic, (playlist, value) in extremes.items():
            print(f"  {characteristic.replace('_', ' ').title()}: {playlist} ({value:.3f})")
        
        self.results['audio_extremes'] = extremes
    
    def analyze_temporal_evolution(self):
        """Analyze temporal patterns in playlist creation"""
        print("\n⏰ Analyzing Temporal Evolution...")
        
        def classify_culture(genres_str):
            if pd.isna(genres_str):
                return 'unknown'
            
            genres_lower = str(genres_str).lower()
            vietnamese_indicators = ['v-pop', 'vietnamese', 'vietnam']
            western_indicators = ['pop', 'hip hop', 'rap', 'rock', 'r&b']
            
            if any(ind in genres_lower for ind in vietnamese_indicators):
                return 'vietnamese'
            elif any(ind in genres_lower for ind in western_indicators):
                return 'western'
            else:
                return 'other'
        
        # Extract dates from 'Added At' column
        all_additions = []
        for playlist_name, df in self.playlists.items():
            for _, row in df.iterrows():
                all_additions.append({
                    'playlist': playlist_name,
                    'added_at': row['Added At'],
                    'track_name': row['Track Name'],
                    'artist': row['Artist Name(s)'],
                    'year': row['Added At'].year,
                    'month': row['Added At'].month,
                    'cultural_classification': classify_culture(row.get('Genres', ''))
                })
        
        additions_df = pd.DataFrame(all_additions)
        
        # Temporal patterns
        temporal_analysis = {
            'date_range': {
                'earliest': additions_df['added_at'].min(),
                'latest': additions_df['added_at'].max(),
                'span_days': (additions_df['added_at'].max() - additions_df['added_at'].min()).days
            },
            'yearly_activity': additions_df.groupby('year').size().to_dict(),
            'monthly_patterns': additions_df.groupby(['year', 'month']).size().to_dict(),
            'playlist_creation_timeline': {}
        }
        
        # Playlist creation patterns
        for playlist_name, df in self.playlists.items():
            if len(df) > 0:
                first_added = df['Added At'].min()
                last_added = df['Added At'].max()
                span = (last_added - first_added).days
                
                temporal_analysis['playlist_creation_timeline'][playlist_name] = {
                    'first_track': first_added,
                    'last_track': last_added,
                    'creation_span_days': span,
                    'tracks_count': len(df),
                    'creation_intensity': len(df) / max(span, 1)  # tracks per day
                }
        
        self.results['temporal_analysis'] = temporal_analysis
        
        print(f"\n📅 Temporal Insights:")
        date_range = temporal_analysis['date_range']
        print(f"  Collection span: {date_range['earliest'].strftime('%Y-%m-%d')} to {date_range['latest'].strftime('%Y-%m-%d')}")
        print(f"  Total days: {date_range['span_days']}")
        
        # Most active periods
        yearly_activity = temporal_analysis['yearly_activity']
        most_active_year = max(yearly_activity.keys(), key=lambda k: yearly_activity[k])
        print(f"  Most active year: {most_active_year} ({yearly_activity[most_active_year]} tracks added)")
        
    def identify_bridge_songs(self):
        """Identify songs that appear in multiple playlists (potential bridges)"""
        print("\n🌉 Identifying Cross-Playlist Bridge Songs...")
        
        # Track appearances across playlists
        track_appearances = defaultdict(list)
        
        for playlist_name, df in self.playlists.items():
            for _, row in df.iterrows():
                track_key = f"{row['Track Name']} - {row['Artist Name(s)']}"
                track_appearances[track_key].append({
                    'playlist': playlist_name,
                    'added_at': row['Added At'],
                    'genres': row.get('Genres', ''),
                    'valence': row.get('Valence', 0),
                    'energy': row.get('Energy', 0),
                    'danceability': row.get('Danceability', 0)
                })
        
        # Find songs in multiple playlists
        bridge_songs = {}
        for track, appearances in track_appearances.items():
            if len(appearances) > 1:
                # Calculate bridge score
                playlist_diversity = len(set(app['playlist'] for app in appearances))
                avg_valence = np.mean([app['valence'] for app in appearances])
                avg_energy = np.mean([app['energy'] for app in appearances])
                
                # Bridge score: playlist diversity + emotional versatility
                bridge_score = playlist_diversity + (1 - abs(avg_valence - 0.5)) + (1 - abs(avg_energy - 0.5))
                
                bridge_songs[track] = {
                    'appearances': len(appearances),
                    'playlists': [app['playlist'] for app in appearances],
                    'bridge_score': bridge_score,
                    'avg_valence': avg_valence,
                    'avg_energy': avg_energy,
                    'first_added': min(app['added_at'] for app in appearances),
                    'playlist_span': len(set(app['playlist'] for app in appearances))
                }
        
        # Sort by bridge score
        top_bridges = sorted(bridge_songs.items(), key=lambda x: x[1]['bridge_score'], reverse=True)
        
        self.results['bridge_songs'] = dict(top_bridges)
        
        print(f"\n🌉 Found {len(bridge_songs)} songs appearing in multiple playlists")
        print(f"\n🏆 Top Bridge Songs:")
        for i, (track, data) in enumerate(top_bridges[:10], 1):
            playlists_str = ', '.join(data['playlists'][:3])
            if len(data['playlists']) > 3:
                playlists_str += f" (+{len(data['playlists'])-3} more)"
            print(f"  {i:2d}. {track}")
            print(f"      Playlists: {playlists_str}")
            print(f"      Bridge Score: {data['bridge_score']:.2f}")
    
    def generate_insights_report(self):
        """Generate comprehensive insights report"""
        print("\n📊 Generating Comprehensive Insights Report...")
        
        report = {
            'metadata': {
                'analysis_date': datetime.now().isoformat(),
                'total_playlists': len(self.playlists),
                'total_tracks': len(self.combined_data),
                'unique_artists': self.combined_data['Artist Name(s)'].nunique(),
                'date_range': f"{self.combined_data['Added At'].min()} to {self.combined_data['Added At'].max()}"
            },
            'key_discoveries': [],
            'playlist_insights': {},
            'cultural_insights': self.results.get('cultural_analysis', {}),
            'audio_insights': self.results.get('audio_profiles', {}),
            'temporal_insights': self.results.get('temporal_analysis', {}),
            'bridge_songs': self.results.get('bridge_songs', {}),
            'recommendations': []
        }
        
        # Key discoveries
        if 'cultural_analysis' in self.results:
            cultural_data = self.results['cultural_analysis']
            
            # Find most cross-cultural playlists
            cross_cultural = [(name, data) for name, data in cultural_data.items() 
                            if data['cultural_diversity'] >= 3]
            
            if cross_cultural:
                report['key_discoveries'].append({
                    'type': 'cross_cultural_playlists',
                    'discovery': f"Found {len(cross_cultural)} highly cross-cultural playlists",
                    'details': [f"{name}: {data['cultural_diversity']} cultures" 
                              for name, data in cross_cultural[:5]]
                })
        
        # Audio characteristics insights
        if 'audio_extremes' in self.results:
            extremes = self.results['audio_extremes']
            report['key_discoveries'].append({
                'type': 'audio_extremes',
                'discovery': 'Identified playlists with extreme audio characteristics',
                'details': {k: f"{v[0]} ({v[1]:.3f})" for k, v in extremes.items()}
            })
        
        # Bridge songs insights
        if 'bridge_songs' in self.results:
            bridge_count = len([song for song, data in self.results['bridge_songs'].items() 
                              if data['appearances'] >= 3])
            report['key_discoveries'].append({
                'type': 'bridge_songs',
                'discovery': f"Identified {bridge_count} songs with high cross-playlist appeal",
                'details': list(self.results['bridge_songs'].keys())[:10]
            })
        
        # Recommendations
        report['recommendations'] = [
            "Create mood-based recommendation clusters using playlist categories",
            "Use bridge songs as cultural transition points in recommendations",
            "Leverage temporal patterns for personalized playlist generation",
            "Build emotion-aware recommendation system using audio characteristics"
        ]
        
        # Save report
        output_path = Path("results/playlist_analysis_report.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str, ensure_ascii=False)
        
        print(f"✅ Comprehensive report saved to {output_path}")
        
        # Print summary
        print(f"\n🎯 ANALYSIS SUMMARY:")
        print(f"  📁 Analyzed: {report['metadata']['total_playlists']} playlists")
        print(f"  🎵 Total tracks: {report['metadata']['total_tracks']:,}")
        print(f"  🎤 Unique artists: {report['metadata']['unique_artists']:,}")
        print(f"  🔍 Key discoveries: {len(report['key_discoveries'])}")
        print(f"  🌉 Bridge songs: {len(report['bridge_songs'])}")
        
        return report


def main():
    """Run comprehensive playlist analysis"""
    print("🎵 Comprehensive Spotify Playlist Analysis")
    print("=" * 50)
    
    analyzer = PlaylistCollectionAnalyzer()
    
    try:
        # Load all playlists
        analyzer.load_all_playlists()
        
        # Run comprehensive analysis
        analyzer.analyze_playlist_categories()
        analyzer.analyze_cultural_patterns()
        analyzer.analyze_audio_characteristics()
        analyzer.analyze_temporal_evolution()
        analyzer.identify_bridge_songs()
        
        # Generate final report
        report = analyzer.generate_insights_report()
        
        print(f"\n🎉 Analysis Complete!")
        print(f"This rich playlist collection reveals sophisticated mood-based and cultural listening patterns")
        print(f"Perfect complement to your existing 71K+ streaming records!")
        
    except Exception as e:
        print(f"❌ Error in analysis: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()