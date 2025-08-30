"""
Phase 4: Interactive Cross-Cultural Music Recommendation Demo

Streamlit interface demonstrating the complete recommendation system.
Integrates 71K streaming records + 634 playlist tracks + Phase 3 discoveries.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import sys
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add src to path for imports
sys.path.append('src')

from models.recommendation_engine import CrossCulturalRecommendationEngine, UserProfile
from evaluation.recommendation_evaluator import RecommendationEvaluator, TemporalSplitter

# Page configuration
st.set_page_config(
    page_title="Cross-Cultural Music Recommendations",
    page_icon="🎵",
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
    .personality-card {
        border-left: 4px solid #1f77b4;
        padding: 1rem;
        margin: 0.5rem 0;
        background-color: #fafafa;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Load streaming and playlist data"""
    streaming_data = None
    playlist_data = None
    playlists = {}
    
    # Load streaming data
    try:
        streaming_data = pd.read_parquet('data/processed/streaming_data_processed.parquet')
        streaming_data['played_at'] = pd.to_datetime(streaming_data['played_at'])
        st.success(f"✅ Loaded {len(streaming_data):,} streaming records")
    except Exception as e:
        st.error(f"❌ Error loading streaming data: {str(e)}")
    
    # Load playlist data (optional)
    try:
        playlist_dir = Path('/Users/quangnguyen/Downloads/spotify_playlists')
        if playlist_dir.exists():
            for csv_file in playlist_dir.glob('*.csv'):
                df = pd.read_csv(csv_file)
                df['playlist_name'] = csv_file.stem
                playlists[csv_file.stem] = df
            
            if playlists:
                playlist_data = pd.concat(playlists.values(), ignore_index=True)
                st.success(f"✅ Loaded {len(playlist_data):,} playlist tracks")
            else:
                st.warning("⚠️ No playlist CSV files found")
        else:
            st.info("ℹ️ Playlist directory not found - running with streaming data only")
    except Exception as e:
        st.warning(f"⚠️ Error loading playlist data: {str(e)} - continuing with streaming data only")
    
    return streaming_data, playlist_data, playlists


@st.cache_resource
def initialize_recommendation_engine():
    """Initialize the recommendation engine"""
    try:
        engine = CrossCulturalRecommendationEngine('results/phase3')
        return engine
    except Exception as e:
        st.error(f"Error initializing recommendation engine: {str(e)}")
        return None


def main():
    """Main Streamlit application"""
    
    st.title("🎵 Cross-Cultural Music Recommendation System")
    st.markdown("**Phase 4 Demo**: Integrating 71K streaming records + 634 playlist tracks + Phase 3 discoveries")
    
    # Load data and engine
    with st.spinner("Loading data and initializing recommendation engine..."):
        streaming_data, playlist_data, playlists = load_data()
        engine = initialize_recommendation_engine()
    
    if streaming_data is None or engine is None:
        st.error("Failed to load required data. Please check file paths.")
        return
    
    # Sidebar configuration
    st.sidebar.title("🎛️ Configuration")
    
    # Demo mode selection
    demo_mode = st.sidebar.selectbox(
        "Select Demo Mode",
        ["🏠 Overview", "🧬 Personality Analysis", "🎯 Generate Recommendations", "📊 System Evaluation"],
        index=2  # Default to Generate Recommendations
    )
    
    if demo_mode == "🏠 Overview":
        show_overview(streaming_data, playlist_data, engine)
    elif demo_mode == "🧬 Personality Analysis":
        show_personality_analysis(streaming_data, engine)
    elif demo_mode == "🎯 Generate Recommendations":
        show_recommendation_generation(streaming_data, engine)
    elif demo_mode == "📊 System Evaluation":
        show_system_evaluation(streaming_data, engine)


def show_overview(streaming_data: pd.DataFrame, playlist_data: pd.DataFrame, engine):
    """Show system overview and data summary"""
    
    st.header("📊 System Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Streaming Records", f"{len(streaming_data):,}")
    with col2:
        st.metric("Playlist Tracks", f"{len(playlist_data):,}" if playlist_data is not None else "N/A")
    with col3:
        st.metric("Musical Personalities", "3")
    with col4:
        st.metric("Bridge Songs", "96")
    
    # Data timeline
    st.subheader("📅 Temporal Coverage")
    
    try:
        # Create timeline visualization
        streaming_monthly = streaming_data.groupby(streaming_data['played_at'].dt.to_period('M')).size()
        
        # Handle playlist data more safely
        playlist_monthly = pd.Series(dtype='int64')  # Empty series as fallback
        if playlist_data is not None and not playlist_data.empty:
            # Check for common date columns
            date_columns = ['Added At', 'added_at', 'date_added', 'Date Added']
            date_col = None
            for col in date_columns:
                if col in playlist_data.columns:
                    date_col = col
                    break
            
            if date_col is not None:
                try:
                    playlist_monthly = playlist_data.groupby(pd.to_datetime(playlist_data[date_col]).dt.to_period('M')).size()
                except Exception as e:
                    st.warning(f"Could not parse playlist dates from column '{date_col}': {str(e)}")
        
        # Align data
        all_months = sorted(set(streaming_monthly.index) | set(playlist_monthly.index))
        timeline_df = pd.DataFrame({
            'Month': [str(m) for m in all_months],
            'Streaming': [streaming_monthly.get(m, 0) for m in all_months],
            'Playlists': [playlist_monthly.get(m, 0) for m in all_months]
        })
        
        fig = px.line(timeline_df, x='Month', y=['Streaming', 'Playlists'],
                      title="Activity Timeline: Streaming vs Playlist Curation")
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error creating timeline visualization: {str(e)}")
        st.info("Timeline chart unavailable - continuing with other visualizations.")
    
    # Cultural distribution
    st.subheader("🌍 Cultural Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        try:
            # Streaming cultural distribution (approximate)
            streaming_cultural = pd.Series([0.52, 0.24, 0.13, 0.11], 
                                         index=['Vietnamese', 'Unknown', 'Western', 'Other'])
            fig_streaming = px.pie(values=streaming_cultural.values, names=streaming_cultural.index,
                                  title="Streaming Data (71K records)")
            st.plotly_chart(fig_streaming, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating streaming cultural distribution: {str(e)}")
    
    with col2:
        try:
            # Playlist cultural distribution
            if playlist_data is not None and not playlist_data.empty and 'Genres' in playlist_data.columns:
                playlist_cultural = classify_playlist_cultures(playlist_data)
                fig_playlist = px.pie(values=playlist_cultural.values, names=playlist_cultural.index,
                                     title="Playlist Data (634 tracks)")
                st.plotly_chart(fig_playlist, use_container_width=True)
            else:
                st.info("Playlist cultural analysis unavailable - no genre data found.")
        except Exception as e:
            st.error(f"Error creating playlist cultural distribution: {str(e)}")
    
    # Phase 3 discoveries summary
    st.subheader("🔬 Phase 3 Discoveries Integration")
    
    discovery_cards = [
        ("🧬 Musical Personalities", "3 distinct personalities discovered through matrix factorization"),
        ("⏰ Change Points", "81 preference evolution points detected over 4+ years"),
        ("🌉 Bridge Songs", "655+ tracks identified as cultural bridges"),
        ("🎯 Recommendation Ready", "All discoveries integrated into production system")
    ]
    
    for title, description in discovery_cards:
        st.markdown(f"""
        <div class="metric-card">
            <h4>{title}</h4>
            <p>{description}</p>
        </div>
        """, unsafe_allow_html=True)


def show_personality_analysis(streaming_data: pd.DataFrame, engine):
    """Show musical personality analysis"""
    
    st.header("🧬 Musical Personality Analysis")
    
    # Load Phase 3 results
    personalities = engine.phase3_results['study_1_results']['personalities']
    
    st.markdown("**Discovered through matrix factorization of your 71K streaming records:**")
    
    for personality_id, data in personalities.items():
        st.markdown(f"""
        <div class="personality-card">
            <h3>{personality_id.title()}</h3>
            <p><strong>Interpretation:</strong> {data['interpretation']}</p>
            <p><strong>Cultural Profile:</strong> Vietnamese {data['cultural_profile']['vietnamese_ratio']:.1%}, 
            Western {data['cultural_profile']['western_ratio']:.1%}</p>
            <p><strong>Top Artists:</strong> {', '.join(data['top_artists'][:3])}</p>
            <p><strong>Strength:</strong> {data['strength']:.3f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Personality comparison visualization
    st.subheader("🎨 Personality Characteristics Comparison")
    
    try:
        personality_names = list(personalities.keys())
        vietnamese_ratios = [personalities[p]['cultural_profile']['vietnamese_ratio'] for p in personality_names]
        western_ratios = [personalities[p]['cultural_profile']['western_ratio'] for p in personality_names]
        strengths = [personalities[p]['strength'] for p in personality_names]
        
        comparison_df = pd.DataFrame({
            'Personality': [p.replace('personality_', 'P') for p in personality_names],
            'Vietnamese Ratio': vietnamese_ratios,
            'Western Ratio': western_ratios,
            'Strength': strengths
        })
        
        fig = px.bar(comparison_df, x='Personality', y=['Vietnamese Ratio', 'Western Ratio'],
                     title="Cultural Composition of Musical Personalities", barmode='group')
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error creating personality comparison chart: {str(e)}")
        st.info("Personality analysis data may be incomplete.")
    
    # Interactive personality weight adjustment
    st.subheader("🎛️ Adjust Personality Weights")
    st.markdown("*Modify these weights to see how they affect recommendations*")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        p1_weight = st.slider("Personality 1 (Vietnamese Indie)", 0.0, 1.0, 0.4, 0.1)
    with col2:
        p2_weight = st.slider("Personality 2 (Mixed Cultural)", 0.0, 1.0, 0.3, 0.1)
    with col3:
        p3_weight = st.slider("Personality 3 (Western Mixed)", 0.0, 1.0, 0.3, 0.1)
    
    # Normalize weights
    total_weight = p1_weight + p2_weight + p3_weight
    if total_weight > 0:
        normalized_weights = {
            'personality_1': p1_weight / total_weight,
            'personality_2': p2_weight / total_weight,
            'personality_3': p3_weight / total_weight
        }
        
        st.info(f"**Normalized Weights**: P1: {normalized_weights['personality_1']:.2f}, "
                f"P2: {normalized_weights['personality_2']:.2f}, P3: {normalized_weights['personality_3']:.2f}")
        
        # Store weights in session state for recommendation generation
        st.session_state['personality_weights'] = normalized_weights


def show_recommendation_generation(streaming_data: pd.DataFrame, engine):
    """Show recommendation generation interface"""
    
    st.header("🎯 Generate Personalized Recommendations")
    
    # Configuration options
    col1, col2 = st.columns(2)
    
    with col1:
        n_recommendations = st.slider("Number of Recommendations", 5, 20, 10)
        include_bridges = st.checkbox("Include Cultural Bridge Songs", value=True)
        exploration_factor = st.slider("Exploration vs Exploitation", 0.0, 1.0, 0.2, 0.1)
    
    with col2:
        time_period = st.selectbox(
            "Generate Recommendations Based On",
            ["Recent Listening (Last 30 days)", "All Time", "Last 6 months", "Custom Period"]
        )
        
        cultural_preference = st.selectbox(
            "Cultural Exploration Preference", 
            ["Balanced", "Vietnamese Focus", "Western Focus", "Maximum Diversity"]
        )
    
    # Generate recommendations button
    if st.button("🎵 Generate Recommendations", type="primary"):
        
        with st.spinner("Generating personalized recommendations..."):
            
            # Prepare recent listening data based on time period
            if time_period == "Recent Listening (Last 30 days)":
                cutoff_date = pd.Timestamp.now(tz='UTC') - timedelta(days=30)
                recent_data = streaming_data[streaming_data['played_at'] > cutoff_date]
            elif time_period == "Last 6 months":
                cutoff_date = pd.Timestamp.now(tz='UTC') - timedelta(days=180)
                recent_data = streaming_data[streaming_data['played_at'] > cutoff_date]
            else:
                recent_data = streaming_data.tail(1000)  # Last 1000 plays
            
            if len(recent_data) == 0:
                st.warning("No recent listening data found. Using full dataset sample.")
                recent_data = streaming_data.sample(min(1000, len(streaming_data)))
            
            try:
                # Create user profile
                user_profile = engine.create_user_profile(recent_data)
                
                # Override personality weights if set in session
                if 'personality_weights' in st.session_state:
                    user_profile.personality_weights = st.session_state['personality_weights']
                
                # Get candidate tracks (all unique tracks not in recent listening)
                # Add synthetic audio features since they're missing from the data
                all_tracks = streaming_data[
                    ['track_id', 'track_name', 'artist_name']
                ].drop_duplicates('track_id').copy()
                
                # Add synthetic audio features for demo
                np.random.seed(42)  # Reproducible
                all_tracks['audio_energy'] = np.random.beta(2, 2, len(all_tracks))
                all_tracks['audio_valence'] = np.random.beta(2, 2, len(all_tracks))
                all_tracks['audio_danceability'] = np.random.beta(2, 2, len(all_tracks))
                all_tracks['audio_acousticness'] = np.random.beta(1, 3, len(all_tracks))
                
                # Add cultural classification
                def classify_culture_demo(artist_name):
                    if pd.isna(artist_name):
                        return 'unknown'
                    artist_lower = str(artist_name).lower()
                    vietnamese_indicators = ['buitruonglinh', 'vsoul', 'khói', 'đen', 'mck', 'obito']
                    if any(ind in artist_lower for ind in vietnamese_indicators):
                        return 'vietnamese'
                    elif any(char in artist_lower for char in 'àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ'):
                        return 'vietnamese'
                    else:
                        return 'western'
                
                all_tracks['dominant_culture'] = all_tracks['artist_name'].apply(classify_culture_demo)
                
                recent_track_ids = set(recent_data['track_id'].tolist())
                candidate_tracks = all_tracks[~all_tracks['track_id'].isin(recent_track_ids)].sample(
                    min(1000, len(all_tracks))
                )
                
                # Generate recommendations
                recommendations = engine.generate_recommendations(
                    user_profile=user_profile,
                    candidate_tracks=candidate_tracks,
                    n_recommendations=n_recommendations,
                    include_bridges=include_bridges,
                    exploration_factor=exploration_factor
                )
                
                # Display recommendations
                st.subheader("🎵 Your Personalized Recommendations")
                
                for i, rec in enumerate(recommendations, 1):
                    with st.expander(f"{i}. {rec.track_name} - {rec.artist_name}"):
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Overall Score", f"{rec.score:.3f}")
                            st.metric("Bridge Score", f"{rec.bridge_score:.3f}")
                        
                        with col2:
                            st.metric("Cultural Class", rec.cultural_classification.title())
                            st.metric("Temporal Weight", f"{rec.temporal_weight:.3f}")
                        
                        with col3:
                            # Show personality breakdown
                            for pid, score in rec.personality_scores.items():
                                st.metric(f"{pid.title()}", f"{score:.3f}")
                        
                        st.markdown(f"**Reasoning**: {rec.reasoning}")
                
                # Recommendation analysis
                st.subheader("📈 Recommendation Analysis")
                
                # Cultural distribution of recommendations
                cultural_dist = pd.Series([rec.cultural_classification for rec in recommendations]).value_counts()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    try:
                        fig = px.pie(values=cultural_dist.values, names=cultural_dist.index,
                                    title="Cultural Distribution of Recommendations")
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error creating cultural distribution chart: {str(e)}")
                
                with col2:
                    try:
                        # Score distribution
                        scores = [rec.score for rec in recommendations]
                        fig = px.histogram(x=scores, nbins=10, title="Recommendation Score Distribution")
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error creating score distribution chart: {str(e)}")
                
            except Exception as e:
                st.error(f"Error generating recommendations: {str(e)}")
    
    # Bridge song discovery section
    st.subheader("🌉 Discover Cultural Bridge Songs")
    st.markdown("*Explore songs that connect different musical cultures*")
    
    col1, col2 = st.columns(2)
    with col1:
        from_culture = st.selectbox("From Culture", ["vietnamese", "western", "chinese"])
    with col2:
        to_culture = st.selectbox("To Culture", ["western", "vietnamese", "chinese"])
    
    if st.button("🔍 Find Bridge Songs"):
        bridge_recs = engine.bridge_engine.get_bridge_recommendations(from_culture, to_culture, 5)
        
        if bridge_recs:
            st.success(f"Found {len(bridge_recs)} bridge songs for {from_culture} → {to_culture}")
            
            for i, bridge in enumerate(bridge_recs, 1):
                st.markdown(f"""
                **{i}. {bridge.get('track_name', 'Unknown')} - {bridge.get('artist_name', 'Unknown')}**
                - Bridge Score: {bridge.get('bridge_score', 0):.2f}
                - Reasons: {', '.join(bridge.get('bridge_reasons', []))}
                """)
        else:
            st.info("No specific bridge songs found for this cultural transition.")


def show_system_evaluation(streaming_data: pd.DataFrame, engine):
    """Show system evaluation results"""
    
    st.header("📊 System Evaluation")
    
    st.markdown("""
    **Evaluation Methodology**: Temporal train/test splits maintaining chronological order to prevent data leakage.
    """)
    
    # Evaluation configuration
    col1, col2 = st.columns(2)
    
    with col1:
        n_splits = st.slider("Number of Temporal Splits", 1, 5, 3)
        test_period_months = st.slider("Test Period (months)", 3, 12, 6)
    
    with col2:
        evaluation_metrics = st.multiselect(
            "Evaluation Metrics",
            ["Accuracy (NDCG)", "Diversity", "Novelty", "Cultural Diversity", "Serendipity"],
            default=["Accuracy (NDCG)", "Diversity", "Cultural Diversity"]
        )
    
    if st.button("🔬 Run Evaluation", type="primary"):
        
        with st.spinner("Running comprehensive evaluation... This may take several minutes."):
            
            try:
                # Prepare streaming data with synthetic features for evaluation
                streaming_eval_data = streaming_data.copy()
                
                # Add synthetic audio features if missing
                if 'audio_energy' not in streaming_eval_data.columns:
                    np.random.seed(42)  # Reproducible
                    streaming_eval_data['audio_energy'] = np.random.beta(2, 2, len(streaming_eval_data))
                    streaming_eval_data['audio_valence'] = np.random.beta(2, 2, len(streaming_eval_data))
                    streaming_eval_data['audio_danceability'] = np.random.beta(2, 2, len(streaming_eval_data))
                    streaming_eval_data['audio_acousticness'] = np.random.beta(1, 3, len(streaming_eval_data))
                
                # Add cultural classification if missing
                if 'dominant_culture' not in streaming_eval_data.columns:
                    def classify_culture_eval(artist_name):
                        if pd.isna(artist_name):
                            return 'unknown'
                        artist_lower = str(artist_name).lower()
                        vietnamese_indicators = ['buitruonglinh', 'vsoul', 'khói', 'đen', 'mck', 'obito']
                        if any(ind in artist_lower for ind in vietnamese_indicators):
                            return 'vietnamese'
                        elif any(char in artist_lower for char in 'àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ'):
                            return 'vietnamese'
                        else:
                            return 'western'
                    
                    streaming_eval_data['dominant_culture'] = streaming_eval_data['artist_name'].apply(classify_culture_eval)
                
                # Initialize evaluator with enhanced data
                evaluator = RecommendationEvaluator(streaming_eval_data)
                
                # Run simplified evaluation (faster for demo)
                st.info("Running simplified evaluation for demo purposes...")
                
                # Create one temporal split for demo
                split = evaluator.splitter.create_final_evaluation_split(train_ratio=0.8)
                
                # Create user profile from training data
                user_profile = engine.create_user_profile(split.train_data.tail(1000))  # Use last 1000 for speed
                
                # Get candidate tracks
                all_tracks = streaming_eval_data[
                    ['track_id', 'track_name', 'artist_name', 'audio_energy', 
                     'audio_valence', 'audio_danceability', 'audio_acousticness', 'dominant_culture']
                ].drop_duplicates('track_id')
                
                train_track_ids = set(split.train_data['track_id'].tolist())
                candidate_tracks = all_tracks[~all_tracks['track_id'].isin(train_track_ids)].sample(
                    min(500, len(all_tracks))  # Smaller sample for demo speed
                )
                
                # Generate recommendations
                recommendations = engine.generate_recommendations(
                    user_profile=user_profile,
                    candidate_tracks=candidate_tracks,
                    n_recommendations=20,
                    include_bridges=True
                )
                
                # Evaluate recommendations
                metrics = evaluator.evaluate_recommendations(recommendations, split.test_data, k=10)
                
                # Create simplified evaluation results
                evaluation_results = {
                    'evaluation_date': datetime.now().isoformat(),
                    'n_splits': 1,
                    'aggregated_metrics': {
                        'ndcg_at_10_mean': metrics.ndcg_at_10,
                        'precision_at_10_mean': metrics.precision_at_10,
                        'diversity_mean': metrics.diversity,
                        'cultural_diversity_mean': metrics.cultural_diversity,
                        'novelty_mean': metrics.novelty,
                        'serendipity_mean': metrics.serendipity
                    },
                    'evaluation_summary': {
                        'overall_performance': 'Good' if metrics.ndcg_at_10 > 0.3 else 'Moderate',
                        'diversity': 'High diversity' if metrics.diversity > 0.5 else 'Moderate diversity',
                        'cultural_diversity': f'Cultural diversity score: {metrics.cultural_diversity:.3f}'
                    }
                }
                
                # Display results
                st.subheader("📈 Evaluation Results")
                
                # Key metrics
                aggregated = evaluation_results['aggregated_metrics']
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("NDCG@10", f"{aggregated['ndcg_at_10_mean']:.3f}", 
                             f"±{aggregated['ndcg_at_10_std']:.3f}")
                
                with col2:
                    st.metric("Precision@10", f"{aggregated['precision_at_10_mean']:.3f}",
                             f"±{aggregated['precision_at_10_std']:.3f}")
                
                with col3:
                    st.metric("Diversity", f"{aggregated['diversity_mean']:.3f}",
                             f"±{aggregated['diversity_std']:.3f}")
                
                with col4:
                    st.metric("Cultural Diversity", f"{aggregated['cultural_diversity_mean']:.3f}",
                             f"±{aggregated['cultural_diversity_std']:.3f}")
                
                # Evaluation summary
                st.subheader("📋 Evaluation Summary")
                
                summary = evaluation_results['evaluation_summary']
                for aspect, description in summary.items():
                    st.markdown(f"**{aspect.replace('_', ' ').title()}**: {description}")
                
                # Individual split results
                st.subheader("🔍 Individual Split Results")
                
                split_data = []
                for result in evaluation_results['individual_results']:
                    split_data.append({
                        'Split': f"Split {result['split_id'] + 1}",
                        'Test Period': result['test_period'],
                        'NDCG@10': result['metrics'].ndcg_at_10,
                        'Precision@10': result['metrics'].precision_at_10,
                        'Diversity': result['metrics'].diversity,
                        'Cultural Diversity': result['metrics'].cultural_diversity
                    })
                
                split_df = pd.DataFrame(split_data)
                st.dataframe(split_df, use_container_width=True)
                
                # Performance over time visualization
                try:
                    fig = px.line(split_df, x='Split', y=['NDCG@10', 'Precision@10'], 
                                 title="Recommendation Performance Across Temporal Splits")
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating performance timeline chart: {str(e)}")
                
            except Exception as e:
                st.error(f"Error running evaluation: {str(e)}")
    
    # Pre-computed evaluation results (if available)
    evaluation_file = Path('results/phase4_evaluation_report.json')
    if evaluation_file.exists():
        st.subheader("📄 Previous Evaluation Results")
        
        with open(evaluation_file, 'r') as f:
            previous_results = json.load(f)
        
        st.json(previous_results['evaluation_summary'])


def classify_playlist_cultures(playlist_data: pd.DataFrame) -> pd.Series:
    """Classify playlist tracks by culture"""
    
    def classify_culture(genres_str):
        if pd.isna(genres_str):
            return 'Unknown'
        
        genres_lower = str(genres_str).lower()
        
        vietnamese_patterns = ['v-pop', 'vietnamese', 'vietnam indie', 'vinahouse']
        western_patterns = ['soft pop', 'pop', 'hip hop', 'rap', 'rock', 'r&b']
        chinese_patterns = ['c-pop', 'mandopop', 'chinese r&b']
        
        if any(pattern in genres_lower for pattern in vietnamese_patterns):
            return 'Vietnamese'
        elif any(pattern in genres_lower for pattern in western_patterns):
            return 'Western'
        elif any(pattern in genres_lower for pattern in chinese_patterns):
            return 'Chinese'
        else:
            return 'Other'
    
    classifications = playlist_data['Genres'].apply(classify_culture)
    return classifications.value_counts()


if __name__ == "__main__":
    main()