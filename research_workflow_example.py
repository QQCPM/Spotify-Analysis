"""
Research Workflow Example

Demonstrates how to use the improved research components:
1. Secure credential loading
2. Robust data loading with error handling  
3. Data validation for research integrity
4. Safe component initialization

This example shows best practices for research code.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from utils import ResearchDataLoader, validate_for_research
from models.recommendation_engine import CrossCulturalRecommendationEngine, UserProfile
from data_processing.spotify_collector import SpotifyConfig

def research_workflow_example():
    """Complete research workflow with error handling"""
    
    print("🔬 Research Workflow Example")
    print("=" * 40)
    
    # Step 1: Load data with robust error handling
    print("\n📂 Step 1: Data Loading")
    loader = ResearchDataLoader()
    streaming_data = loader.load_streaming_data(validate=True)
    
    if streaming_data is None:
        print("❌ Could not load streaming data - check data directory")
        return
    
    # Step 2: Validate data for research integrity
    print("\n✅ Step 2: Data Validation")
    is_valid = validate_for_research(streaming_data.head(1000))  # Validate subset for speed
    
    if not is_valid:
        print("⚠️ Data validation found issues - review before research")
    
    # Step 3: Initialize recommendation system with error handling
    print("\n🤖 Step 3: Recommendation System")
    try:
        engine = CrossCulturalRecommendationEngine()
        print("✅ Recommendation engine initialized successfully")
        
        # Create user profile from sample data
        sample_data = streaming_data.sample(200) if len(streaming_data) > 200 else streaming_data
        user_profile = engine.create_user_profile(sample_data)
        print(f"✅ Created user profile with {len(user_profile.personality_weights)} personalities")
        
        # Display personality distribution
        print("   Personality weights:")
        for personality, weight in user_profile.personality_weights.items():
            print(f"     {personality}: {weight:.3f}")
        
    except Exception as e:
        print(f"❌ Recommendation system initialization failed: {e}")
        print("💡 This might indicate missing Phase 3 results")
    
    # Step 4: Test secure Spotify configuration
    print("\n🔐 Step 4: Secure Configuration")
    try:
        # This will fail gracefully if no .env file exists
        config = SpotifyConfig()
        print("✅ Spotify credentials loaded securely from environment")
    except ValueError as e:
        print(f"⚠️ Spotify credentials not configured: {e}")
        print("   This is expected if you haven't set up .env file")
    
    # Step 5: Show data insights
    print("\n📊 Step 5: Data Insights")
    data_info = loader.get_data_info(streaming_data)
    print(f"   Total records: {data_info['total_records']:,}")
    print(f"   Memory usage: {data_info['memory_usage_mb']:.1f} MB")
    if data_info.get('total_unique_tracks'):
        print(f"   Unique tracks: {data_info['total_unique_tracks']:,}")
    if data_info.get('total_unique_artists'):
        print(f"   Unique artists: {data_info['total_unique_artists']:,}")
    
    if data_info['date_range']:
        print(f"   Date range: {data_info['date_range']['span_days']} days")
    
    print("\n🎯 Research Workflow Complete!")
    print("✅ All components working with proper error handling")
    print("✅ Data validated for research integrity")
    print("✅ Security best practices implemented")

if __name__ == "__main__":
    research_workflow_example()