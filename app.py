"""
Transfer Market Predictor - Beta v1.0 Web App
Streamlit application for predicting football player market values
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Transfer Market Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-header">⚽ Transfer Market Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Machine Learning Model for Football Player Valuations | Beta v1.0</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/football2--v1.png", width=100)
    st.title("Navigation")
    page = st.radio(
        "Select Page",
        ["🏠 Home", "🔍 Search Player", "🔮 Predict Value", "📊 Model Insights", "ℹ️ About"]
    )

    st.markdown("---")
    st.markdown("### Model Stats")
    st.metric("R-squared", "0.689", "68.9%")
    st.metric("MAE", "€4.97M", "Avg error")
    st.metric("Dataset", "7,722", "Observations")
    st.metric("Features", "67", "Engineered")

    st.markdown("---")
    st.markdown("**Version:** Beta v1.0")
    st.markdown("**Status:** Production Ready")
    st.markdown("**Date:** Dec 2024")

# Load model (cached)
@st.cache_resource
def load_model():
    try:
        model_data = joblib.load('data/market_value_model_historical.pkl')
        return model_data
    except FileNotFoundError:
        st.error("⚠️ Model file not found. Please ensure 'data/market_value_model_historical.pkl' exists.")
        return None

model_data = load_model()

# Load sample data (cached)
@st.cache_data
def load_sample_data():
    try:
        df = pd.read_csv('data/training_dataset_with_fifa.csv')
        return df
    except FileNotFoundError:
        st.warning("⚠️ Sample data not available. Some features may be limited.")
        return None

sample_data = load_sample_data()

# ============================================================================
# HOME PAGE
# ============================================================================
if page == "🏠 Home":
    st.header("Welcome to the Transfer Market Predictor!")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 🎯 Purpose")
        st.write("""
        Predict football player market values using machine learning.
        Built for sporting director and technical director applications.
        """)

    with col2:
        st.markdown("### 🔬 Technology")
        st.write("""
        - Gradient Boosting Regressor
        - 67 engineered features
        - Multi-source data integration
        - Zero data leakage
        """)

    with col3:
        st.markdown("### 📊 Performance")
        st.write("""
        - R² = 0.689 (69% variance explained)
        - MAE = €4.97M average error
        - 7,722 player-season observations
        - Big 5 European leagues
        """)

    st.markdown("---")

    # Feature categories
    st.subheader("📦 Feature Categories")

    feature_categories = {
        "Performance (23)": "Goals, assists, xG, xA, progressive stats",
        "Team Context (11)": "Team performance, player importance (NO leakage)",
        "FIFA Ratings (7)": "Overall, potential, wonderkid status",
        "Multipliers (7)": "League, age, position premiums",
        "Progressive Stats (6)": "Ball progression metrics from FBref",
        "Demographics (9)": "Age, position, league, nationality",
        "Age Categories (4)": "Prime age, young talent, veteran flags"
    }

    cols = st.columns(3)
    for i, (category, description) in enumerate(feature_categories.items()):
        with cols[i % 3]:
            st.markdown(f"**{category}**")
            st.caption(description)

    st.markdown("---")

    # Sample players table
    if sample_data is not None:
        st.subheader("⭐ Sample High-Value Players")
        top_players = sample_data.nlargest(10, 'market_value_millions')[
            ['player_name', 'team', 'league', 'age', 'goals', 'assists', 'market_value_millions']
        ]
        top_players.columns = ['Player', 'Team', 'League', 'Age', 'Goals', 'Assists', 'Value (€M)']
        st.dataframe(top_players, use_container_width=True, hide_index=True)

    # Quick stats
    st.markdown("---")
    st.subheader("📈 Quick Statistics")

    if sample_data is not None:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Players", f"{len(sample_data):,}")
        col2.metric("Avg Value", f"€{sample_data['market_value_millions'].mean():.2f}M")
        col3.metric("Max Value", f"€{sample_data['market_value_millions'].max():.0f}M")
        col4.metric("Leagues", sample_data['league'].nunique())

# ============================================================================
# SEARCH PLAYER PAGE
# ============================================================================
elif page == "🔍 Search Player":
    st.header("Search Player & Compare Predictions")

    if model_data is None or sample_data is None:
        st.error("⚠️ Model or data not loaded. Cannot make predictions.")
    else:
        st.info("💡 Search for any player in our dataset to see predicted vs actual market value using the full 67-feature model.")

        # Create searchable player list
        if 'player_name' in sample_data.columns:
            # Add season if available
            if 'season' in sample_data.columns:
                sample_data['search_label'] = sample_data['player_name'] + " (" + sample_data['team'] + ", " + sample_data['season'].astype(str) + ")"
            else:
                sample_data['search_label'] = sample_data['player_name'] + " (" + sample_data['team'] + ")"

            # Search box
            search_query = st.text_input("🔍 Type player name to search", placeholder="e.g., Haaland, Mbappe, Vinicius...")

            if search_query:
                # Filter players
                filtered = sample_data[sample_data['search_label'].str.contains(search_query, case=False, na=False)]

                if len(filtered) > 0:
                    # Remove duplicates based on player_name and season (keep first occurrence)
                    if 'season' in filtered.columns:
                        filtered = filtered.drop_duplicates(subset=['player_name', 'season'], keep='first')
                        # Sort by season (most recent first)
                        filtered = filtered.sort_values('season', ascending=False)
                    else:
                        # Remove duplicates based on player_name and team
                        filtered = filtered.drop_duplicates(subset=['player_name', 'team'], keep='first')

                    # Get unique player names and default to most recent season
                    # Group by player_name and take the first (most recent) entry
                    if 'season' in filtered.columns:
                        latest_entries = filtered.groupby('player_name').first().reset_index()
                        # Rebuild search labels
                        latest_entries['search_label'] = latest_entries['player_name'] + " (" + latest_entries['team'] + ", " + latest_entries['season'].astype(str) + ")"
                        # Remove any duplicate search labels (in case groupby created duplicates)
                        latest_entries = latest_entries.drop_duplicates(subset=['search_label'], keep='first')
                        display_options = latest_entries['search_label'].tolist()

                        # But keep all entries available in filtered for selection
                        all_options = filtered['search_label'].tolist()
                    else:
                        display_options = filtered['search_label'].tolist()
                        all_options = display_options

                    # Show dropdown with filtered results (most recent season first)
                    selected_player = st.selectbox(
                        f"Select from {len(display_options)} player(s) - showing most recent season",
                        options=display_options,
                        key="player_select"
                    )

                    # Option to view all seasons
                    if 'season' in filtered.columns and len(all_options) > len(display_options):
                        show_all = st.checkbox("Show all seasons for matched players", value=False)
                        if show_all:
                            selected_player = st.selectbox(
                                f"All matches ({len(all_options)} total)",
                                options=all_options,
                                key="player_select_all"
                            )

                    if selected_player:
                        # Get player data
                        player_row = filtered[filtered['search_label'] == selected_player].iloc[0]

                        # Prepare features for prediction
                        model = model_data['model']
                        feature_names = model_data['feature_names']

                        # Get actual value
                        actual_value = player_row['market_value_millions']

                        # Get label encoders
                        label_encoders = model_data.get('label_encoders', {})

                        # Create a copy of player_row to modify
                        player_data = player_row.copy()

                        # Apply label encoding to categorical features
                        for col, encoder in label_encoders.items():
                            if col in player_data.index and player_data[col] is not None:
                                try:
                                    player_data[col] = encoder.transform([player_data[col]])[0]
                                except ValueError:
                                    # If value not in encoder, use mode or default
                                    player_data[col] = 0

                        # Check which features are available
                        available_features = [f for f in feature_names if f in player_data.index]
                        missing_features = [f for f in feature_names if f not in player_data.index]

                        # Only proceed if we have most features
                        if len(available_features) >= len(feature_names) * 0.8:  # At least 80% of features
                            # Extract available features and fill missing with 0
                            X_player = []
                            for feat in feature_names:
                                if feat in player_data.index:
                                    val = player_data[feat]
                                    # Convert to float, handle any remaining non-numeric
                                    try:
                                        X_player.append(float(val))
                                    except (ValueError, TypeError):
                                        X_player.append(0)
                                else:
                                    X_player.append(0)  # Default value for missing features

                            X_player = np.array(X_player, dtype=float).reshape(1, -1)

                            # Make prediction
                            if model_data.get('use_log_transform', False):
                                predicted_log = model.predict(X_player)[0]
                                predicted_value = np.exp(predicted_log)
                            else:
                                predicted_value = model.predict(X_player)[0]

                            # Show warning if features are missing
                            if len(missing_features) > 0:
                                st.warning(f"⚠️ **Note:** {len(missing_features)} engineered features are missing and have been filled with default values. Prediction accuracy may be reduced.")
                                with st.expander("Show missing features"):
                                    st.write(", ".join(missing_features))
                        else:
                            st.error(f"⚠️ Cannot make prediction. Missing {len(missing_features)} critical features.")
                            st.stop()

                        # Calculate error
                        error = predicted_value - actual_value
                        error_pct = (error / actual_value) * 100 if actual_value > 0 else 0

                        # Display results
                        st.markdown("---")
                        st.subheader(f"⚽ {player_row['player_name']}")

                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Actual Value", f"€{actual_value:.2f}M", "Transfermarkt")
                        col2.metric("Predicted Value", f"€{predicted_value:.2f}M", "AI Model")
                        col3.metric("Error", f"€{abs(error):.2f}M", f"{error_pct:+.1f}%")

                        # Accuracy indicator
                        if abs(error_pct) < 10:
                            accuracy = "Excellent"
                            color = "🟢"
                        elif abs(error_pct) < 25:
                            accuracy = "Good"
                            color = "🟡"
                        else:
                            accuracy = "Fair"
                            color = "🟠"
                        col4.metric("Accuracy", accuracy, color)

                        # Comparison chart
                        st.markdown("### 📊 Value Comparison")
                        comparison_df = pd.DataFrame({
                            'Type': ['Actual (Transfermarkt)', 'Predicted (AI Model)'],
                            'Value (€M)': [actual_value, predicted_value]
                        })

                        fig = px.bar(
                            comparison_df,
                            x='Type',
                            y='Value (€M)',
                            color='Type',
                            title=f'{player_row["player_name"]} - Market Value Comparison',
                            color_discrete_map={
                                'Actual (Transfermarkt)': '#1f77b4',
                                'Predicted (AI Model)': '#ff7f0e'
                            }
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        # Player summary statistics
                        st.markdown("### 📋 Player Profile")

                        summary_col1, summary_col2, summary_col3 = st.columns(3)

                        with summary_col1:
                            st.metric("Name", player_row['player_name'])
                            st.metric("Team", player_row['team'])

                        with summary_col2:
                            # Clean up nationality display with standardized country codes
                            nationality = player_row.get('nationality', 'N/A')
                            if nationality != 'N/A':
                                nationality = str(nationality).upper().strip()
                                # Map common 2-letter codes to 3-letter ISO codes
                                nationality_map = {
                                    'NO': 'NOR',  # Norway
                                    'EN': 'ENG',  # England
                                    'SC': 'SCO',  # Scotland
                                    'WA': 'WAL',  # Wales
                                    'NI': 'NIR',  # Northern Ireland
                                    'AR': 'ARG',  # Argentina
                                    'BR': 'BRA',  # Brazil
                                    'DE': 'GER',  # Germany
                                    'FR': 'FRA',  # France
                                    'ES': 'ESP',  # Spain
                                    'IT': 'ITA',  # Italy
                                    'PT': 'POR',  # Portugal
                                    'NL': 'NED',  # Netherlands
                                    'BE': 'BEL',  # Belgium
                                }
                                # Split by spaces/commas and deduplicate
                                nat_parts = [nationality_map.get(part.strip(), part.strip())
                                           for part in nationality.replace(',', ' ').split() if part.strip()]
                                # Remove duplicates while preserving order
                                seen = set()
                                unique_parts = []
                                for part in nat_parts:
                                    if part not in seen:
                                        seen.add(part)
                                        unique_parts.append(part)
                                nationality = unique_parts[0] if unique_parts else 'N/A'
                            st.metric("Nationality", nationality)
                            st.metric("Position", player_row.get('position_group', 'N/A'))

                        with summary_col3:
                            st.metric("Age", f"{player_row['age']} years")
                            st.metric("FIFA Overall", player_row.get('fifa_overall', 'N/A'))

                        if len(missing_features) == 0:
                            st.success(f"✅ Prediction used all **67 features** with the full Gradient Boosting model (R²=0.689)")
                        else:
                            st.info(f"✅ Prediction used **{len(available_features)}/{len(feature_names)} features** with the Gradient Boosting model (R²=0.689)")
                else:
                    st.warning(f"No players found matching '{search_query}'. Try a different name.")

# ============================================================================
# PREDICT VALUE PAGE
# ============================================================================
elif page == "🔮 Predict Value":
    st.header("Predict Player Market Value")

    st.info("💡 Enter player statistics to get a market value prediction. All fields are required for accurate predictions.")

    if model_data is None:
        st.error("⚠️ Model not loaded. Cannot make predictions.")
    else:
        # Create input form
        with st.form("prediction_form"):
            st.subheader("Player Information")

            col1, col2, col3 = st.columns(3)

            with col1:
                age = st.number_input("Age", min_value=16, max_value=40, value=25)
                minutes_played = st.number_input("Minutes Played", min_value=0, max_value=4000, value=2500)
                goals = st.number_input("Goals", min_value=0, max_value=100, value=10)

            with col2:
                assists = st.number_input("Assists", min_value=0, max_value=100, value=5)
                appearances = st.number_input("Appearances", min_value=0, max_value=60, value=30)
                xg_total = st.number_input("Expected Goals (xG)", min_value=0.0, max_value=100.0, value=8.5, step=0.1)

            with col3:
                xa_total = st.number_input("Expected Assists (xA)", min_value=0.0, max_value=100.0, value=4.0, step=0.1)
                league = st.selectbox("League", ["eng Premier League", "es La Liga", "de Bundesliga", "it Serie A", "fr Ligue 1"])
                position_group = st.selectbox("Position", ["ATT", "MID", "DEF", "GK"])

            submit = st.form_submit_button("🔮 Predict Market Value", use_container_width=True)

        if submit:
            st.markdown("---")
            st.subheader("Prediction Result")

            # Note: This is a simplified example
            # In production, you'd need to create ALL 67 features
            st.warning("⚠️ **Note:** This is a simplified demo. Full prediction requires all 67 features. This gives an approximate estimate.")

            # Simplified prediction logic
            # In reality, you'd compute all 67 features
            base_value = (goals * 2) + (assists * 1.5) + (xg_total * 1.2) + (xa_total * 1.0)
            base_value = base_value * (minutes_played / 3000) * (35 - abs(age - 25)) / 10

            # League multipliers
            league_mult = {
                "eng Premier League": 1.35,
                "es La Liga": 1.15,
                "de Bundesliga": 1.05,
                "it Serie A": 1.00,
                "fr Ligue 1": 0.90
            }

            # Position multipliers
            pos_mult = {"ATT": 1.0, "MID": 0.95, "DEF": 0.85, "GK": 0.70}

            predicted_value = base_value * league_mult[league] * pos_mult[position_group]
            predicted_value = max(0.5, min(predicted_value, 180))  # Clip to reasonable range

            col1, col2, col3 = st.columns(3)
            col1.metric("Predicted Value", f"€{predicted_value:.2f}M", "AI Estimate")
            col2.metric("Confidence", "Medium", "Simplified model")
            col3.metric("Range", f"€{predicted_value*0.8:.1f}-{predicted_value*1.2:.1f}M", "±20%")

            # Player profile
            st.markdown("### Player Profile")
            profile_data = {
                "Metric": ["Age", "League", "Position", "Games", "Minutes", "Goals", "Assists", "xG", "xA"],
                "Value": [age, league.split()[1], position_group, appearances, minutes_played, goals, assists, xg_total, xa_total]
            }
            st.table(pd.DataFrame(profile_data))

            st.info("💡 **For accurate predictions:** Use the 'Search Player' page for full 67-feature model predictions.")

# ============================================================================
# MODEL INSIGHTS PAGE
# ============================================================================
elif page == "📊 Model Insights":
    st.header("Model Insights & Performance")

    # Performance metrics
    st.subheader("🎯 Overall Performance")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("R-squared", "0.689", "+159% vs baseline")
    col2.metric("MAE", "€4.97M", "Test set")
    col3.metric("Train R²", "0.745", "Good fit")
    col4.metric("CV Score", "0.685", "5-fold")

    # Feature importance (top 10)
    st.markdown("---")
    st.subheader("🔝 Top 10 Features by Importance")

    feature_importance = pd.DataFrame({
        'Feature': ['minutes_played', 'team_xa', 'team_xg', 'age', 'team_minutes',
                   'xa_total', 'xg_total', 'team_goals', 'goals_per_90', 'npxg_total'],
        'Importance': [15.6, 12.8, 8.1, 5.7, 5.2, 4.9, 4.7, 4.6, 3.8, 3.2],
        'Category': ['Performance', 'Team Context', 'Team Context', 'Demographics', 'Team Context',
                    'Performance', 'Performance', 'Team Context', 'Performance', 'Performance']
    })

    fig = px.bar(
        feature_importance,
        x='Importance',
        y='Feature',
        color='Category',
        orientation='h',
        title='Top 10 Most Important Features',
        labels={'Importance': 'Importance (%)', 'Feature': ''},
        color_discrete_map={
            'Performance': '#1f77b4',
            'Team Context': '#ff7f0e',
            'Demographics': '#2ca02c'
        }
    )
    fig.update_layout(yaxis={'categoryorder':'total ascending'}, height=400)
    st.plotly_chart(fig, use_container_width=True)

    st.info("💡 **Key Insight:** Team performance context (team_xa, team_xg, team_goals) accounts for 25.5% of total importance - validates the performance-based approach without data leakage.")

    # Performance by value range
    st.markdown("---")
    st.subheader("📊 Performance by Value Range")

    perf_by_range = pd.DataFrame({
        'Value Range': ['€0-5M', '€5-25M', '€25-50M', '€50M+'],
        'Players': [3706, 3220, 471, 325],
        'Percentage': [48, 42, 6, 4],
        'MAE (€M)': [2.1, 5.8, 8.3, 15.2],
        'R²': [0.52, 0.61, 0.48, 0.42]
    })

    col1, col2 = st.columns(2)

    with col1:
        fig1 = px.pie(
            perf_by_range,
            values='Players',
            names='Value Range',
            title='Player Distribution by Value'
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        fig2 = px.bar(
            perf_by_range,
            x='Value Range',
            y='MAE (€M)',
            title='Average Error by Value Range',
            color='MAE (€M)',
            color_continuous_scale='Reds'
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.caption("📌 Model performs best on mid-tier players (€5-25M). High-value players (€50M+) have more variance due to brand value and intangibles.")

# ============================================================================
# ABOUT PAGE
# ============================================================================
elif page == "ℹ️ About":
    st.header("About This Project")

    st.markdown("""
    ### 🎯 Project Overview

    The **Transfer Market Predictor** is a machine learning model that predicts football player market values
    using performance statistics, team context, and FIFA ratings. Built as a portfolio project for
    sporting director and technical director internship applications.

    ### 🔬 Methodology

    **Model:** Gradient Boosting Regressor with 200 estimators

    **Features:** 67 engineered features across 6 categories:
    - Performance statistics (goals, assists, xG, xA, etc.)
    - Progressive statistics (ball progression metrics)
    - Team context (performance-based aggregations)
    - FIFA ratings (overall, potential, wonderkid status)
    - Age categories and multipliers
    - League and position premiums

    **Data Sources:**
    - FBref (via Kaggle) - Performance statistics
    - Transfermarkt (via Kaggle) - Market valuations
    - EA Sports FIFA (via Kaggle) - Player ratings

    ### 🛡️ Data Leakage Prevention

    **Critical Design Choice:** All team context features derived from performance metrics (goals, xG, assists)
    rather than market values to prevent data leakage.

    ✅ **What we do:**
    - Calculate team_goals, team_xg excluding the target player
    - Use performance-based player importance metrics
    - No circular dependencies on market value

    ❌ **What we DON'T do:**
    - Use player's own market value in features
    - Use team total value that includes the player
    - Use future information to predict past seasons

    ### 📊 Performance

    - **R² = 0.689** (68.9% variance explained)
    - **MAE = €4.97M** (mean absolute error)
    - **Dataset:** 7,722 player-season observations
    - **Coverage:** Big 5 European leagues (2021-22, 2023-24, 2024-25)

    ### 🚀 Future Development

    **Beta v2.0 (Planned - January 2025):**
    - Temporal dataset: 71K observations, 14 seasons (2011-2024)
    - Time-based train/test splits
    - Expected R² = 0.90+, MAE = €3-3.5M

    ### 👤 Author

    **Owen James**
    - Economics Student & Football Analytics Enthusiast
    - Aspiring Sporting Director / Technical Director
    - Target: Summer 2025 Internships

    ### 📄 Links

    - 🔗 [GitHub Repository](https://github.com/Foxesrevenge/transfer-market-predictor)
    - 🚀 [Live App](https://transfer-market-predictor.streamlit.app)

    ### 📝 License

    MIT License - Educational and portfolio use.
    Data attribution required (FBref, Transfermarkt, EA Sports FIFA).

    ---

    **Version:** Beta v1.0
    **Last Updated:** December 19, 2024
    **Status:** Production Ready
    """)

    # Contact
    st.markdown("---")
    st.subheader("📧 Contact")
    col1, col2, col3 = st.columns(3)
    col1.markdown("**LinkedIn:** Owen James")
    col2.markdown("**GitHub:** [Repository](https://github.com/Foxesrevenge/transfer-market-predictor)")
    col3.markdown("**Email:** owenajames05swim@gmail.com")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 1rem;'>
        Transfer Market Predictor Beta v1.0 | Built with ❤️ using Streamlit |
        <a href='#'>Documentation</a> | <a href='#'>GitHub</a>
    </div>
    """,
    unsafe_allow_html=True
)
