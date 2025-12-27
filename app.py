import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import random
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import numpy as np

# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(
    page_title="Cricket Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "Cricket Analytics Dashboard - ML-powered ball-by-ball cricket intelligence"
    }
)

# ==================================================
# CONSTANTS
# ==================================================
PHASE_OVERS = {
    "powerplay": (0, 6),
    "middle": (6, 15),
    "death": (15, 20)
}

BALL_RUNS_WEIGHTS = {
    "powerplay": [35, 30, 15, 12, 8],
    "middle": [40, 30, 15, 10, 5],
    "death": [30, 25, 15, 15, 15]
}

# ==================================================
# LIGHT MODE CSS
# ==================================================
st.markdown("""
<style>
.stApp {
    background-color: #ffffff;
    color: #1f1f1f;
}
section[data-testid="stSidebar"] {
    background-color: #f8f9fa;
}
h1, h2, h3, h4 {
    color: #1e88e5;
}
[data-testid="stMetric"] {
    background-color: #f8f9fa;
    padding: 15px;
    border-radius: 10px;
    border: 1px solid #e0e0e0;
}
[data-testid="stMetricLabel"] {
    font-size: 14px;
    color: #666666;
}
[data-testid="stMetricValue"] {
    font-size: 24px;
    font-weight: bold;
    color: #1f1f1f;
}
.stButton > button {
    background-color: #1e88e5;
    color: white;
    border-radius: 8px;
    height: 3em;
    font-weight: bold;
    border: none;
    transition: all 0.3s;
}
.stButton > button:hover {
    background-color: #1565c0;
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(30, 136, 229, 0.3);
}
.stProgress > div > div {
    background-color: #1e88e5;
}
thead tr th {
    background-color: #e3f2fd;
    padding: 12px;
    font-weight: bold;
    color: #1f1f1f;
}
tbody tr td {
    background-color: #ffffff;
    padding: 10px;
    color: #1f1f1f;
}
.dataframe {
    border: 1px solid #e0e0e0;
    border-radius: 8px;
}
.element-container {
    margin-bottom: 1rem;
}
div[data-testid="stExpander"] {
    background-color: #f8f9fa;
    border: 1px solid #e0e0e0;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

# ==================================================
# TITLE & HEADER
# ==================================================
st.markdown(
    """
    <h1 style='text-align:center; margin-bottom:0;'>🏏 Cricket Analytics & Wicket Prediction</h1>
    <p style='text-align:center; color:#666666; margin-top:10px; font-size:18px;'>
    ML-powered ball-by-ball cricket intelligence
    </p>
    """,
    unsafe_allow_html=True
)

st.divider()

# ==================================================
# DATA LOADING WITH ERROR HANDLING
# ==================================================
@st.cache_resource(show_spinner="Loading ML model...")
def load_model(model_path: str = "models/wicket_prediction_rf.pkl"):
    """Load the trained Random Forest model."""
    try:
        if not Path(model_path).exists():
            st.error(f"Model file not found at {model_path}")
            return None
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

@st.cache_data(show_spinner="Loading cricket data...")
def load_data() -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Load all cricket statistics datasets."""
    try:
        data_dir = Path("data")
        
        batter = pd.read_csv(data_dir / "batter_stats.csv")
        bowler = pd.read_csv(data_dir / "bowler_stats.csv")
        matchups = pd.read_csv(data_dir / "batter_bowler_matchups.csv")
        
        # Data validation
        required_batter_cols = ["batter", "matches_played", "runs", "strike_rate", "average"]
        required_bowler_cols = ["bowler"]
        required_matchup_cols = ["batter", "bowler", "strike_rate", "dismissals"]
        
        for col in required_batter_cols:
            if col not in batter.columns:
                st.warning(f"Missing column '{col}' in batter_stats.csv")
        
        return batter, bowler, matchups
        
    except FileNotFoundError as e:
        st.error(f"Data file not found: {str(e)}")
        return None, None, None
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None, None

# Initialize data
model = load_model()
batter_stats, bowler_stats, matchup_stats = load_data()

# Check if data loaded successfully
if any(df is None for df in [batter_stats, bowler_stats, matchup_stats]):
    st.error("Failed to load required data. Please check your data files.")
    st.stop()

# ==================================================
# SIDEBAR NAVIGATION
# ==================================================
st.sidebar.title("📌 Navigation")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Select Page",
    [
        "🎯 Wicket Prediction",
        "🔍 Player Search",
        "⚔ Batter vs Bowler",
        "🏟 Venue Analysis",
        "📊 Match Simulation"
    ],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.info(
    "💡 **Tip**: Use this dashboard to analyze player performance, "
    "predict wicket probabilities, and simulate matches."
)

# ==================================================
# HELPER FUNCTIONS
# ==================================================
def get_phase_from_over(over: int) -> str:
    """Determine match phase from over number."""
    if over < 6:
        return "powerplay"
    elif over < 15:
        return "middle"
    else:
        return "death"

def get_h2h_stats(batter: str, bowler: str) -> Tuple[float, int]:
    """Get head-to-head statistics between batter and bowler."""
    try:
        row = matchup_stats[
            (matchup_stats["batter"] == batter) &
            (matchup_stats["bowler"] == bowler)
        ]
        if row.empty:
            return 0.0, 0
        return float(row.iloc[0]["strike_rate"]), int(row.iloc[0]["dismissals"])
    except Exception as e:
        st.warning(f"Error fetching H2H stats: {str(e)}")
        return 0.0, 0

def calculate_pressure(runs: int, balls: int) -> float:
    """Calculate match pressure based on runs and balls."""
    if balls == 0:
        return 0.0
    return min((runs + 1) / (balls + 1), 3.0)

def get_risk_category(probability: float) -> Tuple[str, str, str]:
    """Categorize wicket probability into risk levels."""
    if probability > 0.5:
        return "🔥 High Risk", "error", "#d32f2f"
    elif probability > 0.3:
        return "⚠️ Medium Risk", "warning", "#f57c00"
    else:
        return "✅ Low Risk", "success", "#388e3c"

def create_player_radar_chart(stats: pd.Series) -> plt.Figure:
    """Create a radar chart for player statistics."""
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    
    categories = ['Strike Rate', 'Average', 'Boundaries', 'Form']
    values = [
        stats.get('strike_rate', 0) / 2,  # Normalize
        stats.get('average', 0),
        stats.get('boundaries', 0) / 10,  # Normalize
        75  # Placeholder for form
    ]
    
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]
    
    ax.plot(angles, values, 'o-', linewidth=2, color='#1e88e5')
    ax.fill(angles, values, alpha=0.25, color='#1e88e5')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=10)
    ax.set_ylim(0, 100)
    ax.grid(True)
    
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#f8f9fa')
    ax.tick_params(colors='#333333')
    
    return fig

def simulate_match(model, num_overs: int = 20) -> Tuple[int, int, List[str], Dict]:
    """Simulate a complete cricket match with detailed statistics."""
    runs = 0
    wickets = 0
    timeline = []
    over_stats = []
    total_balls = 0
    
    for over in range(num_overs):
        over_runs = 0
        over_wickets = 0
        
        for ball in range(6):
            if wickets >= 10:
                break
                
            phase = get_phase_from_over(over)
            pressure = calculate_pressure(runs, total_balls)
            
            input_df = pd.DataFrame([{
                "over": over,
                "phase": phase,
                "pressure": pressure,
                "bat_pos": min(4 + wickets, 11),
                "batter_form": np.random.uniform(0.8, 1.5),
                "bowler_form": np.random.uniform(0.8, 1.5),
                "venue_avg_runs": 1.4,
                "h2h_strike_rate": 120,
                "h2h_dismissals": 0
            }])
            
            try:
                prob = model.predict_proba(input_df)[0][1]
            except:
                prob = 0.15  # Default probability if prediction fails
            
            # Simulate ball outcome
            if random.random() < prob:
                wickets += 1
                over_wickets += 1
                timeline.append(f"**Over {over}.{ball+1}** → ❌ WICKET (Prob: {prob:.2%})")
                if wickets == 10:
                    break
            else:
                ball_runs = random.choices(
                    [0, 1, 2, 4, 6],
                    weights=BALL_RUNS_WEIGHTS[phase]
                )[0]
                runs += ball_runs
                over_runs += ball_runs
                
                emoji = "🔴" if ball_runs == 0 else "1️⃣" if ball_runs == 1 else "2️⃣" if ball_runs == 2 else "🟢" if ball_runs == 4 else "🚀"
                timeline.append(f"**Over {over}.{ball+1}** → {emoji} {ball_runs} run(s)")
            
            total_balls += 1
        
        over_stats.append({
            "over": over + 1,
            "runs": over_runs,
            "wickets": over_wickets,
            "run_rate": runs / ((over + 1) * 6 / 6)
        })
        
        if wickets >= 10:
            break
    
    match_stats = {
        "over_stats": over_stats,
        "final_run_rate": runs / (total_balls / 6) if total_balls > 0 else 0,
        "total_balls": total_balls
    }
    
    return runs, wickets, timeline, match_stats

# ==================================================
# PAGE: 🎯 WICKET PREDICTION
# ==================================================
if page == "🎯 Wicket Prediction":
    st.subheader("🎯 Wicket Probability Predictor")
    st.markdown("Predict the likelihood of a wicket on the next ball based on match conditions")
    
    if model is None:
        st.error("Model not loaded. Cannot make predictions.")
    else:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("##### Match Situation")
            over = st.slider("Over", 0, 19, 10, help="Current over number (0-19)")
            phase = st.selectbox(
                "Phase", 
                ["powerplay", "middle", "death"],
                index=1,
                help="Match phase: Powerplay (0-6), Middle (6-15), Death (15-20)"
            )
            pressure = st.slider(
                "Pressure Index", 
                0.5, 3.0, 1.2, 0.1,
                help="Higher values indicate more pressure on the batting team"
            )
        
        with col2:
            st.markdown("##### Batter Details")
            batter = st.selectbox("Batter", sorted(batter_stats["batter"].unique()))
            bat_pos = st.slider("Batting Position", 1, 11, 4, help="Position in batting order")
            batter_form = st.slider(
                "Batter Form", 
                0.5, 2.0, 1.3, 0.1,
                help="Recent form multiplier (1.0 = average)"
            )
        
        with col3:
            st.markdown("##### Bowler Details")
            bowler = st.selectbox("Bowler", sorted(bowler_stats["bowler"].unique()))
            bowler_form = st.slider(
                "Bowler Form", 
                0.5, 2.0, 1.1, 0.1,
                help="Recent form multiplier (1.0 = average)"
            )
            venue_avg_runs = st.slider(
                "Venue Avg Runs", 
                0.8, 2.0, 1.4, 0.1,
                help="Venue run-scoring tendency"
            )
        
        # Get H2H statistics
        h2h_sr, h2h_dismissals = get_h2h_stats(batter, bowler)
        
        # Display H2H stats
        st.markdown("---")
        st.markdown("##### 📊 Head-to-Head Statistics")
        h2h_col1, h2h_col2, h2h_col3 = st.columns(3)
        h2h_col1.metric("Strike Rate", f"{h2h_sr:.1f}" if h2h_sr > 0 else "N/A")
        h2h_col2.metric("Dismissals", h2h_dismissals)
        h2h_col3.metric("Balls Faced", int(h2h_dismissals * 20) if h2h_dismissals > 0 else "N/A")
        
        st.markdown("---")
        
        # Prediction
        if st.button("🔮 Predict Wicket Probability", use_container_width=True, type="primary"):
            input_df = pd.DataFrame([{
                "over": over,
                "phase": phase,
                "pressure": pressure,
                "bat_pos": bat_pos,
                "batter_form": batter_form,
                "bowler_form": bowler_form,
                "venue_avg_runs": venue_avg_runs,
                "h2h_strike_rate": h2h_sr,
                "h2h_dismissals": h2h_dismissals
            }])
            
            try:
                prob = model.predict_proba(input_df)[0][1]
                
                # Display results
                st.markdown("### Prediction Results")
                
                result_col1, result_col2, result_col3 = st.columns([2, 1, 1])
                
                with result_col1:
                    st.progress(prob, text=f"Wicket Probability: {prob*100:.2f}%")
                
                with result_col2:
                    st.metric("Wicket Chance", f"{prob*100:.1f}%")
                
                with result_col3:
                    risk_text, risk_type, risk_color = get_risk_category(prob)
                    st.markdown(f"<h3 style='color:{risk_color};'>{risk_text}</h3>", unsafe_allow_html=True)
                
                # Detailed analysis
                with st.expander("📈 Detailed Analysis"):
                    st.markdown(f"""
                    **Factors Contributing to Prediction:**
                    - **Match Phase**: {phase.capitalize()} (Over {over})
                    - **Pressure Index**: {pressure:.2f}
                    - **Batter Position**: {bat_pos}
                    - **Form Differential**: {(batter_form - bowler_form):.2f} (Batter advantage: {'Yes' if batter_form > bowler_form else 'No'})
                    - **Historical H2H**: {h2h_dismissals} dismissals in past encounters
                    """)
                    
                    if prob > 0.5:
                        st.warning("⚠️ **High Risk Scenario** - Consider defensive field placement")
                    elif prob > 0.3:
                        st.info("ℹ️ **Moderate Risk** - Balanced approach recommended")
                    else:
                        st.success("✅ **Low Risk** - Aggressive field placement possible")
                
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")

# ==================================================
# PAGE: 🔍 PLAYER SEARCH
# ==================================================
elif page == "🔍 Player Search":
    st.subheader("🔍 Player Analytics Dashboard")
    st.markdown("Comprehensive statistics and performance analysis")
    
    # Player selection
    search_col1, search_col2 = st.columns([3, 1])
    with search_col1:
        player = st.selectbox("Select Player", sorted(batter_stats["batter"].unique()))
    with search_col2:
        player_type = st.radio("Type", ["Batter", "Bowler"], horizontal=True)
    
    st.markdown("---")
    
    if player_type == "Batter":
        player_data = batter_stats[batter_stats["batter"] == player]
        if player_data.empty:
            st.warning("No data available for this batter.")
        else:
            stats = player_data.iloc[0]
            
            # Key metrics
            st.markdown("### 📊 Key Performance Indicators")
            metric_col1, metric_col2, metric_col3, metric_col4, metric_col5 = st.columns(5)
            
            metric_col1.metric("Matches", int(stats.get("matches_played", 0)))
            metric_col2.metric("Runs", int(stats.get("runs", 0)))
            metric_col3.metric("Strike Rate", f"{stats.get('strike_rate', 0):.2f}")
            metric_col4.metric("Average", f"{stats.get('average', 0):.2f}")
            metric_col5.metric("Boundaries", int(stats.get("boundaries", 0)) if "boundaries" in stats else "N/A")
            
            st.markdown("---")
            
            # Visualizations
            viz_col1, viz_col2 = st.columns(2)
            
            with viz_col1:
                st.markdown("### 📈 Performance Distribution")
                fig, ax = plt.subplots(figsize=(8, 5))
                
                metrics = ["Runs", "Boundaries", "Strike Rate", "Average"]
                values = [
                    stats.get("runs", 0) / 10,  # Scale down for visualization
                    stats.get("boundaries", 0),
                    stats.get("strike_rate", 0) / 10,
                    stats.get("average", 0)
                ]
                
                bars = ax.bar(metrics, values, color=['#1e88e5', '#388e3c', '#f57c00', '#d32f2f'])
                ax.set_ylabel("Value (Scaled)", color='#333333')
                ax.set_title(f"{player} - Performance Metrics", color='#1f1f1f')
                ax.tick_params(colors='#333333')
                
                fig.patch.set_facecolor('#ffffff')
                ax.set_facecolor('#f8f9fa')
                ax.spines['bottom'].set_color('#e0e0e0')
                ax.spines['left'].set_color('#e0e0e0')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                
                st.pyplot(fig)
            
            with viz_col2:
                st.markdown("### 🎯 Performance Radar")
                if all(col in stats.index for col in ['strike_rate', 'average']):
                    radar_fig = create_player_radar_chart(stats)
                    st.pyplot(radar_fig)
                else:
                    st.info("Insufficient data for radar chart")
            
            # Historical matchups
            st.markdown("---")
            st.markdown("### ⚔️ Recent Matchups")
            player_matchups = matchup_stats[matchup_stats["batter"] == player].head(10)
            if not player_matchups.empty:
                st.dataframe(
                    player_matchups,
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("No matchup data available for this player.")
    
    else:  # Bowler
        player_data = bowler_stats[bowler_stats["bowler"] == player]
        if player_data.empty:
            st.warning("No data available for this bowler.")
        else:
            stats = player_data.iloc[0]
            
            # Key metrics
            st.markdown("### 📊 Key Performance Indicators")
            metric_col1, metric_col2, metric_col3, metric_col4, metric_col5 = st.columns(5)
            
            metric_col1.metric("Matches", int(stats.get("matches_played", 0)))
            metric_col2.metric("Wickets", int(stats.get("wickets", 0)))
            metric_col3.metric("Economy", f"{stats.get('economy', 0):.2f}")
            metric_col4.metric("Average", f"{stats.get('average', 0):.2f}")
            metric_col5.metric("Strike Rate", f"{stats.get('strike_rate', 0):.2f}" if "strike_rate" in stats else "N/A")
            
            st.markdown("---")
            
            # Visualizations
            viz_col1, viz_col2 = st.columns(2)
            
            with viz_col1:
                st.markdown("### 📈 Performance Distribution")
                fig, ax = plt.subplots(figsize=(8, 5))
                
                metrics = ["Wickets", "Economy", "Average", "Matches"]
                values = [
                    stats.get("wickets", 0),
                    stats.get("economy", 0) * 5,  # Scale up for visualization
                    stats.get("average", 0),
                    stats.get("matches_played", 0) / 2  # Scale down
                ]
                
                bars = ax.bar(metrics, values, color=['#d32f2f', '#f57c00', '#388e3c', '#1e88e5'])
                ax.set_ylabel("Value (Scaled)", color='#333333')
                ax.set_title(f"{player} - Bowling Performance", color='#1f1f1f')
                ax.tick_params(colors='#333333')
                
                fig.patch.set_facecolor('#ffffff')
                ax.set_facecolor('#f8f9fa')
                ax.spines['bottom'].set_color('#e0e0e0')
                ax.spines['left'].set_color('#e0e0e0')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                
                st.pyplot(fig)
            
            with viz_col2:
                st.markdown("### 🎯 Bowling Radar")
                fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
                
                categories = ['Wickets', 'Economy', 'Average', 'Consistency']
                # Normalize values for radar chart (0-100 scale)
                values = [
                    min(stats.get('wickets', 0) * 3, 100),  # Scale wickets
                    max(100 - stats.get('economy', 0) * 10, 0),  # Lower economy is better
                    max(100 - stats.get('average', 0) * 2, 0),  # Lower average is better
                    75  # Placeholder for consistency
                ]
                
                angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
                values += values[:1]
                angles += angles[:1]
                
                ax.plot(angles, values, 'o-', linewidth=2, color='#d32f2f')
                ax.fill(angles, values, alpha=0.25, color='#d32f2f')
                ax.set_xticks(angles[:-1])
                ax.set_xticklabels(categories, size=10)
                ax.set_ylim(0, 100)
                ax.grid(True)
                
                fig.patch.set_facecolor('#ffffff')
                ax.set_facecolor('#f8f9fa')
                ax.tick_params(colors='#333333')
                
                st.pyplot(fig)
            
            # Performance breakdown
            st.markdown("---")
            st.markdown("### 📊 Detailed Statistics")
            
            detail_col1, detail_col2, detail_col3 = st.columns(3)
            
            with detail_col1:
                st.markdown("##### 🎯 Wicket-taking")
                st.metric("Total Wickets", int(stats.get("wickets", 0)))
                st.metric("Best Figures", stats.get("best_figures", "N/A") if "best_figures" in stats else "N/A")
                st.metric("4+ Wicket Hauls", int(stats.get("four_wickets", 0)) if "four_wickets" in stats else "N/A")
            
            with detail_col2:
                st.markdown("##### 💰 Economy")
                st.metric("Economy Rate", f"{stats.get('economy', 0):.2f}")
                st.metric("Runs Conceded", int(stats.get("runs_conceded", 0)) if "runs_conceded" in stats else "N/A")
                st.metric("Balls Bowled", int(stats.get("balls_bowled", 0)) if "balls_bowled" in stats else "N/A")
            
            with detail_col3:
                st.markdown("##### 📈 Efficiency")
                st.metric("Bowling Average", f"{stats.get('average', 0):.2f}")
                st.metric("Bowling SR", f"{stats.get('strike_rate', 0):.2f}" if "strike_rate" in stats else "N/A")
                st.metric("Dot Ball %", f"{stats.get('dot_ball_percent', 0):.1f}%" if "dot_ball_percent" in stats else "N/A")
            
            # Historical matchups against batters
            st.markdown("---")
            st.markdown("### ⚔️ Matchups Against Batters")
            player_matchups = matchup_stats[matchup_stats["bowler"] == player].head(10)
            if not player_matchups.empty:
                st.dataframe(
                    player_matchups,
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("No matchup data available for this bowler.")
            
            # Phase-wise analysis (if data available)
            st.markdown("---")
            with st.expander("📊 Phase-wise Performance Analysis"):
                st.markdown("""
                **Bowling performance across match phases:**
                - **Powerplay (0-6)**: New ball conditions, field restrictions
                - **Middle Overs (6-15)**: Building pressure, containment phase
                - **Death Overs (15-20)**: Defensive bowling, yorkers and variations
                
                *Note: Detailed phase-wise statistics require additional data fields*
                """)
                
                # Create a sample phase distribution if data is available
                if all(col in stats.index for col in ['wickets', 'economy', 'matches_played']):
                    st.info(f"💡 {player} has taken {int(stats.get('wickets', 0))} wickets across {int(stats.get('matches_played', 0))} matches with an economy of {stats.get('economy', 0):.2f}")

# ==================================================
# PAGE: ⚔ BATTER VS BOWLER
# ==================================================
elif page == "⚔ Batter vs Bowler":
    st.subheader("⚔️ Batter vs Bowler Head-to-Head")
    st.markdown("Analyze historical performance between specific batter-bowler combinations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        batter = st.selectbox("Select Batter", sorted(batter_stats["batter"].unique()), key="h2h_batter")
    
    with col2:
        bowler = st.selectbox("Select Bowler", sorted(bowler_stats["bowler"].unique()), key="h2h_bowler")
    
    st.markdown("---")
    
    if st.button("🔍 Analyze Matchup", use_container_width=True, type="primary"):
        matchup = matchup_stats[
            (matchup_stats["batter"] == batter) &
            (matchup_stats["bowler"] == bowler)
        ]
        
        if matchup.empty:
            st.warning("⚠️ No historical data available for this matchup.")
            st.info("💡 This could indicate they haven't faced each other in recorded matches.")
        else:
            stats = matchup.iloc[0]
            
            # Display matchup statistics
            st.markdown("### 📊 Matchup Statistics")
            h2h_col1, h2h_col2, h2h_col3, h2h_col4 = st.columns(4)
            
            h2h_col1.metric("Strike Rate", f"{stats.get('strike_rate', 0):.2f}")
            h2h_col2.metric("Dismissals", int(stats.get('dismissals', 0)))
            h2h_col3.metric("Balls Faced", int(stats.get('balls_faced', 0)) if 'balls_faced' in stats else "N/A")
            h2h_col4.metric("Runs Scored", int(stats.get('runs', 0)) if 'runs' in stats else "N/A")
            
            st.markdown("---")
            
            # Full matchup data
            st.markdown("### 📋 Complete Matchup Data")
            st.dataframe(matchup, use_container_width=True, hide_index=True)
            
            # Analysis
            st.markdown("### 🔍 Analysis")
            sr = stats.get('strike_rate', 0)
            dismissals = stats.get('dismissals', 0)
            
            if dismissals > 3:
                st.error(f"🎯 **Bowler Dominance**: {bowler} has dismissed {batter} {dismissals} times - clear advantage to the bowler")
            elif sr > 150:
                st.success(f"💪 **Batter Dominance**: {batter} scores at {sr:.1f} SR against {bowler} - clear advantage to the batter")
            else:
                st.info(f"⚖️ **Balanced Matchup**: Both players have shown competitive performance against each other")

# ==================================================
# PAGE: 🏟 VENUE ANALYSIS
# ==================================================
elif page == "🏟 Venue Analysis":
    st.subheader("🏟️ Venue-Based Performance Analysis")
    st.markdown("Identify batting-friendly conditions and top performers")
    
    # Top performers
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🏆 Top Run Scorers")
        top_batters = batter_stats.nlargest(10, "runs")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(top_batters["batter"], top_batters["runs"], color='#1e88e5')
        ax.set_xlabel("Total Runs", color='#333333')
        ax.set_title("Top 10 Run Scorers", color='#1f1f1f', pad=20)
        ax.tick_params(colors='#333333')
        
        # Styling
        fig.patch.set_facecolor('#ffffff')
        ax.set_facecolor('#f8f9fa')
        ax.spines['bottom'].set_color('#e0e0e0')
        ax.spines['left'].set_color('#e0e0e0')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='x', alpha=0.3, color='#e0e0e0')
        
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        st.markdown("### ⚡ Highest Strike Rates")
        top_sr = batter_stats.nlargest(10, "strike_rate")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(top_sr["batter"], top_sr["strike_rate"], color='#388e3c')
        ax.set_xlabel("Strike Rate", color='#333333')
        ax.set_title("Top 10 Strike Rates", color='#1f1f1f', pad=20)
        ax.tick_params(colors='#333333')
        
        # Styling
        fig.patch.set_facecolor('#ffffff')
        ax.set_facecolor('#f8f9fa')
        ax.spines['bottom'].set_color('#e0e0e0')
        ax.spines['left'].set_color('#e0e0e0')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='x', alpha=0.3, color='#e0e0e0')
        
        plt.tight_layout()
        st.pyplot(fig)
    
    st.markdown("---")
    
    # Venue insights
    st.markdown("### 📊 Statistical Overview")
    overview_col1, overview_col2, overview_col3 = st.columns(3)
    
    overview_col1.metric("Total Players", len(batter_stats))
    overview_col2.metric("Average Strike Rate", f"{batter_stats['strike_rate'].mean():.2f}")
    overview_col3.metric("Total Runs Scored", int(batter_stats['runs'].sum()))

# ==================================================
# PAGE: 📊 MATCH SIMULATION
# ==================================================
elif page == "📊 Match Simulation":
    st.subheader("📊 20-Over Match Simulation")
    st.markdown("Simulate a complete T20 innings with ball-by-ball predictions")
    
    if model is None:
        st.error("Model not loaded. Cannot simulate match.")
    else:
        sim_col1, sim_col2 = st.columns([2, 1])
        
        with sim_col1:
            num_overs = st.slider("Number of Overs", 5, 20, 20, help="Simulate partial or full innings")
        
        with sim_col2:
            st.markdown("")  # Spacing
            st.markdown("")
            simulate_btn = st.button("▶️ Simulate Match", use_container_width=True, type="primary")
        
        if simulate_btn:
            with st.spinner("Simulating match..."):
                runs, wickets, timeline, match_stats = simulate_match(model, num_overs)
            
            # Final score
            st.markdown("---")
            st.markdown("### 🏏 Final Score")
            score_col1, score_col2, score_col3, score_col4 = st.columns(4)
            
            score_col1.metric("Total Runs", runs)
            score_col2.metric("Wickets Lost", wickets)
            score_col3.metric("Run Rate", f"{match_stats['final_run_rate']:.2f}")
            score_col4.metric("Balls Bowled", match_stats['total_balls'])
            
            # Over-by-over summary
            st.markdown("---")
            st.markdown("### 📈 Over-by-Over Summary")
            
            if match_stats['over_stats']:
                over_df = pd.DataFrame(match_stats['over_stats'])
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
                
                # Runs per over
                ax1.plot(over_df['over'], over_df['runs'], marker='o', linewidth=2, color='#1e88e5')
                ax1.fill_between(over_df['over'], over_df['runs'], alpha=0.3, color='#1e88e5')
                ax1.set_xlabel("Over", color='#333333')
                ax1.set_ylabel("Runs", color='#333333')
                ax1.set_title("Runs per Over", color='#1f1f1f')
                ax1.grid(alpha=0.3, color='#e0e0e0')
                ax1.tick_params(colors='#333333')
                
                # Cumulative run rate
                cumulative_runs = over_df['runs'].cumsum()
                cumulative_rr = cumulative_runs / over_df['over']
                ax2.plot(over_df['over'], cumulative_rr, marker='s', linewidth=2, color='#388e3c')
                ax2.set_xlabel("Over", color='#333333')
                ax2.set_ylabel("Run Rate", color='#333333')
                ax2.set_title("Cumulative Run Rate", color='#1f1f1f')
                ax2.grid(alpha=0.3, color='#e0e0e0')
                ax2.tick_params(colors='#333333')
                
                for ax in [ax1, ax2]:
                    fig.patch.set_facecolor('#ffffff')
                    ax.set_facecolor('#f8f9fa')
                    for spine in ax.spines.values():
                        spine.set_color('#e0e0e0')
                
                plt.tight_layout()
                st.pyplot(fig)
            
            # Ball-by-ball timeline
            st.markdown("---")
            st.markdown("### 📜 Ball-by-Ball Commentary")
            
            with st.expander("View Full Timeline", expanded=True):
                # Display in batches for better readability
                for i in range(0, len(timeline), 20):
                    batch = timeline[i:i+20]
                    for event in batch:
                        st.markdown(event)
                    if i + 20 < len(timeline):
                        st.markdown("---")

# ==================================================
# FOOTER
# ==================================================
st.markdown("---")
st.markdown(
    """
    <div style='text-align:center; color:#666666; padding:20px;'>
    <p>Cricket Analytics Dashboard | Powered by Machine Learning</p>
    <p style='font-size:12px;'>Built with Streamlit 🎈 | Data Science 📊 | Random Forest ML 🤖</p>
    </div>
    """,
    unsafe_allow_html=True
)