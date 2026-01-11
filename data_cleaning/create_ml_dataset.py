import pandas as pd
import numpy as np
from pathlib import Path
import sys

"""
ML Dataset Creation Script
==========================
This script combines all cleaned data files into a single comprehensive dataset
suitable for machine learning model training to predict match outcomes.
"""

# Create cleared_data folder path
cleared_data_folder = Path("../cleared_data")
cleared_data_folder.mkdir(exist_ok=True)

print("Creating ML Dataset from cleaned data files...")
print("=" * 60)

# Verify processed files exist
required_files = [
    "matches_processed.csv",
    "team_statistics_aggregated.csv",
    "team_match_statistics_aggregated.csv",
    "standings_processed.csv",
    "scorers_processed.csv",
    "teams_processed.csv"
]

missing_files = []
for file in required_files:
    if not (cleared_data_folder / file).exists():
        missing_files.append(file)

if missing_files:
    print(f"ERROR: Missing required files: {missing_files}")
    print("Please run cleaning_code.py first to generate processed files.")
    sys.exit(1)

# ============================================================================
# 1. LOAD ALL DATASETS
# ============================================================================
print("\n1. Loading processed datasets...")

matches_df = pd.read_csv(cleared_data_folder / "matches_processed.csv")
team_stats_df = pd.read_csv(cleared_data_folder / "team_statistics_aggregated.csv")
team_match_stats_df = pd.read_csv(cleared_data_folder / "team_match_statistics_aggregated.csv")
standings_df = pd.read_csv(cleared_data_folder / "standings_processed.csv")
scorers_df = pd.read_csv(cleared_data_folder / "scorers_processed.csv")
teams_df = pd.read_csv(cleared_data_folder / "teams_processed.csv")

print(f"   ✓ Loaded matches: {len(matches_df)} rows")
print(f"   ✓ Loaded team stats: {len(team_stats_df)} rows")
print(f"   ✓ Loaded team match stats: {len(team_match_stats_df)} rows")
print(f"   ✓ Loaded standings: {len(standings_df)} rows")
print(f"   ✓ Loaded scorers: {len(scorers_df)} rows")
print(f"   ✓ Loaded teams: {len(teams_df)} rows")

# ============================================================================
# 2. PREPARE MATCH DATA AS BASE
# ============================================================================
print("\n2. Preparing match data as base dataset...")

# Start with matches as the base
ml_dataset = matches_df.copy()

# Create target variable: match outcome
# 0 = Away Win, 1 = Draw, 2 = Home Win
# First try to use score_winner, if missing or "Unknown", derive from scores
if 'score_winner' in ml_dataset.columns:
    ml_dataset['match_outcome'] = ml_dataset['score_winner'].map({
        'HOME_TEAM': 2,
        'DRAW': 1,
        'AWAY_TEAM': 0
    })
    print(f"   ✓ Created target variable from score_winner")
    
    # Fill missing outcomes by deriving from scores
    missing_outcomes = ml_dataset['match_outcome'].isna()
    if missing_outcomes.any():
        if 'score_fullTime_home' in ml_dataset.columns and 'score_fullTime_away' in ml_dataset.columns:
            # Derive outcome from scores where score_winner is missing/Unknown
            mask = missing_outcomes & ml_dataset['score_fullTime_home'].notna() & ml_dataset['score_fullTime_away'].notna()
            ml_dataset.loc[mask, 'match_outcome'] = np.where(
                ml_dataset.loc[mask, 'score_fullTime_home'] > ml_dataset.loc[mask, 'score_fullTime_away'], 2,  # Home win
                np.where(
                    ml_dataset.loc[mask, 'score_fullTime_home'] == ml_dataset.loc[mask, 'score_fullTime_away'], 1,  # Draw
                    0  # Away win
                )
            )
            filled_count = mask.sum()
            print(f"   ✓ Derived {filled_count} outcomes from scores where score_winner was missing")
        else:
            print(f"   ⚠ Warning: Cannot derive outcomes - score columns missing")
else:
    # If no score_winner column, derive entirely from scores
    if 'score_fullTime_home' in ml_dataset.columns and 'score_fullTime_away' in ml_dataset.columns:
        ml_dataset['match_outcome'] = np.where(
            ml_dataset['score_fullTime_home'] > ml_dataset['score_fullTime_away'], 2,  # Home win
            np.where(
                ml_dataset['score_fullTime_home'] == ml_dataset['score_fullTime_away'], 1,  # Draw
                0  # Away win
            )
        )
        print(f"   ✓ Created target variable from scores")
    else:
        print("   ⚠ ERROR: Cannot create target variable - missing score_winner and score columns")
        sys.exit(1)

# Drop rows where match_outcome is still NaN (incomplete match data)
initial_rows = len(ml_dataset)
ml_dataset = ml_dataset.dropna(subset=['match_outcome'])
dropped_rows = initial_rows - len(ml_dataset)
if dropped_rows > 0:
    print(f"   ✓ Dropped {dropped_rows} rows with missing match outcomes")
    print(f"   ✓ Remaining matches: {len(ml_dataset)}")

# Drop match_date and related date columns
date_columns_to_drop = ['match_date', 'utcDate', 'date']
date_columns_to_drop = [col for col in date_columns_to_drop if col in ml_dataset.columns]
ml_dataset = ml_dataset.drop(columns=date_columns_to_drop)
if date_columns_to_drop:
    print(f"   ✓ Dropped date columns: {date_columns_to_drop}")

print(f"   ✓ Base dataset: {len(ml_dataset)} matches")

# ============================================================================
# 3. ADD HOME TEAM STATISTICS
# ============================================================================
print("\n3. Adding home team statistics...")

# Merge team statistics for home team
if 'homeTeam_name' in ml_dataset.columns and 'team_name' in team_stats_df.columns:
    team_stats_home = team_stats_df.copy()
    team_stats_home.columns = ['home_' + col if col not in ['team_name', 'team_id'] else col for col in team_stats_home.columns]
    ml_dataset = pd.merge(ml_dataset, team_stats_home, 
                         left_on='homeTeam_name', right_on='team_name', 
                         how='left', suffixes=('', '_home_stats'))
    ml_dataset = ml_dataset.drop(columns=['team_name'], errors='ignore')
    if 'team_id' in ml_dataset.columns and 'team_id_home_stats' not in ml_dataset.columns:
        ml_dataset = ml_dataset.rename(columns={'team_id': 'home_team_id'})
    print("   ✓ Added home team statistics")
else:
    print("   ⚠ Warning: Could not merge home team statistics")

# Merge team match statistics for home team
if 'homeTeam_name' in ml_dataset.columns and 'team_name' in team_match_stats_df.columns:
    team_match_home = team_match_stats_df.copy()
    team_match_home.columns = ['home_' + col if col not in ['team_name', 'team_id'] else col for col in team_match_home.columns]
    ml_dataset = pd.merge(ml_dataset, team_match_home,
                         left_on='homeTeam_name', right_on='team_name',
                         how='left', suffixes=('', '_home_match'))
    ml_dataset = ml_dataset.drop(columns=['team_name'], errors='ignore')
    if 'team_id' in ml_dataset.columns and 'team_id_home_match' not in ml_dataset.columns:
        ml_dataset = ml_dataset.rename(columns={'team_id': 'home_team_id_match'})
    print("   ✓ Added home team match statistics")

# ============================================================================
# 4. ADD AWAY TEAM STATISTICS
# ============================================================================
print("\n4. Adding away team statistics...")

# Merge team statistics for away team
if 'awayTeam_name' in ml_dataset.columns and 'team_name' in team_stats_df.columns:
    team_stats_away = team_stats_df.copy()
    team_stats_away.columns = ['away_' + col if col not in ['team_name', 'team_id'] else col for col in team_stats_away.columns]
    ml_dataset = pd.merge(ml_dataset, team_stats_away,
                         left_on='awayTeam_name', right_on='team_name',
                         how='left', suffixes=('', '_away_stats'))
    ml_dataset = ml_dataset.drop(columns=['team_name'], errors='ignore')
    if 'team_id' in ml_dataset.columns and 'team_id_away_stats' not in ml_dataset.columns:
        ml_dataset = ml_dataset.rename(columns={'team_id': 'away_team_id'})
    print("   ✓ Added away team statistics")
else:
    print("   ⚠ Warning: Could not merge away team statistics")

# Merge team match statistics for away team
if 'awayTeam_name' in ml_dataset.columns and 'team_name' in team_match_stats_df.columns:
    team_match_away = team_match_stats_df.copy()
    team_match_away.columns = ['away_' + col if col not in ['team_name', 'team_id'] else col for col in team_match_away.columns]
    ml_dataset = pd.merge(ml_dataset, team_match_away,
                         left_on='awayTeam_name', right_on='team_name',
                         how='left', suffixes=('', '_away_match'))
    ml_dataset = ml_dataset.drop(columns=['team_name'], errors='ignore')
    if 'team_id' in ml_dataset.columns and 'team_id_away_match' not in ml_dataset.columns:
        ml_dataset = ml_dataset.rename(columns={'team_id': 'away_team_id_match'})
    print("   ✓ Added away team match statistics")

# ============================================================================
# 5. ADD TEAM OFFENSIVE STATISTICS (from scorers - player-based features)
# ============================================================================
print("\n5. Adding player-based team statistics...")

# Calculate comprehensive team offensive stats from player/scorer data
if 'team_name' in scorers_df.columns:
    # Basic totals
    team_offensive = scorers_df.groupby('team_name').agg({
        'goals': ['sum', 'mean', 'count'],  # Total goals, avg per player, number of scorers
        'assists': ['sum', 'mean'],
        'penalties': 'sum'
    }).reset_index()
    
    # Flatten column names
    team_offensive.columns = ['team_name', 'total_team_goals', 'avg_goals_per_player', 'num_scorers',
                              'total_team_assists', 'avg_assists_per_player', 'total_penalties']
    
    # Top scorer statistics (top 3 players per team)
    top_scorers_per_team = scorers_df.groupby('team_name').apply(
        lambda x: x.nlargest(3, 'goals')[['goals', 'assists']].sum()
    ).reset_index()
    top_scorers_per_team.columns = ['team_name', 'top3_goals', 'top3_assists']
    
    # Merge top scorer stats
    team_offensive = pd.merge(team_offensive, top_scorers_per_team, on='team_name', how='left')
    
    # Top scorer individual stats (best player per team)
    top_player_per_team = scorers_df.groupby('team_name').apply(
        lambda x: x.nlargest(1, 'goals')[['goals', 'assists', 'penalties']].iloc[0]
    ).reset_index()
    top_player_per_team.columns = ['team_name', 'top_scorer_goals', 'top_scorer_assists', 'top_scorer_penalties']
    
    # Merge top player stats
    team_offensive = pd.merge(team_offensive, top_player_per_team, on='team_name', how='left')
    
    # Calculate scoring depth (goals from players beyond top scorer)
    team_offensive['scoring_depth'] = team_offensive['total_team_goals'] - team_offensive['top_scorer_goals']
    
    # Calculate contribution ratio (top 3 / total)
    team_offensive['top3_goals_ratio'] = team_offensive['top3_goals'] / (team_offensive['total_team_goals'] + 1e-6)  # Avoid division by zero
    
    # Fill NaN values from merges
    team_offensive = team_offensive.fillna(0)
    
    # Add for home team
    team_offensive_home = team_offensive.copy()
    team_offensive_home.columns = ['home_' + col if col != 'team_name' else col for col in team_offensive_home.columns]
    if 'homeTeam_name' in ml_dataset.columns:
        ml_dataset = pd.merge(ml_dataset, team_offensive_home,
                             left_on='homeTeam_name', right_on='team_name',
                             how='left', suffixes=('', '_home_off'))
        ml_dataset = ml_dataset.drop(columns=['team_name'], errors='ignore')
    
    # Add for away team
    team_offensive_away = team_offensive.copy()
    team_offensive_away.columns = ['away_' + col if col != 'team_name' else col for col in team_offensive_away.columns]
    if 'awayTeam_name' in ml_dataset.columns:
        ml_dataset = pd.merge(ml_dataset, team_offensive_away,
                             left_on='awayTeam_name', right_on='team_name',
                             how='left', suffixes=('', '_away_off'))
        ml_dataset = ml_dataset.drop(columns=['team_name'], errors='ignore')
    
    print("   ✓ Added comprehensive player-based team statistics:")
    print("     - Total goals/assists per team")
    print("     - Top scorer stats (goals, assists, penalties)")
    print("     - Top 3 players combined stats")
    print("     - Scoring depth (goals beyond top scorer)")
    print("     - Number of different scorers per team")
else:
    print("   ⚠ Warning: Could not add player-based statistics")

# ============================================================================
# 6. CREATE DERIVED FEATURES
# ============================================================================
print("\n6. Creating derived features...")

# Goal difference features
if 'home_goalsFor' in ml_dataset.columns and 'away_goalsFor' in ml_dataset.columns:
    ml_dataset['goals_diff'] = ml_dataset['home_goalsFor'] - ml_dataset['away_goalsFor']
    ml_dataset['goals_against_diff'] = ml_dataset['home_goalsAgainst'] - ml_dataset['away_goalsAgainst']
    print("   ✓ Added goal difference features")

# Points difference
if 'home_points' in ml_dataset.columns and 'away_points' in ml_dataset.columns:
    ml_dataset['points_diff'] = ml_dataset['home_points'] - ml_dataset['away_points']
    print("   ✓ Added points difference feature")

# Position difference (if available)
if 'home_position' in ml_dataset.columns and 'away_position' in ml_dataset.columns:
    ml_dataset['position_diff'] = ml_dataset['away_position'] - ml_dataset['home_position']  # Lower position = better
    print("   ✓ Added position difference feature")

# Win rate difference
if 'home_won' in ml_dataset.columns and 'away_won' in ml_dataset.columns:
    if 'home_playedGames' in ml_dataset.columns or 'home_points' in ml_dataset.columns:
        # Approximate: wins/total games (if we have playedGames)
        pass  # Can be calculated if we have games played
    print("   ✓ Win statistics available")

# Home/Away performance ratio
if 'home_home_wins' in ml_dataset.columns and 'away_away_wins' in ml_dataset.columns:
    ml_dataset['home_advantage_metric'] = ml_dataset['home_home_wins'] - ml_dataset['away_away_wins']
    print("   ✓ Added home advantage metric")

# Player quality differences
if 'home_top_scorer_goals' in ml_dataset.columns and 'away_top_scorer_goals' in ml_dataset.columns:
    ml_dataset['top_scorer_goals_diff'] = ml_dataset['home_top_scorer_goals'] - ml_dataset['away_top_scorer_goals']
    ml_dataset['top3_goals_diff'] = ml_dataset['home_top3_goals'] - ml_dataset['away_top3_goals']
    ml_dataset['scoring_depth_diff'] = ml_dataset['home_scoring_depth'] - ml_dataset['away_scoring_depth']
    ml_dataset['num_scorers_diff'] = ml_dataset['home_num_scorers'] - ml_dataset['away_num_scorers']
    print("   ✓ Added player quality difference features")

# Match score features (if available)
if 'score_fullTime_home' in ml_dataset.columns and 'score_fullTime_away' in ml_dataset.columns:
    ml_dataset['total_goals'] = ml_dataset['score_fullTime_home'] + ml_dataset['score_fullTime_away']
    ml_dataset['goal_margin'] = ml_dataset['score_fullTime_home'] - ml_dataset['score_fullTime_away']
    print("   ✓ Added match score features")

# ============================================================================
# 7. CLEAN UP AND FINALIZE
# ============================================================================
print("\n7. Finalizing dataset...")

# Drop redundant columns that won't help ML
columns_to_drop = ['homeTeam_crest', 'awayTeam_crest', 'homeTeam_id', 'awayTeam_id',
                   'home_team_id', 'away_team_id', 'match_id', 'competition_id',
                   'season_id', 'season_startDate', 'season_endDate', 'season_currentMatchday',
                   'score_winner']  # Drop score_winner after creating match_outcome

# Only drop if they exist
columns_to_drop = [col for col in columns_to_drop if col in ml_dataset.columns]
ml_dataset = ml_dataset.drop(columns=columns_to_drop)

# Handle any remaining NaN values in numeric columns
numeric_cols = ml_dataset.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    if col != 'match_outcome':  # Don't fill target variable
        ml_dataset[col] = ml_dataset[col].fillna(ml_dataset[col].median())

# Fill text columns
text_cols = ml_dataset.select_dtypes(include=['object']).columns
for col in text_cols:
    ml_dataset[col] = ml_dataset[col].fillna('Unknown')

print(f"   ✓ Final dataset shape: {ml_dataset.shape[0]} rows, {ml_dataset.shape[1]} columns")

# ============================================================================
# 8. SAVE DATASET
# ============================================================================
output_file = cleared_data_folder / "ml_dataset.csv"
ml_dataset.to_csv(output_file, index=False)

print("\n" + "=" * 60)
print("ML Dataset created successfully!")
print("=" * 60)
print(f"\nDataset saved to: {output_file.absolute()}")
print(f"Total rows: {len(ml_dataset)}")
print(f"Total columns: {len(ml_dataset.columns)}")

if 'match_outcome' in ml_dataset.columns:
    print(f"\nTarget variable distribution:")
    print(ml_dataset['match_outcome'].value_counts().sort_index())
    print("\n(0 = Away Win, 1 = Draw, 2 = Home Win)")

print(f"\nFeature columns: {len(ml_dataset.columns) - 1}")  # Minus target variable
print("\nFirst few columns:")
print(ml_dataset.columns[:10].tolist())

