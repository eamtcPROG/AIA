import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from pathlib import Path

"""
IMPORTANT NOTE ON HANDLING ZEROS:
===================================
In football data, 0 is a VALID value (0 goals, 0-0 draws, 0 assists, etc.)
pandas .fillna(0) ONLY fills actual NaN/missing values, NOT existing 0s.
This means:
- Existing 0 values remain unchanged (legitimate zero scores/stats)
- Only truly missing/NaN values get filled with 0
- This preserves all legitimate football scores and statistics
"""

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Create cleared_data folder at the same level as data folder
cleared_data_folder = Path("../cleared_data")
cleared_data_folder.mkdir(exist_ok=True)

print("Starting data cleaning and analysis...")
print("=" * 60)

# Verify data files exist (one level up)
required_files = ["../data/teams.csv", "../data/standings.csv", "../data/scorers.csv", "../data/matches.csv"]
missing_files = [f for f in required_files if not Path(f).exists()]
if missing_files:
    print(f"ERROR: Missing required files: {missing_files}")
    sys.exit(1)

# ============================================================================
# 1. TEAMS DATA CLEANING
# ============================================================================
print("\n1. Processing teams.csv...")
teams_df = pd.read_csv("../data/teams.csv")

# Drop columns with raw JSON data and redundant information
columns_to_drop_teams = ['raw_json', 'runningCompetitions_json']
teams_df_clean = teams_df.drop(columns=[col for col in columns_to_drop_teams if col in teams_df.columns])

# Handle missing values
# IMPORTANT: fillna(0) ONLY fills actual NaN/missing values, NOT existing 0s
# For marketValue: converting to numeric and filling missing with 0
# (Note: If a team truly has 0 market value vs missing data, they're treated the same here)
if 'marketValue' in teams_df_clean.columns:
    teams_df_clean['marketValue'] = pd.to_numeric(teams_df_clean['marketValue'], errors='coerce').fillna(0)

# Fill missing text values with 'Unknown'
text_columns = teams_df_clean.select_dtypes(include=['object']).columns
for col in text_columns:
    teams_df_clean[col] = teams_df_clean[col].fillna('Unknown')

# Save cleaned teams data
teams_df_clean.to_csv(cleared_data_folder / "teams_processed.csv", index=False)
print(f"   ✓ Cleaned teams data: {len(teams_df_clean)} rows, {len(teams_df_clean.columns)} columns")
print(f"   ✓ Removed columns: {columns_to_drop_teams}")

# ============================================================================
# 2. STANDINGS DATA CLEANING
# ============================================================================
print("\n2. Processing standings.csv...")
standings_df = pd.read_csv("../data/standings.csv")

# Drop raw JSON columns
columns_to_drop_standings = ['raw_row_json', 'raw_team_json']
standings_df_clean = standings_df.drop(columns=[col for col in columns_to_drop_standings if col in standings_df.columns])

# Handle missing values
# IMPORTANT: fillna(0) ONLY fills actual NaN/missing values, NOT existing 0s
# In football, 0 is valid (0 goals, 0 wins, 0 losses, 0 points, etc.)
# fillna() preserves all existing 0 values and only fills truly missing data
numeric_columns = standings_df_clean.select_dtypes(include=[np.number]).columns
for col in numeric_columns:
    standings_df_clean[col] = standings_df_clean[col].fillna(0)

# Fill missing text values
text_columns = standings_df_clean.select_dtypes(include=['object']).columns
for col in text_columns:
    if col not in ['form']:  # form can be empty
        standings_df_clean[col] = standings_df_clean[col].fillna('Unknown')

# Save cleaned standings data
standings_df_clean.to_csv(cleared_data_folder / "standings_processed.csv", index=False)
print(f"   ✓ Cleaned standings data: {len(standings_df_clean)} rows, {len(standings_df_clean.columns)} columns")

# ============================================================================
# 3. SCORERS DATA CLEANING
# ============================================================================
print("\n3. Processing scorers.csv...")
scorers_df = pd.read_csv("../data/scorers.csv")

# Drop raw JSON columns
columns_to_drop_scorers = ['raw_player_json', 'raw_scorer_json', 'raw_team_json']
scorers_df_clean = scorers_df.drop(columns=[col for col in columns_to_drop_scorers if col in scorers_df.columns])

# Handle missing values
# IMPORTANT: fillna(0) ONLY fills actual NaN/missing values, NOT existing 0s
# In football, 0 is a valid value (0 goals, 0 assists, 0 penalties are legitimate)
# pd.to_numeric with errors='coerce' converts invalid text to NaN, then fillna(0) handles missing values
# All existing 0 values remain unchanged - this only fills truly missing data
scorers_df_clean['goals'] = pd.to_numeric(scorers_df_clean['goals'], errors='coerce').fillna(0)
scorers_df_clean['assists'] = pd.to_numeric(scorers_df_clean['assists'], errors='coerce').fillna(0)
scorers_df_clean['penalties'] = pd.to_numeric(scorers_df_clean['penalties'], errors='coerce').fillna(0)

# Fill missing text values
text_columns = scorers_df_clean.select_dtypes(include=['object']).columns
for col in text_columns:
    scorers_df_clean[col] = scorers_df_clean[col].fillna('Unknown')

# Save cleaned scorers data
scorers_df_clean.to_csv(cleared_data_folder / "scorers_processed.csv", index=False)
print(f"   ✓ Cleaned scorers data: {len(scorers_df_clean)} rows, {len(scorers_df_clean.columns)} columns")

# ============================================================================
# 4. MATCHES DATA CLEANING
# ============================================================================
print("\n4. Processing matches.csv...")
matches_df = pd.read_csv("../data/matches.csv")

# Drop raw JSON columns
columns_to_drop_matches = ['raw_json', 'referees_json']
matches_df_clean = matches_df.drop(columns=[col for col in columns_to_drop_matches if col in matches_df.columns])

# Handle missing values
# IMPORTANT: fillna(0) ONLY fills actual NaN/missing values, NOT existing 0s
# In football, 0 is valid for scores (0-0 draws are common), bookings count, etc.
# fillna() preserves all existing 0 values (like legitimate 0-0 scores)
numeric_columns = matches_df_clean.select_dtypes(include=[np.number]).columns
for col in numeric_columns:
    # Skip odds columns - they should stay NaN if missing (0 odds doesn't make sense)
    if 'odds' in col.lower():
        continue  # Don't fill odds with 0, leave as NaN if missing
    # For scores, bookings, and other counts: 0 is valid, fillna(0) is safe
    matches_df_clean[col] = matches_df_clean[col].fillna(0)

# Fill missing text values
text_columns = matches_df_clean.select_dtypes(include=['object']).columns
for col in text_columns:
    matches_df_clean[col] = matches_df_clean[col].fillna('Unknown')

# Convert date column to datetime
if 'utcDate' in matches_df_clean.columns:
    matches_df_clean['utcDate'] = pd.to_datetime(matches_df_clean['utcDate'], errors='coerce')

# Save cleaned matches data
matches_df_clean.to_csv(cleared_data_folder / "matches_processed.csv", index=False)
print(f"   ✓ Cleaned matches data: {len(matches_df_clean)} rows, {len(matches_df_clean.columns)} columns")

# ============================================================================
# 5. AGGREGATIONS
# ============================================================================
print("\n5. Creating aggregated datasets...")

# Aggregation 1: Team Statistics Summary
team_stats = standings_df_clean.groupby('team_name').agg({
    'points': 'max',
    'won': 'max',
    'draw': 'max',
    'lost': 'max',
    'goalsFor': 'max',
    'goalsAgainst': 'max',
    'goalDifference': 'max',
    'position': 'min'  # Best position
}).reset_index()
team_stats = team_stats.sort_values('points', ascending=False)
team_stats.to_csv(cleared_data_folder / "team_statistics_aggregated.csv", index=False)
print("   ✓ Created team_statistics_aggregated.csv")

# Aggregation 2: Top Scorers by Season
top_scorers = scorers_df_clean.groupby(['season_id', 'player_name']).agg({
    'goals': 'sum',
    'assists': 'sum',
    'penalties': 'sum',
    'team_name': 'first'
}).reset_index()
top_scorers = top_scorers.sort_values(['season_id', 'goals'], ascending=[True, False])
top_scorers.to_csv(cleared_data_folder / "top_scorers_aggregated.csv", index=False)
print("   ✓ Created top_scorers_aggregated.csv")

# Aggregation 3: Match Results by Team (Home and Away)
matches_df_clean['homeWin'] = (matches_df_clean['score_winner'] == 'HOME_TEAM').astype(int)
matches_df_clean['awayWin'] = (matches_df_clean['score_winner'] == 'AWAY_TEAM').astype(int)
matches_df_clean['draw'] = (matches_df_clean['score_winner'] == 'DRAW').astype(int)

home_stats = matches_df_clean.groupby('homeTeam_name').agg({
    'homeWin': 'sum',
    'draw': 'sum',
    'score_fullTime_home': 'sum',
    'score_fullTime_away': 'sum'
}).reset_index()
home_stats.columns = ['team_name', 'home_wins', 'home_draws', 'goals_for_home', 'goals_against_home']

away_stats = matches_df_clean.groupby('awayTeam_name').agg({
    'awayWin': 'sum',
    'draw': 'sum',
    'score_fullTime_away': 'sum',
    'score_fullTime_home': 'sum'
}).reset_index()
away_stats.columns = ['team_name', 'away_wins', 'away_draws', 'goals_for_away', 'goals_against_away']

team_match_stats = pd.merge(home_stats, away_stats, on='team_name', how='outer').fillna(0)
team_match_stats['total_wins'] = team_match_stats['home_wins'] + team_match_stats['away_wins']
team_match_stats['total_goals_for'] = team_match_stats['goals_for_home'] + team_match_stats['goals_for_away']
team_match_stats['total_goals_against'] = team_match_stats['goals_against_home'] + team_match_stats['goals_against_away']
team_match_stats = team_match_stats.sort_values('total_wins', ascending=False)
team_match_stats.to_csv(cleared_data_folder / "team_match_statistics_aggregated.csv", index=False)
print("   ✓ Created team_match_statistics_aggregated.csv")

# Aggregation 4: Goals per Match Day
if 'matchday' in matches_df_clean.columns and matches_df_clean['matchday'].notna().any():
    goals_per_matchday = matches_df_clean.groupby('matchday').agg({
        'score_fullTime_home': 'sum',
        'score_fullTime_away': 'sum',
        'match_id': 'count'
    }).reset_index()
    goals_per_matchday.columns = ['matchday', 'total_goals_home', 'total_goals_away', 'matches_count']
    goals_per_matchday['total_goals'] = goals_per_matchday['total_goals_home'] + goals_per_matchday['total_goals_away']
    goals_per_matchday['avg_goals_per_match'] = goals_per_matchday['total_goals'] / goals_per_matchday['matches_count']
    goals_per_matchday = goals_per_matchday[goals_per_matchday['matchday'] > 0]  # Filter valid matchdays
    goals_per_matchday.to_csv(cleared_data_folder / "goals_per_matchday_aggregated.csv", index=False)
    print("   ✓ Created goals_per_matchday_aggregated.csv")

print("\n" + "=" * 60)
print("Data cleaning and aggregation completed!")
print("=" * 60)

# ============================================================================
# 6. VISUALIZATIONS
# ============================================================================
print("\n6. Creating visualizations...")
plots_folder = cleared_data_folder / "plots"
plots_folder.mkdir(exist_ok=True)

# Plot 1: Top Teams by Points
plt.figure(figsize=(12, 8))
top_teams = team_stats.head(15)
plt.barh(range(len(top_teams)), top_teams['points'], color='steelblue')
plt.yticks(range(len(top_teams)), top_teams['team_name'])
plt.xlabel('Points', fontsize=12)
plt.title('Top 15 Teams by Points', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(plots_folder / "top_teams_by_points.png", dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Created top_teams_by_points.png")

# Plot 2: Goals For vs Goals Against
plt.figure(figsize=(10, 8))
top_20 = team_stats.head(20)
plt.scatter(top_20['goalsAgainst'], top_20['goalsFor'], s=100, alpha=0.6, color='coral')
for i, row in top_20.iterrows():
    plt.annotate(row['team_name'], (row['goalsAgainst'], row['goalsFor']), 
                fontsize=8, alpha=0.7)
plt.xlabel('Goals Against', fontsize=12)
plt.ylabel('Goals For', fontsize=12)
plt.title('Goals For vs Goals Against (Top 20 Teams)', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(plots_folder / "goals_for_vs_against.png", dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Created goals_for_vs_against.png")

# Plot 3: Top 15 Scorers
plt.figure(figsize=(12, 8))
top_scorers_plot = scorers_df_clean.nlargest(15, 'goals')
plt.barh(range(len(top_scorers_plot)), top_scorers_plot['goals'], color='darkgreen')
plt.yticks(range(len(top_scorers_plot)), 
          [f"{row['player_name']} ({row['team_name']})" for _, row in top_scorers_plot.iterrows()])
plt.xlabel('Goals', fontsize=12)
plt.title('Top 15 Goal Scorers', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(plots_folder / "top_scorers.png", dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Created top_scorers.png")

# Plot 4: Goals and Assists Comparison
plt.figure(figsize=(12, 8))
top_players = scorers_df_clean.nlargest(15, 'goals')
x = np.arange(len(top_players))
width = 0.35
plt.bar(x - width/2, top_players['goals'], width, label='Goals', color='darkblue', alpha=0.8)
plt.bar(x + width/2, top_players['assists'], width, label='Assists', color='orange', alpha=0.8)
plt.xlabel('Players', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.title('Top 15 Players: Goals vs Assists', fontsize=14, fontweight='bold')
plt.xticks(x, [name[:20] for name in top_players['player_name']], rotation=45, ha='right')
plt.legend()
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(plots_folder / "goals_vs_assists.png", dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Created goals_vs_assists.png")

# Plot 5: Win-Draw-Loss Distribution
plt.figure(figsize=(10, 8))
top_10 = team_stats.head(10)
x = np.arange(len(top_10))
width = 0.25
plt.bar(x - width, top_10['won'], width, label='Wins', color='green', alpha=0.8)
plt.bar(x, top_10['draw'], width, label='Draws', color='gray', alpha=0.8)
plt.bar(x + width, top_10['lost'], width, label='Losses', color='red', alpha=0.8)
plt.xlabel('Teams', fontsize=12)
plt.ylabel('Number of Games', fontsize=12)
plt.title('Win-Draw-Loss Distribution (Top 10 Teams)', fontsize=14, fontweight='bold')
plt.xticks(x, [name[:15] for name in top_10['team_name']], rotation=45, ha='right')
plt.legend()
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(plots_folder / "win_draw_loss_distribution.png", dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Created win_draw_loss_distribution.png")

# Plot 6: Goals per Matchday (if data available)
if 'goals_per_matchday_aggregated.csv' in os.listdir(cleared_data_folder):
    goals_per_matchday = pd.read_csv(cleared_data_folder / "goals_per_matchday_aggregated.csv")
    if len(goals_per_matchday) > 0:
        plt.figure(figsize=(14, 6))
        plt.plot(goals_per_matchday['matchday'], goals_per_matchday['avg_goals_per_match'], 
                marker='o', linewidth=2, markersize=6, color='purple')
        plt.xlabel('Matchday', fontsize=12)
        plt.ylabel('Average Goals per Match', fontsize=12)
        plt.title('Average Goals per Match by Matchday', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_folder / "goals_per_matchday.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("   ✓ Created goals_per_matchday.png")

# Plot 7: Team Performance Heatmap
plt.figure(figsize=(12, 8))
top_15 = team_stats.head(15)
heatmap_data = top_15[['won', 'draw', 'lost', 'goalsFor', 'goalsAgainst', 'goalDifference']].T
heatmap_data.columns = [name[:20] for name in top_15['team_name']]
sns.heatmap(heatmap_data, annot=True, fmt='.0f', cmap='RdYlGn', center=0, 
            cbar_kws={'label': 'Value'}, linewidths=0.5)
plt.title('Team Performance Heatmap (Top 15 Teams)', fontsize=14, fontweight='bold', pad=20)
plt.ylabel('Metrics', fontsize=12)
plt.xlabel('Teams', fontsize=12)
plt.tight_layout()
plt.savefig(plots_folder / "team_performance_heatmap.png", dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Created team_performance_heatmap.png")

# Plot 8: Goal Difference Distribution
plt.figure(figsize=(10, 6))
plt.hist(team_stats['goalDifference'], bins=30, color='steelblue', edgecolor='black', alpha=0.7)
plt.xlabel('Goal Difference', fontsize=12)
plt.ylabel('Number of Teams', fontsize=12)
plt.title('Distribution of Goal Difference Across Teams', fontsize=14, fontweight='bold')
plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Goal Difference')
plt.legend()
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(plots_folder / "goal_difference_distribution.png", dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Created goal_difference_distribution.png")

# Plot 9: Boxplot - Team Performance Metrics Distribution
plt.figure(figsize=(12, 8))
boxplot_data = team_stats[['won', 'draw', 'lost', 'goalsFor', 'goalsAgainst', 'goalDifference']]
boxplot_data.columns = ['Wins', 'Draws', 'Losses', 'Goals For', 'Goals Against', 'Goal Diff']
sns.boxplot(data=boxplot_data, palette='Set2')
plt.title('Distribution of Team Performance Metrics', fontsize=14, fontweight='bold')
plt.ylabel('Count/Value', fontsize=12)
plt.xlabel('Metrics', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(plots_folder / "team_performance_boxplot.png", dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Created team_performance_boxplot.png")

# Plot 10: Boxplot - Goals Distribution by Position
if 'position' in standings_df_clean.columns and standings_df_clean['position'].notna().any():
    plt.figure(figsize=(14, 8))
    # Get top, middle, bottom third teams by position
    top_teams_pos = standings_df_clean[(standings_df_clean['position'] <= 7) & (standings_df_clean['position'] > 0)]  # Top 7
    mid_teams_pos = standings_df_clean[(standings_df_clean['position'] > 7) & (standings_df_clean['position'] <= 14)]  # Middle
    bot_teams_pos = standings_df_clean[standings_df_clean['position'] > 14]  # Bottom

    box_data = []
    box_labels = []
    colors = ['lightgreen', 'lightblue', 'lightcoral']
    if len(top_teams_pos) > 0:
        box_data.append(top_teams_pos['goalsFor'].values)
        box_labels.append('Top 7 Teams')
    if len(mid_teams_pos) > 0:
        box_data.append(mid_teams_pos['goalsFor'].values)
        box_labels.append('Middle Teams (8-14)')
    if len(bot_teams_pos) > 0:
        box_data.append(bot_teams_pos['goalsFor'].values)
        box_labels.append('Bottom Teams (15+)')

    if len(box_data) > 0:
        bp = plt.boxplot(box_data, labels=box_labels, patch_artist=True,
                        medianprops=dict(color='red', linewidth=2),
                        whiskerprops=dict(color='black', linewidth=1.5),
                        capprops=dict(color='black', linewidth=1.5))
        # Color the boxes
        for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        plt.title('Goals Scored Distribution by League Position', fontsize=14, fontweight='bold')
        plt.ylabel('Goals For', fontsize=12)
        plt.xlabel('Team Position Category', fontsize=12)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(plots_folder / "goals_by_position_boxplot.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("   ✓ Created goals_by_position_boxplot.png")

# Plot 11: Boxplot - Player Goals and Assists Comparison
plt.figure(figsize=(10, 8))
player_stats = scorers_df_clean[['goals', 'assists']].copy()
player_stats.columns = ['Goals', 'Assists']
sns.boxplot(data=player_stats, palette=['darkgreen', 'orange'])
plt.title('Distribution of Goals and Assists for All Players', fontsize=14, fontweight='bold')
plt.ylabel('Count', fontsize=12)
plt.xlabel('Metric', fontsize=12)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(plots_folder / "player_stats_boxplot.png", dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Created player_stats_boxplot.png")

# Plot 12: Boxplot - Match Scores Distribution (Home vs Away)
plt.figure(figsize=(12, 8))
match_scores = pd.DataFrame({
    'Home Goals': matches_df_clean['score_fullTime_home'].values,
    'Away Goals': matches_df_clean['score_fullTime_away'].values
})
# Remove any NaN values for boxplot
match_scores = match_scores.dropna()
if len(match_scores) > 0:
    sns.boxplot(data=match_scores, palette=['lightcoral', 'lightblue'])
    plt.title('Distribution of Goals Scored: Home vs Away Teams', fontsize=14, fontweight='bold')
    plt.ylabel('Goals Scored', fontsize=12)
    plt.xlabel('Team Type', fontsize=12)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(plots_folder / "home_away_goals_boxplot.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✓ Created home_away_goals_boxplot.png")

print("\n" + "=" * 60)
print("All visualizations created successfully!")
print("=" * 60)
print(f"\nCleaned files saved in: {cleared_data_folder.absolute()}")
print(f"Plots saved in: {plots_folder.absolute()}")
print("\nSummary:")
print(f"  - Processed CSV files: 4")
print(f"  - Aggregated datasets: 4")
print(f"  - Visualizations: 12 (8 regular + 4 boxplots)")

