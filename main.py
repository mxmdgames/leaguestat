import streamlit as st
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from datetime import datetime
from dotenv import load_dotenv
import time
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# Load API Key
load_dotenv()
RIOT_API_KEY = st.secrets.get('RIOT_API_KEY') or os.getenv("RIOT_API_KEY")

# Constants
QUEUE_IDS = {
    420: "Ranked Solo/Duo",
    440: "Ranked Flex",
    400: "Draft Pick",
    430: "Blind Pick",
    450: "ARAM",
}

TIER_BENCHMARKS = {
    'Iron': {
        'KDA': 0.3, 'CS/Min': 0.25, 'Vision': 0.2,
        'Damage': 0.3, 'Gold': 0.3, 'KP%': 0.4, 'Dmg%': 0.35
    },
    'Bronze': {
        'KDA': 0.4, 'CS/Min': 0.35, 'Vision': 0.3,
        'Damage': 0.4, 'Gold': 0.4, 'KP%': 0.5, 'Dmg%': 0.4
    },
    'Silver': {
        'KDA': 0.5, 'CS/Min': 0.45, 'Vision': 0.4,
        'Damage': 0.5, 'Gold': 0.5, 'KP%': 0.55, 'Dmg%': 0.45
    },
    'Gold': {
        'KDA': 0.6, 'CS/Min': 0.55, 'Vision': 0.5,
        'Damage': 0.6, 'Gold': 0.6, 'KP%': 0.6, 'Dmg%': 0.5
    },
    'Platinum': {
        'KDA': 0.7, 'CS/Min': 0.65, 'Vision': 0.6,
        'Damage': 0.7, 'Gold': 0.7, 'KP%': 0.65, 'Dmg%': 0.55
    },
    'Diamond': {
        'KDA': 0.8, 'CS/Min': 0.75, 'Vision': 0.7,
        'Damage': 0.8, 'Gold': 0.8, 'KP%': 0.7, 'Dmg%': 0.6
    },
    'Master': {
        'KDA': 0.85, 'CS/Min': 0.8, 'Vision': 0.75,
        'Damage': 0.85, 'Gold': 0.85, 'KP%': 0.75, 'Dmg%': 0.65
    },
    'Grandmaster': {
        'KDA': 0.9, 'CS/Min': 0.85, 'Vision': 0.8,
        'Damage': 0.9, 'Gold': 0.9, 'KP%': 0.8, 'Dmg%': 0.7
    },
    'Challenger': {
        'KDA': 1.0, 'CS/Min': 0.95, 'Vision': 0.9,
        'Damage': 1.0, 'Gold': 1.0, 'KP%': 0.85, 'Dmg%': 0.75
    }
}


class RiotAPIError(Exception):
    pass


# API Functions
def make_api_request(url, params=None):
    if params is None:
        params = {}
    params["api_key"] = RIOT_API_KEY

    try:
        response = requests.get(url, params=params)
        if response.status_code == 429:
            retry_after = int(response.headers.get('Retry-After', 5))
            st.warning(f"Rate limit exceeded. Retrying in {retry_after} seconds...")
            time.sleep(retry_after)
            return make_api_request(url, params)
        if response.status_code != 200:
            raise RiotAPIError(f"API Error {response.status_code}: {response.text}")
        return response.json()
    except requests.exceptions.RequestException as e:
        raise RiotAPIError(f"Network error: {e}")


def get_account_info(game_name, tag_line):
    url = f"https://americas.api.riotgames.com/riot/account/v1/accounts/by-riot-id/{game_name}/{tag_line}"
    return make_api_request(url)


def get_matches(puuid, count=20):
    url = f"https://americas.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids"
    return make_api_request(url, {"start": 0, "count": count})


def get_match_details(match_id):
    url = f"https://americas.api.riotgames.com/lol/match/v5/matches/{match_id}"
    return make_api_request(url)


def collect_player_stats(match_details, puuid):
    for p in match_details['info']['participants']:
        if p['puuid'] == puuid:
            duration = match_details['info']['gameDuration'] / 60
            cs = p['totalMinionsKilled'] + p['neutralMinionsKilled']

            team_id = p['teamId']
            team_kills = sum(pl['kills'] for pl in match_details['info']['participants'] if pl['teamId'] == team_id)
            kp = (p['kills'] + p['assists']) / team_kills * 100 if team_kills > 0 else 0

            team_dmg = sum(pl['totalDamageDealtToChampions'] for pl in match_details['info']['participants'] if
                           pl['teamId'] == team_id)
            dmg_share = p['totalDamageDealtToChampions'] / team_dmg * 100 if team_dmg > 0 else 0

            return {
                'match_id': match_details['metadata']['matchId'],
                'champion': p['championName'],
                'position': p.get('individualPosition', 'Unknown'),
                'kills': p['kills'],
                'deaths': p['deaths'],
                'assists': p['assists'],
                'kda': (p['kills'] + p['assists']) / max(1, p['deaths']),
                'cs': cs,
                'cs_per_minute': cs / duration,
                'damage': p['totalDamageDealtToChampions'],
                'damage_share': dmg_share,
                'gold': p['goldEarned'],
                'vision_score': p['visionScore'],
                'kill_participation': kp,
                'game_duration': duration,
                'win': p['win'],
                'queue': QUEUE_IDS.get(match_details['info']['queueId'], 'Unknown'),
                'game_date': datetime.fromtimestamp(match_details['info']['gameCreation'] / 1000),
            }
    return None


# Analysis Functions
def analyze_stats(stats_list):
    df = pd.DataFrame(stats_list)
    if df.empty:
        return None

    champion_stats = df.groupby('champion').agg({
        'win': 'mean',
        'kda': 'mean',
        'match_id': 'count'
    }).reset_index()
    champion_stats['win_rate'] = champion_stats['win'] * 100

    position_stats = df.groupby('position')['win'].agg(['sum', 'count']).reset_index()
    position_stats['win_rate'] = (position_stats['sum'] / position_stats['count'] * 100).round(1)

    return {
        'df': df,
        'overall': {
            'game_count': len(df),
            'win_count': df['win'].sum(),
            'win_rate': (df['win'].mean() * 100).round(1),
            'avg_kda': df['kda'].mean().round(2),
            'avg_cs_per_min': df['cs_per_minute'].mean().round(1),
            'avg_vision_score': df['vision_score'].mean().round(1),
            'avg_damage': int(df['damage'].mean()),
            'avg_kill_participation': df['kill_participation'].mean().round(1),
            'avg_damage_share': df['damage_share'].mean().round(1),
            'avg_gold': int(df['gold'].mean()),
        },
        'champions': df['champion'].value_counts().head(3).to_dict(),
        'champion_win_rates': champion_stats.set_index('champion')['win_rate'].to_dict(),
        'positions': df['position'].value_counts().to_dict(),
        'position_win_rates': position_stats.set_index('position')['win_rate'].to_dict(),
        'queues': df['queue'].value_counts().to_dict(),
    }


# Visualization Functions
def normalize_to_percent(value, metric):
    """Convert raw values to 0-1 scale based on realistic max values"""
    max_values = {
        'KDA': 10,  # 10:1 KDA is exceptional
        'CS/Min': 12,  # 12 CS/min is near perfect
        'Vision': 100,  # 100 vision score in a long game
        'Damage': 40000,  # 40k damage in a long game
        'Gold': 40000,  # 40k gold in a long game
        'KP%': 100,  # 100% kill participation
        'Dmg%': 50  # 50% damage share is very high
    }
    return min(value / max_values[metric], 1.0)


def draw_comparison_radar(player_stats, compare_tier):
    """Create radar chart comparing player to selected tier"""
    # Player stats normalized to 0-1 scale
    player_normalized = {
        'KDA': normalize_to_percent(player_stats['avg_kda'], 'KDA'),
        'CS/Min': normalize_to_percent(player_stats['avg_cs_per_min'], 'CS/Min'),
        'Vision': normalize_to_percent(player_stats['avg_vision_score'], 'Vision'),
        'Damage': normalize_to_percent(player_stats['avg_damage'], 'Damage'),
        'Gold': normalize_to_percent(player_stats['avg_gold'], 'Gold'),
        'KP%': normalize_to_percent(player_stats['avg_kill_participation'], 'KP%'),
        'Dmg%': normalize_to_percent(player_stats['avg_damage_share'], 'Dmg%')
    }

    # Get benchmark data
    benchmark = TIER_BENCHMARKS[compare_tier]

    # Prepare data for radar chart
    categories = list(player_normalized.keys())
    player_values = list(player_normalized.values()) + [player_normalized['KDA']]
    benchmark_values = list(benchmark.values()) + [benchmark['KDA']]

    # Create angles for radar chart
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    # Initialize radar chart
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    # Draw axis lines
    plt.xticks(angles[:-1], categories, color='grey', size=10)
    ax.set_rlabel_position(0)
    plt.yticks([0.25, 0.5, 0.75, 1], ["25%", "50%", "75%", "100%"], color="grey", size=8)
    plt.ylim(0, 1)

    # Plot player data
    ax.plot(angles, player_values, linewidth=2, linestyle='solid',
            color='#1f77b4', label='Your Stats')
    ax.fill(angles, player_values, '#1f77b4', alpha=0.25)

    # Plot benchmark data
    ax.plot(angles, benchmark_values, linewidth=2, linestyle='solid',
            color='#ff7f0e', label=f'{compare_tier} Average')
    ax.fill(angles, benchmark_values, '#ff7f0e', alpha=0.1)

    # Add title and legend
    plt.title(f'Your Performance vs {compare_tier} Averages\n(Normalized to max expected values)',
              size=14, y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

    return fig


def visualize_stats(df):
    """Generate visualizations matching the original analysis"""
    sns.set_style("darkgrid")
    fig, axs = plt.subplots(3, 2, figsize=(22, 16))
    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.titlesize': 16
    })

    # 1. KDA Trend with Win/Loss Markers
    df_sorted = df.sort_values('game_date')
    df_sorted['game_number'] = range(1, len(df_sorted) + 1)
    ax = axs[0, 0]
    sns.lineplot(data=df_sorted, x='game_number', y='kda', ax=ax,
                 color='purple', marker='o')
    colors = df_sorted['win'].map({True: 'green', False: 'red'})
    ax.scatter(df_sorted['game_number'], df_sorted['kda'],
               c=colors, s=100, zorder=5)
    ax.legend(handles=[
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor='green', markersize=10, label='Win'),
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor='red', markersize=10, label='Loss')
    ])
    ax.set(title='KDA Trend with Win/Loss',
           xlabel='Game Number', ylabel='KDA Ratio')

    # 2. Top Champion Performance
    ax = axs[0, 1]
    champ_stats = df.groupby('champion').agg(
        win_rate=('win', 'mean'),
        avg_kda=('kda', 'mean'),
        games=('match_id', 'count')
    ).reset_index()
    top_champs = champ_stats.nlargest(5, 'games')
    sns.barplot(data=top_champs, y='champion', x='win_rate', ax=ax,
                palette='Blues_r')
    for i, (_, row) in enumerate(top_champs.iterrows()):
        ax.text(row.win_rate + 3, i,
                f"Games: {row.games}\nKDA: {row.avg_kda:.2f}", va='center')
    ax.set(title='Top 5 Most Played Champions',
           xlabel='Win Rate (%)', xlim=(0, 100))

    # 3. CS vs Game Duration
    ax = axs[1, 0]
    sns.scatterplot(data=df, x='game_duration', y='cs_per_minute', ax=ax,
                    hue='win', palette={True: 'green', False: 'red'},
                    size='damage_share', sizes=(50, 200), alpha=0.7)
    ax.set(title='CS Per Minute vs Game Duration',
           xlabel='Game Duration (min)', ylabel='CS/Min')

    # 4. Damage Share Timeline
    ax = axs[1, 1]
    damage_df = df.sort_values('game_date').reset_index(drop=True)
    damage_df['game_number'] = damage_df.index + 1
    sns.barplot(data=damage_df, x='game_number', y='damage_share', ax=ax,
                hue='win', palette={True: 'green', False: 'red'}, alpha=0.7)
    ax.set(title='Damage Share by Game', xlabel='Game Number',
           ylabel='Damage Share (%)')

    # 5. Vision Score Comparison
    ax = axs[2, 0]
    vision_stats = df.groupby('win')['vision_score'].mean().reset_index()
    vision_stats['result'] = vision_stats['win'].map(
        {True: 'Wins', False: 'Losses'})
    sns.barplot(data=vision_stats, x='result', y='vision_score', ax=ax,
                palette={'Wins': 'green', 'Losses': 'red'})
    ax.set(title='Average Vision Score', xlabel='', ylabel='Vision Score')

    # 6. Kill Participation Distribution
    ax = axs[2, 1]
    sns.kdeplot(data=df, x='kill_participation', ax=ax,
                hue='win', palette={True: 'green', False: 'red'},
                fill=True, alpha=0.5)
    ax.set(title='Kill Participation Distribution',
           xlabel='Kill Participation (%)', ylabel='Density')

    plt.tight_layout(pad=3.0)
    return fig


def display_summary(analysis):
    """Display formatted summary statistics"""
    st.subheader("📊 Overall Performance")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Games", analysis['overall']['game_count'])
        st.metric("Win Rate", f"{analysis['overall']['win_rate']}%")
        st.metric("Average KDA", analysis['overall']['avg_kda'])

    with col2:
        st.metric("CS/Min", analysis['overall']['avg_cs_per_min'])
        st.metric("Vision Score", analysis['overall']['avg_vision_score'])
        st.metric("Damage Share", f"{analysis['overall']['avg_damage_share']}%")

    with col3:
        st.metric("Kill Participation", f"{analysis['overall']['avg_kill_participation']}%")
        st.metric("Average Damage", f"{analysis['overall']['avg_damage']:,}")
        st.metric("Gold Earned", f"{analysis['overall']['avg_gold']:,}")

    st.subheader("🏆 Champion Performance")
    champ_cols = st.columns(len(analysis['champions']))
    for i, (champ, games) in enumerate(analysis['champions'].items()):
        with champ_cols[i]:
            st.metric(
                label=champ,
                value=f"{games} games",
                delta=f"{analysis['champion_win_rates'].get(champ, 0):.1f}% win rate"
            )

    if analysis['positions']:
        st.subheader("🧭 Position Breakdown")
        pos_cols = st.columns(len(analysis['positions']))
        for i, (pos, count) in enumerate(analysis['positions'].items()):
            if pos != "Unknown":
                with pos_cols[i]:
                    st.metric(
                        label=pos,
                        value=f"{count} games",
                        delta=f"{analysis['position_win_rates'].get(pos, 0):.1f}% win rate"
                    )


# --- Streamlit UI ---
st.title("League of Legends Match Analyzer 🔍")

with st.form("summoner_form"):
    game_name = st.text_input("Game Name (e.g. Faker)", value="YourName")
    tag_line = st.text_input("Tag Line (e.g. KR1)", value="NA1")
    match_count = st.slider("Number of Recent Matches", min_value=5, max_value=50, value=20)
    compare_tier = st.selectbox(
        "Compare your stats to:",
        list(TIER_BENCHMARKS.keys()),
        index=4  # Default to Platinum
    )
    submitted = st.form_submit_button("Analyze")

if submitted:
    try:
        with st.spinner("Fetching account info..."):
            account = get_account_info(game_name, tag_line)
        puuid = account['puuid']
        st.success(f"Found account: {account['gameName']}#{account['tagLine']}")

        with st.spinner(f"Fetching {match_count} matches..."):
            match_ids = get_matches(puuid, match_count)

        stats_list = []
        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, match_id in enumerate(match_ids):
            try:
                status_text.text(f"Processing match {i + 1}/{len(match_ids)}...")
                match = get_match_details(match_id)
                stats = collect_player_stats(match, puuid)
                if stats:
                    stats_list.append(stats)
                progress_bar.progress((i + 1) / len(match_ids))
            except RiotAPIError as e:
                st.warning(f"Skipping match due to error: {e}")
                continue

        if not stats_list:
            st.error("No valid matches found for analysis.")
            st.stop()

        analysis = analyze_stats(stats_list)
        display_summary(analysis)

        st.subheader("📈 Performance Visualizations")

        # Main visualizations
        fig = visualize_stats(analysis['df'])
        st.pyplot(fig)

        # Radar chart comparison
        st.subheader("🎯 Performance Tier Comparison")
        radar_fig = draw_comparison_radar(analysis['overall'], compare_tier)
        st.pyplot(radar_fig)

        with st.expander("How to read the radar chart"):
            st.markdown("""
            - **KDA**: Kill/Death/Assist ratio (normalized to 10.0)
            - **CS/Min**: Creep Score per minute (normalized to 12.0)
            - **Vision**: Vision score (normalized to 100)
            - **Damage**: Average damage to champions (normalized to 40,000)
            - **Gold**: Average gold earned (normalized to 40,000)
            - **KP%**: Kill participation percentage (0-100%)
            - **Dmg%**: Percentage of team's total damage (0-50%)

            The blue area shows your performance, while the orange area shows the 
            average for the selected tier. Areas where your blue extends beyond 
            the orange indicate you're performing above that tier's average.
            """)

        st.subheader("📥 Export Data")
        csv = analysis['df'].to_csv(index=False).encode()
        st.download_button(
            label="Download Match Data as CSV",
            data=csv,
            file_name="league_match_data.csv",
            mime="text/csv"
        )

    except RiotAPIError as err:
        st.error(f"Failed to fetch data: {err}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
