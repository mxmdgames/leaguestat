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

# Keeping tier benchmarks as a fallback option
TIER_BENCHMARKS = {
    'Iron': {
        'KDA': 0.25,  # 1.5-2.0 → midpoint 1.75 normalized to 6.0 scale
        'CS/Min': 0.35,  # 4-5 → midpoint 4.5 normalized to 13 scale
        'Vision': 0.17,  # 0.5-1.0 → midpoint 0.75 normalized to 4.5 scale
        'Damage': 0.29,  # 15-20% → midpoint 17.5% normalized to 60% scale
        'Gold': 0.43,  # 300-350 → midpoint 325 normalized to 750 scale
        'KP%': 0.47,  # 40-50% → midpoint 45% normalized to 95% scale
        'Dmg%': 0.25  # 15-20% → midpoint 17.5% normalized to 70% scale
    },
    # Other tiers remain the same...
    'Challenger': {
        'KDA': 1.0,  # 6.0+
        'CS/Min': 1.0,  # 13+
        'Vision': 1.0,  # 4.5+
        'Damage': 0.83,  # 45-55% → 50%
        'Gold': 1.0,  # 750+
        'KP%': 0.89,  # 85-95% → 90%
        'Dmg%': 0.75  # 45-55% → 50%
    }
}

# Preset pro players for easy comparison
PRO_PLAYERS = {
    "Faker": {"game_name": "Faker", "tag_line": "KR1"},
    "Caps": {"game_name": "Caps", "tag_line": "EUW"},
    "Bjergsen": {"game_name": "Bjergsen", "tag_line": "NA1"},
    "Chovy": {"game_name": "Chovy", "tag_line": "KR1"},
    "Showmaker": {"game_name": "Showmaker", "tag_line": "KR1"},
    # Add more pros as needed
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


@st.cache_data(ttl=3600)
def get_item_data():
    """Fetches the current item data from Data Dragon and caches it"""
    try:
        # Get latest version
        versions_url = "https://ddragon.leagueoflegends.com/api/versions.json"
        versions = requests.get(versions_url).json()
        latest_version = versions[0]

        # Get item data
        items_url = f"https://ddragon.leagueoflegends.com/cdn/{latest_version}/data/en_US/item.json"
        items_data = requests.get(items_url).json()

        return items_data['data'], latest_version
    except Exception as e:
        st.warning(f"Could not fetch item data: {e}")
        return {}, "14.9.1"  # Fallback version


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

            # Get player items (item0 through item6)
            items = [p.get(f'item{i}', 0) for i in range(7)]
            # Filter out items with id 0 (empty slots)
            items = [item for item in items if item != 0]

            # Get mythic item stats if available
            mythic_stats = {}
            if 'challenges' in p and 'mythicItemUsed' in p['challenges']:
                mythic_stats = {
                    'mythic_id': p['challenges'].get('mythicItemUsed', 0),
                    'time_purchased': p['challenges'].get('mythicItemTimeConsumed', 0) / 60  # Convert to minutes
                }

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
                'items': items,  # Add items to the return data
                'mythic_stats': mythic_stats,  # Add mythic item stats
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

    # Item analysis
    # Create a list of all items used across all matches
    all_items = []
    for items in df['items']:
        all_items.extend(items)

    # Count item frequencies
    item_counts = pd.Series(all_items).value_counts().head(10).to_dict()

    # Calculate win rates for most frequent items
    item_wins = {}
    for item_id in item_counts.keys():
        # Find all matches where this item was used
        matches_with_item = df[[item_id in items for items in df['items']]]
        if not matches_with_item.empty:
            win_rate = (matches_with_item['win'].mean() * 100).round(1)
            item_wins[item_id] = win_rate

    # Mythic item analysis
    mythic_data = []
    for _, row in df.iterrows():
        if 'mythic_stats' in row and row['mythic_stats'].get('mythic_id', 0) > 0:
            mythic_data.append({
                'mythic_id': row['mythic_stats']['mythic_id'],
                'time_purchased': row['mythic_stats']['time_purchased'],
                'win': row['win'],
                'champion': row['champion']
            })

    mythic_df = pd.DataFrame(mythic_data) if mythic_data else pd.DataFrame()
    mythic_stats = {}

    if not mythic_df.empty:
        # Get most common mythics
        mythic_counts = mythic_df['mythic_id'].value_counts().head(5).to_dict()

        # Calculate win rates and avg purchase times
        mythic_stats = {
            'counts': mythic_counts,
            'win_rates': {},
            'avg_times': {}
        }

        for mythic_id in mythic_counts.keys():
            mythic_matches = mythic_df[mythic_df['mythic_id'] == mythic_id]
            mythic_stats['win_rates'][mythic_id] = (mythic_matches['win'].mean() * 100).round(1)
            mythic_stats['avg_times'][mythic_id] = mythic_matches['time_purchased'].mean().round(1)

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
        'item_counts': item_counts,  # Add item frequency
        'item_win_rates': item_wins,  # Add item win rates
        'mythic_stats': mythic_stats,  # Add mythic item stats
    }


# NEW: Function to fetch comparison player's data
def get_comparison_player_stats(game_name, tag_line, match_count=10):
    try:
        # Fetch account info
        account = get_account_info(game_name, tag_line)
        puuid = account['puuid']

        # Fetch recent matches
        match_ids = get_matches(puuid, match_count)

        # Collect stats from each match
        stats_list = []
        for match_id in match_ids:
            try:
                match = get_match_details(match_id)
                stats = collect_player_stats(match, puuid)
                if stats:
                    stats_list.append(stats)
            except RiotAPIError:
                # Continue with other matches if one fails
                continue

        if not stats_list:
            return None

        # Analyze the collected stats
        analysis = analyze_stats(stats_list)

        return {
            'player_name': f"{account['gameName']}#{account['tagLine']}",
            'stats': analysis['overall'] if analysis else None
        }
    except Exception as e:
        st.warning(f"Error fetching comparison data for {game_name}#{tag_line}: {str(e)}")
        return None


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


# MODIFIED: Updated to support comparing to other summoners
def draw_comparison_radar(player_stats, comparison_data):
    """Create radar chart comparing player to other summoners or tier"""
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

    # Prepare data for radar chart
    categories = list(player_normalized.keys())
    player_values = list(player_normalized.values()) + [player_normalized['KDA']]

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

    # Plot comparison data (can be multiple players or a tier)
    colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    legend_handles = [Line2D([0], [0], color='#1f77b4', lw=2, label='Your Stats')]

    # If comparison is a tier benchmark
    if isinstance(comparison_data, str):
        benchmark = TIER_BENCHMARKS[comparison_data]
        benchmark_values = list(benchmark.values()) + [benchmark['KDA']]
        ax.plot(angles, benchmark_values, linewidth=2, linestyle='solid',
                color=colors[0], label=f'{comparison_data} Average')
        ax.fill(angles, benchmark_values, colors[0], alpha=0.1)
        legend_handles.append(Line2D([0], [0], color=colors[0], lw=2, label=f'{comparison_data} Average'))

    # If comparison is other summoners
    else:
        for i, (name, comp_stats) in enumerate(comparison_data.items()):
            if comp_stats and i < len(colors):
                color_idx = i % len(colors)
                comp_normalized = {
                    'KDA': normalize_to_percent(comp_stats['avg_kda'], 'KDA'),
                    'CS/Min': normalize_to_percent(comp_stats['avg_cs_per_min'], 'CS/Min'),
                    'Vision': normalize_to_percent(comp_stats['avg_vision_score'], 'Vision'),
                    'Damage': normalize_to_percent(comp_stats['avg_damage'], 'Damage'),
                    'Gold': normalize_to_percent(comp_stats['avg_gold'], 'Gold'),
                    'KP%': normalize_to_percent(comp_stats['avg_kill_participation'], 'KP%'),
                    'Dmg%': normalize_to_percent(comp_stats['avg_damage_share'], 'Dmg%')
                }
                comp_values = list(comp_normalized.values()) + [comp_normalized['KDA']]

                ax.plot(angles, comp_values, linewidth=2, linestyle='solid',
                        color=colors[color_idx], label=name)
                ax.fill(angles, comp_values, colors[color_idx], alpha=0.1)
                legend_handles.append(Line2D([0], [0], color=colors[color_idx], lw=2, label=name))

    # Add title and legend
    if isinstance(comparison_data, str):
        plt.title(f'Your Performance vs {comparison_data} Averages\n(Normalized to max expected values)',
                  size=14, y=1.1)
    else:
        plt.title(f'Your Performance vs Other Summoners\n(Normalized to max expected values)',
                  size=14, y=1.1)

    plt.legend(handles=legend_handles, loc='upper right', bbox_to_anchor=(1.3, 1.1))

    return fig


def visualize_items(items_data, item_counts, item_win_rates, version):
    """Create item visualization with images and win rates"""

    # Create figure for item analysis
    fig, ax = plt.subplots(figsize=(12, 6))

    # Sort items by frequency
    sorted_items = sorted(item_counts.items(), key=lambda x: x[1], reverse=True)
    item_ids = [item[0] for item in sorted_items[:8]]  # Top 8 items
    item_freq = [item_counts[item_id] for item_id in item_ids]

    # Create bar chart
    bars = ax.bar(range(len(item_ids)), item_freq, color='steelblue')

    # Add win rate as text on each bar
    for i, item_id in enumerate(item_ids):
        win_rate = item_win_rates.get(item_id, 0)
        item_name = "Unknown"
        if str(item_id) in items_data:
            item_name = items_data[str(item_id)].get('name', 'Unknown')

        # Truncate long names
        if len(item_name) > 15:
            item_name = item_name[:12] + "..."

        ax.text(i, item_freq[i] + 0.3, f"{win_rate}% WR",
                ha='center', va='bottom', fontsize=9)
        ax.text(i, -0.5, item_name, ha='center', va='top',
                rotation=45, fontsize=8)

    # Customize chart
    ax.set_xticks(range(len(item_ids)))
    ax.set_xticklabels([f"Item {i + 1}" for i in range(len(item_ids))], rotation=0)
    ax.set_title("Top Items Built and Win Rates")
    ax.set_xlabel("Items")
    ax.set_ylabel("Number of Games")

    return fig


def visualize_mythic_timing(mythic_stats, items_data, version):
    """Create visualization of mythic item purchase timing"""
    if not mythic_stats or 'counts' not in mythic_stats or not mythic_stats['counts']:
        return None

    # Create figure for mythic timing analysis
    fig, ax = plt.subplots(figsize=(12, 6))

    # Get top mythics
    mythic_ids = list(mythic_stats['counts'].keys())
    purchase_times = [mythic_stats['avg_times'].get(mid, 0) for mid in mythic_ids]
    win_rates = [mythic_stats['win_rates'].get(mid, 0) for mid in mythic_ids]

    # Create sorted indices by purchase time
    sorted_indices = sorted(range(len(purchase_times)), key=lambda i: purchase_times[i])

    # Sort data
    sorted_mythic_ids = [mythic_ids[i] for i in sorted_indices]
    sorted_times = [purchase_times[i] for i in sorted_indices]
    sorted_win_rates = [win_rates[i] for i in sorted_indices]

    # Get mythic names
    mythic_names = []
    for mid in sorted_mythic_ids:
        if str(mid) in items_data:
            name = items_data[str(mid)].get('name', f"Item {mid}")
            # Truncate long names
            if len(name) > 12:
                name = name[:9] + "..."
            mythic_names.append(name)
        else:
            mythic_names.append(f"Item {mid}")

    # Create bar chart for purchase times
    bars = ax.bar(range(len(sorted_mythic_ids)), sorted_times, color='purple', alpha=0.7)

    # Create twin axis for win rates
    ax2 = ax.twinx()
    ax2.plot(range(len(sorted_mythic_ids)), sorted_win_rates, 'ro-', linewidth=2, markersize=8)

    # Add purchase time to each bar
    for i, time in enumerate(sorted_times):
        ax.text(i, time + 0.5, f"{time} min", ha='center', va='bottom', fontsize=9)

    # Add win rate to each point
    for i, wr in enumerate(sorted_win_rates):
        ax2.text(i, wr + 2, f"{wr}%", ha='center', va='bottom', fontsize=9, color='red')

    # Customize chart
    ax.set_xticks(range(len(sorted_mythic_ids)))
    ax.set_xticklabels(mythic_names, rotation=45, ha='right')
    ax.set_title("Mythic Item Purchase Timing and Win Rates")
    ax.set_xlabel("Mythic Items")
    ax.set_ylabel("Avg. Purchase Time (minutes)")
    ax2.set_ylabel("Win Rate (%)", color='red')
    ax2.tick_params(axis='y', colors='red')

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


def display_item_details(item_id, items_data, version):
    """Display detailed information about an item"""
    if str(item_id) not in items_data:
        st.warning(f"Item {item_id} not found in the current item database.")
        return

    item = items_data[str(item_id)]

    # Create columns for image and details
    col1, col2 = st.columns([1, 3])

    with col1:
        st.image(
            f"https://ddragon.leagueoflegends.com/cdn/{version}/img/item/{item_id}.png",
            width=80
        )

    with col2:
        st.markdown(f"## {item['name']}")
        st.markdown(f"**Gold**: {item['gold']['total']} (Base: {item['gold']['base']})")

        if item.get('description'):
            # Clean up HTML tags from description
            desc = item['description'].replace('<br>', '\n').replace('<stats>', '').replace('</stats>', '')
            desc = desc.replace('<unique>', '').replace('</unique>', '')
            desc = desc.replace('<active>', '**Active:** ').replace('</active>', '')
            desc = desc.replace('<passive>', '**Passive:** ').replace('</passive>', '')
            st.markdown(desc)

        if 'plaintext' in item and item['plaintext']:
            st.markdown(f"*{item['plaintext']}*")

        # Item tags
        if 'tags' in item:
            st.markdown("**Tags:** " + ", ".join(item['tags']))

        # Item builds into
        if 'into' in item:
            st.markdown("**Builds into:**")
            build_cols = st.columns(min(5, len(item['into'])))
            for i, target_id in enumerate(item['into']):
                if str(target_id) in items_data:
                    with build_cols[i % 5]:
                        st.image(
                            f"https://ddragon.leagueoflegends.com/cdn/{version}/img/item/{target_id}.png",
                            width=40
                        )
                        st.caption(items_data[str(target_id)]['name'])

        # Item builds from
        if 'from' in item:
            st.markdown("**Built from:**")
            build_cols = st.columns(min(5, len(item['from'])))
            for i, source_id in enumerate(item['from']):
                if str(source_id) in items_data:
                    with build_cols[i % 5]:
                        st.image(
                            f"https://ddragon.leagueoflegends.com/cdn/{version}/img/item/{source_id}.png",
                            width=40
                        )
                        st.caption(items_data[str(source_id)]['name'])


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


def display_item_analysis(analysis, items_data, version):
    """Display item analysis section"""
    st.subheader("🛡️ Item Analysis")

    # Display most frequent items
    st.markdown("### Most Frequently Built Items")

    # Create columns for the top items
    num_items = min(5, len(analysis['item_counts']))
    cols = st.columns(num_items)

    for i, (item_id, count) in enumerate(list(analysis['item_counts'].items())[:num_items]):
        with cols[i]:
            item_id_str = str(item_id)
            if item_id_str in items_data:
                item = items_data[item_id_str]

                # Display item image
                st.image(
                    f"https://ddragon.leagueoflegends.com/cdn/{version}/img/item/{item_id}.png",
                    caption=item['name'],
                    width=60
                )

                # Display item stats
                win_rate = analysis['item_win_rates'].get(item_id, 0)
                st.metric(
                    label=f"Used in {count} games",
                    value=f"{win_rate}% winrate"
                )

                # Add item details on click
                if st.button(f"Details: {item['name']}", key=f"item_{item_id}"):
                    display_item_details(item_id, items_data, version)

    # Item visualization
    item_fig = visualize_items(items_data, analysis['item_counts'], analysis['item_win_rates'], version)
    st.pyplot(item_fig)

    # Mythic item analysis if available
    if 'mythic_stats' in analysis and analysis['mythic_stats'] and len(analysis['mythic_stats'].get('counts', {})) > 0:
        st.markdown("### Mythic Item Analysis")

        # Create columns for mythic items
        mythic_ids = list(analysis['mythic_stats']['counts'].keys())
        num_mythics = min(3, len(mythic_ids))
        mythic_cols = st.columns(num_mythics)

        for i, mythic_id in enumerate(mythic_ids[:num_mythics]):
            with mythic_cols[i]:
                count = analysis['mythic_stats']['counts'][mythic_id]
                win_rate = analysis['mythic_stats']['win_rates'].get(mythic_id, 0)
                avg_time = analysis['mythic_stats']['avg_times'].get(mythic_id, 0)

                if str(mythic_id) in items_data:
                    mythic = items_data[str(mythic_id)]

                    # Display mythic image
                    st.image(
                        f"https://ddragon.leagueoflegends.com/cdn/{version}/img/item/{mythic_id}.png",
                        caption=mythic['name'],
                        width=60
                    )

                    # Display mythic stats
                    st.metric(
                        label=f"Built in {count} games",
                        value=f"{win_rate}% winrate",
                        delta=f"Avg. {avg_time} min"
                    )

                    # Add mythic details on click
                    if st.button(f"Details: {mythic['name']}", key=f"mythic_{mythic_id}"):
                        display_item_details(mythic_id, items_data, version)

        # Mythic visualization
        mythic_fig = visualize_mythic_timing(analysis['mythic_stats'], items_data, version)
        if mythic_fig:
            st.pyplot(mythic_fig)

    # Champion-specific build paths
    st.markdown("### Champion Build Paths")

    # Create tabs for each champion
    champion_names = list(analysis['champions'].keys())
    if champion_names:
        tabs = st.tabs(champion_names)

        for i, champion in enumerate(champion_names):
            with tabs[i]:
                champion_matches = analysis['df'][analysis['df']['champion'] == champion]

                # Most common first items
                first_items = []
                for _, row in champion_matches.iterrows():
                    if len(row['items']) > 0:
                        first_items.append(row['items'][0])

                if first_items:
                    first_item_counts = pd.Series(first_items).value_counts()
                    most_common_first = first_item_counts.index[0] if not first_item_counts.empty else None

                    col1, col2 = st.columns([1, 3])

                    with col1:
                        if most_common_first and str(most_common_first) in items_data:
                            st.markdown("**First Item**")
                            st.image(
                                f"https://ddragon.leagueoflegends.com/cdn/{version}/img/item/{most_common_first}.png",
                                width=60
                            )
                            st.caption(items_data[str(most_common_first)]['name'])

                    with col2:
                        # Find build paths (sequences of 3+ items)
                        build_paths = []
                        for _, row in champion_matches.iterrows():
                            if len(row['items']) >= 3:
                                build_paths.append(tuple(row['items'][:3]))

                        if build_paths:
                            path_counts = pd.Series(build_paths).value_counts()
                            if not path_counts.empty:
                                most_common_path = path_counts.index[0]
                                st.markdown(f"**Most Common Build Path** (Used in {path_counts[0]} games)")

                                path_cols = st.columns(3)
                                for j, item_id in enumerate(most_common_path):
                                    with path_cols[j]:
                                        if str(item_id) in items_data:
                                            st.image(
                                                f"https://ddragon.leagueoflegends.com/cdn/{version}/img/item/{item_id}.png",
                                                width=60
                                            )
                                            st.caption(items_data[str(item_id)]['name'])

                # Show champion performance metrics with different items
                st.markdown("**Performance with Key Items**")

                # Find most built items for this champion
                champ_items = []
                for _, row in champion_matches.iterrows():
                    champ_items.extend(row['items'])

                champ_item_counts = pd.Series(champ_items).value_counts().head(5)

                if not champ_item_counts.empty:
                    item_stats = []

                    for item_id in champ_item_counts.index:
                        # Find games with this item
                        games_with_item = champion_matches[[item_id in items for items in champion_matches['items']]]

                        if not games_with_item.empty:
                            item_stats.append({
                                'item_id': item_id,
                                'games': len(games_with_item),
                                'win_rate': (games_with_item['win'].mean() * 100).round(1),
                                'avg_kda': games_with_item['kda'].mean().round(2),
                                'name': items_data.get(str(item_id), {}).get('name', f"Item {item_id}")
                            })

                    if item_stats:
                        # Create a DataFrame for display
                        item_df = pd.DataFrame(item_stats)

                        # Display the table
                        st.dataframe(
                            item_df[['name', 'games', 'win_rate', 'avg_kda']].rename(
                                columns={
                                    'name': 'Item',
                                    'games': 'Games',
                                    'win_rate': 'Win Rate %',
                                    'avg_kda': 'Avg KDA'
                                }
                            ),
                            hide_index=True
                        )


# --- NEW FUNCTIONS ---
def display_comparison_summary(comparison_data):
    """Display summary of comparison players' stats"""
    st.subheader("⚔️ Comparison Summoners")

    # Create columns for each comparison player
    cols = st.columns(len(comparison_data))

    for i, (name, stats) in enumerate(comparison_data.items()):
        with cols[i]:
            st.markdown(f"### {name}")
            if stats:
                st.metric("Win Rate", f"{stats['win_rate']}%")
                st.metric("Average KDA", f"{stats['avg_kda']}")
                st.metric("CS/Min", f"{stats['avg_cs_per_min']}")
                st.metric("Vision Score", f"{stats['avg_vision_score']}")
                st.metric("Damage Share", f"{stats['avg_damage_share']}%")
                st.metric("Kill Participation", f"{stats['avg_kill_participation']}%")
            else:
                st.write("No data available")


# --- STREAMLIT UI ---
st.title("League of Legends Match Analyzer 🔍")

# MODIFIED: UI to support comparison players
with st.form("summoner_form"):
    col1, col2 = st.columns([1, 1])

    with col1:
        game_name = st.text_input("Your Game Name (e.g. Faker)", value="YourName")
        tag_line = st.text_input("Your Tag Line (e.g. KR1)", value="NA1")
        match_count = st.slider("Number of Recent Matches", min_value=5, max_value=50, value=20)

    with col2:
        # Comparison options
        comparison_type = st.radio(
            "Compare with:",
            ["Other Summoners", "Tier Benchmarks"],
            index=0
        )

        if comparison_type == "Tier Benchmarks":
            compare_tier = st.selectbox(
                "Compare your stats to tier:",
                list(TIER_BENCHMARKS.keys()),
                index=4  # Default to Platinum
            )
            comparison_players = {}
        else:
            # Comparison player options
            st.markdown("### Add Comparison Summoners")

            # Option to select from preset pro players
            use_preset = st.checkbox("Use preset pro player")

            if use_preset:
                pro_player = st.selectbox("Select a pro player", list(PRO_PLAYERS.keys()))
                comp_name = PRO_PLAYERS[pro_player]["game_name"]
                comp_tag = PRO_PLAYERS[pro_player]["tag_line"]
            else:
                # Manual entry
                comp_name = st.text_input("Comparison Game Name", value="")
                comp_tag = st.text_input("Comparison Tag Line", value="")

            # Option to add a second comparison player
            add_second = st.checkbox("Add second comparison")

            if add_second:
                use_preset2 = st.checkbox("Use preset pro player for second comparison")

                if use_preset2:
                    pro_player2 = st.selectbox("Select second pro player", list(PRO_PLAYERS.keys()), key="pro2")
                    comp_name2 = PRO_PLAYERS[pro_player2]["game_name"]
                    comp_tag2 = PRO_PLAYERS[pro_player2]["tag_line"]
                else:
                    # Manual entry for second player
                    comp_name2 = st.text_input("Second Comparison Game Name", value="")
                    comp_tag2 = st.text_input("Second Comparison Tag Line", value="")

    submitted = st.form_submit_button("Analyze")

if submitted:
    try:
        with st.spinner("Fetching your account info..."):
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

        # Fetch item data for visualization
        items_data, version = get_item_data()

        # MODIFIED: Handle comparison data fetching
        comparison_data = {}

        if comparison_type == "Other Summoners":
            # First comparison player
            if use_preset:
                comp_display_name = pro_player
            else:
                comp_display_name = f"{comp_name}#{comp_tag}"

            if comp_name and comp_tag:
                with st.spinner(f"Fetching comparison data for {comp_display_name}..."):
                    comp_data = get_comparison_player_stats(comp_name, comp_tag, 10)
                    if comp_data and comp_data['stats']:
                        comparison_data[comp_display_name] = comp_data['stats']

            # Second comparison player (if added)
            if add_second and comp_name2 and comp_tag2:
                if use_preset2:
                    comp_display_name2 = pro_player2
                else:
                    comp_display_name2 = f"{comp_name2}#{comp_tag2}"

                with st.spinner(f"Fetching comparison data for {comp_display_name2}..."):
                    comp_data2 = get_comparison_player_stats(comp_name2, comp_tag2, 10)
                    if comp_data2 and comp_data2['stats']:
                        comparison_data[comp_display_name2] = comp_data2['stats']

        # Display stat summary
        display_summary(analysis)

        # Display comparison summoners if available
        if comparison_type == "Other Summoners" and comparison_data:
            display_comparison_summary(comparison_data)

        # Create tabs for different analysis sections
        tab1, tab2, tab3 = st.tabs(["Performance Charts", "Item Analysis", "Comparison"])

        with tab1:
            st.subheader("📈 Performance Visualizations")
            fig = visualize_stats(analysis['df'])
            st.pyplot(fig)

        with tab2:
            # Display item analysis
            display_item_analysis(analysis, items_data, version)

        with tab3:
            # Radar chart comparison (modified to support comparison players)
            st.subheader("🎯 Performance Comparison")

            if comparison_type == "Tier Benchmarks":
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
            else:
                if comparison_data:
                    radar_fig = draw_comparison_radar(analysis['overall'], comparison_data)
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

                        The blue area shows your performance, while the other colored areas show the 
                        performance of comparison summoners. Areas where your blue extends beyond 
                        the other colors indicate you're performing better in those categories.
                        """)
                else:
                    st.warning("No comparison data available. Please add valid summoners for comparison.")

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
