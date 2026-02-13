"""Build the core modeling table: one row per (season, episode, player still in game)."""

import pandas as pd
from src.load import load_data 



def get_skeleton(data:dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build the skeleton: one row per (season, episode, player still in game).
    
    Source: Boot Mapping filtered to US + "In the game". 
    Deduplication: episodes with multiple tribals create duplicate rows. We keep the 
    keep the first occurrence per episode to represent the players state at the start. 
    
    Target: eliminated_this_episode = 1 if the player was eliminated in this episode (derived
    from Castaways table). 
    
    
    """
    bm = data["Boot Mapping"]
    castaways = data["Castaways"]

    # Filter to US seasons, players still in the game
    skel = bm[(bm["version"] == "US") & (bm["game_status"] == "In the game")].copy()

    # Deduplicate: keep first row per (season, episode, castaway_id)
    # Boot Mapping is ordered by `order` (boot sequence), so first = earliest
    skel = skel.sort_values("order")
    skel = skel.drop_duplicates(subset=["season", "episode", "castaway_id"], keep="first")

    # Build target variable from Castaways table.
    # Castaways.episode = the episode a player was eliminated in.
    # A player who was never voted out (winner, runner-up) has episode = final episode
    # but their result is not "voted out", so we filter to elimination results.
    us_castaways = castaways[castaways["version"] == "US"].copy()

    # Identify eliminated players: result contains "voted out" or other elimination types
    elimination_results = us_castaways["result"].str.contains(
        "voted out|Quit|Evacuated|Removed|Ejected|Lost final|Eliminated",
        case=False, na=False,
    )
    eliminated = us_castaways.loc[elimination_results, ["season", "castaway_id", "episode"]].copy()
    eliminated = eliminated.rename(columns={"episode": "elim_episode"})

    # Merge: if this player's elimination episode matches the current episode → target = 1
    skel = skel.merge(eliminated, on=["season", "castaway_id"], how="left")
    skel["eliminated_this_episode"] = (skel["episode"] == skel["elim_episode"]).astype(int)
    skel = skel.drop(columns=["elim_episode"])

    # Keep useful columns from the skeleton
    skel = skel[
        ["season", "episode", "castaway_id", "castaway", "tribe", "tribe_status",
         "order", "final_n", "eliminated_this_episode"]
    ].reset_index(drop=True)
    
    return skel
    
def add_static_features(skel: pd.DataFrame, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Add time-invariant player features:
    
    - age: from Castaways (age at time of playing that season)
    - gender: from Castaway Details (one row per unique castaway across all seasons)
    """
    castaways = data["Castaways"]
    details = data["Castaway Details"]
    df = skel.copy()

    # Age: join from Castaways on (season, castaway_id) since age varies per season
    age_lookup = castaways[castaways["version"] == "US"][["season", "castaway_id", "age"]].drop_duplicates()
    df = df.merge(age_lookup, on=["season", "castaway_id"], how="left")

    # Gender: join from Castaway Details on castaway_id
    gender_lookup = details[["castaway_id", "gender"]].drop_duplicates()
    df = df.merge(gender_lookup, on="castaway_id", how="left")

    return df
    

def add_vote_features(skel: pd.DataFrame, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """#TODO: Add cumulative vote features.

    feature ideas:
    - votes_against_cumulative: total votes received against this player so far
    - times_in_danger: episodes where player received at least one vote
    - correct_votes_cumulative: times player voted for the person who went home
    - correct_votes_recent: correct votes in the last 3 episodes (momentum)
    - votes_against_recent: votes in the last 3 episodes (momentum)

    Also: 
    - no future leakage, use vote history table, 
    - group by (season, castaway_id),
    - compute cumulative sums up to episode - 1.
    """
    return skel.copy()


def add_challenge_features(skel: pd.DataFrame, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """#TODO: Add cumulative challenge features.

    feature ideas:
    - individual_immunity_wins: cumulative individual immunity challenge wins
    - tribe_challenge_wins: cumulative tribal challenge wins
    - tribe_challenge_losses: cumulative tribal challenge losses
    - sit_outs: number of times player sat out of a challenge

    Also: 
    - Use Challenge Results table. 
    - Filter by challenge_type and result columns.
    - Same leakage constraint: only episodes < current episode.
    """
    return skel.copy()


def add_confessional_features(skel: pd.DataFrame, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """#TODO: Add confessional / screen time features.

    feature ideas:
    - confessionals_cumulative: total confessional count up to previous episode
    - confessionals_last_3_eps: confessionals in the last 3 episodes (momentum)
    - confessional_share: player's share of total confessionals that episode

    Also: 
    - Use Confessionals table (one row per player per episode with confessional_count).
    - Filter by episode < current episode.
    """
    return skel.copy()


def add_advantage_features(skel: pd.DataFrame, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """#TODO: Add advantage/idol features.

    feature ideas:
    - idols_found_cumulative: number of hidden immunity idols found
    - idols_played_cumulative: number of idols played
    - advantages_held: count of advantages currently held (found - played)

    Also: 
    - Use Advantage Movement table (events: found, played, etc.) joined with
    Advantage Details for advantage type.
    - Filter by episode < current episode.
    """
    return skel.copy()


def build_modeling_table(data:dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build the full modeling table by chaining feature functions.
    
    Returns one row per (season, episode, player still in game) with all features
    and the target variable eliminated_this_episode."""
    skel = get_skeleton(data)
    
    df = add_static_features(skel, data)
    df = add_vote_features(df, data)
    df = add_challenge_features(df, data)
    df = add_confessional_features(df, data)
    df = add_advantage_features(df, data)
    
    return df
    
    
    
if __name__ == "__main__":
    data = load_data()
    df = build_modeling_table(data)
    
    print(f"Shapee: {df.shape[0]} rows x {df.shape[1]} cols")
    print(f"Columns: {df.columns.tolist()}")
    print()

    
