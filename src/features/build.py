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
    
    
#TODO: Add each feature group function
    
def build_modeling_table(data:dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build the full modeling table by chaining feature functions.
    
    Returns one row per (season, episode, player still in game) with all features
    and the target variable eliminated_this_episode."""
    skel = get_skeleton(data)
    
    
    #TODO: df = ... add chaining of feature functions here. 
    
    return skel
    
    
    
if __name__ == "__main__":
    data = load_data()
    df = build_modeling_table(data)
    
    print(f"Shapee: {df.shape[0]} rows x {df.shape[1]} cols")
    print(f"Columns: {df.columns.tolist()}")
    print()

    
