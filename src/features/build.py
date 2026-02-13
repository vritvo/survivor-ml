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
    return bm
    
    
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

    
