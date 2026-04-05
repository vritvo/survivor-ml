"""Build the core modeling table: one row per (season, episode, player still in game)."""


import warnings
import pandas as pd
from src.load import load_data 


# Advantage event classifications — used by _build_holding_periods
_GAIN_EVENTS = {"Found", "Found (beware)", "Received", "Recieved"}
_LOSS_EVENTS = {
    "Played", "Voted out with advantage", "Expired",
    "Left game with advantage", "Medically evacuated with advantage",
    "Quit with advantage", "Destroyed", "Discarded",
}
_NEUTRAL_EVENTS = {
    "Activated", "Banked", "Became hidden immunity idol",
    "Became steal a vote", "Absorbed",
}



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

    # Build target variable "eliminated_this_episode" from Castaways table.
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

    # Build target variable "won_season" from Castaways table. 
    won_season = us_castaways[us_castaways["winner"] == 1.0][["season", "castaway_id", "winner"]].rename(columns={"winner": "won_season"})
    skel = skel.merge(won_season, on=["season", "castaway_id"], how="left")
    skel["won_season"] = skel["won_season"].fillna(0).astype(int)
    
    # Keep useful columns from the skeleton
    skel = skel[
        ["season", "episode", "castaway_id", "castaway", "tribe", "tribe_status",
         "order", "final_n", "eliminated_this_episode", "won_season"]
    ].reset_index(drop=True)
    
    # One hot encode tribe_status:
    skel = pd.get_dummies(skel, columns=["tribe_status"], drop_first=True)
    
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

    df["gender"] = df["gender"].fillna("Unknown") # If any gender is missing, set to "Unknown":
    df = pd.get_dummies(df, columns=["gender"], drop_first=True) # One hot encode gender:

    # Is Returnee & Number of previous seasons: binary indicator, if theyve been in a previous season: 
    appearances = castaways[castaways["version"] == "US"][["season", "castaway_id"]].drop_duplicates()
    appearances["season_rank"] = appearances.groupby("castaway_id")["season"].rank(method="dense")
    appearances["is_returnee"] = (appearances["season_rank"] > 1).astype(int)
    appearances["num_previous_seasons"] = (appearances['season_rank'] - 1).astype(int)
    df = df.merge(appearances[["season", "castaway_id", "is_returnee", "num_previous_seasons"]], on=["season", "castaway_id"], how="left")
    
    # Personality type: split MBTI into 4 binary dimensions rather than 16 one-hot columns
    # E vs I (E = extravert, I = introvert)
    # N vs S (N = intuitive, S = sensing)
    # F vs T (F = feeling, T = thinking)
    # P vs J (P = perceiving, J = judging)
    personality_lookup = details[["castaway_id", "personality_type"]].drop_duplicates()
    personality_lookup["personality_missing"] = personality_lookup["personality_type"].isna().astype(int)
    personality_lookup["mbti_extravert"] = (personality_lookup["personality_type"].str[0] == "E").astype(int)
    personality_lookup["mbti_intuitive"] = (personality_lookup["personality_type"].str[1] == "N").astype(int)
    personality_lookup["mbti_feeling"] = (personality_lookup["personality_type"].str[2] == "F").astype(int)
    personality_lookup["mbti_perceiving"] = (personality_lookup["personality_type"].str[3] == "P").astype(int)

    df = df.merge(
        personality_lookup.drop(columns="personality_type"),
        on="castaway_id", how="left",
    )
    
    # Age rank: relative age among remaining players in each episode
    df["age_rank"] = df.groupby(["season", "episode"])["age"].rank()

    # Interaction effects: 
    df["age_x_episode"] = df["age"] * df["episode"]

    return df
    

def add_vote_features(skel: pd.DataFrame, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Calculates: 
    - votes_against_cumulative_by_previous_ep
    - votes_against_last_3_eps
    - correct_votes_cumulative_by_previous_ep
    """
    
    votes = data["Vote History"]
    
    # limit to US: 
    votes = votes[votes["version"] == "US"]
    df = skel.copy()
    
    # vote_id = who the vote was cast AGAINST (the target)
    # voted_out_id = who actually went home (the result)
    # We want votes received, so group by vote_id
    
    votes_clean = votes.dropna(subset=["vote_id"])  # drop rows with no vote cast (e.g. lost-vote advantage)

    # Votes received (against this player)
    votes_per_ep = (
        votes_clean.groupby(["season", "episode", "vote_id"])
        .size()
        .rename("num_votes_received")
        .reset_index()
        .rename(columns={"vote_id": "castaway_id"})
    )

    df = df.merge(votes_per_ep, on=["season", "episode", "castaway_id"], how="left")
    df["num_votes_received"] = df["num_votes_received"].fillna(0).astype(int)

    # Sort by episode so cumsum/shift/rolling operate in chronological order
    df = df.sort_values(["season", "castaway_id", "episode"]).reset_index(drop=True)

    # Cumulative votes against (shifted to exclude current episode)
    df["votes_against_cumulative"] = df.groupby(["season", "castaway_id"])["num_votes_received"].cumsum()
    df["votes_against_cumulative_by_previous_ep"] = (
        df.groupby(["season", "castaway_id"])["votes_against_cumulative"].shift(1, fill_value=0).astype(int)
    )

    # Votes against in last 3 episodes (shifted, then rolling sum)
    shifted_votes = df.groupby(["season", "castaway_id"])["num_votes_received"].shift(1, fill_value=0)
    df["votes_against_last_3_eps"] = (
        shifted_votes.groupby([df["season"], df["castaway_id"]])
        .transform(lambda x: x.rolling(3, min_periods=1).sum())
        .astype(int)
    )

    # Times in danger: episodes where player received at least 1 vote (cumulative, shifted)
    df["_in_danger"] = (df["num_votes_received"] > 0).astype(int)
    df["_in_danger_cum"] = df.groupby(["season", "castaway_id"])["_in_danger"].cumsum()
    df["times_in_danger"] = (
        df.groupby(["season", "castaway_id"])["_in_danger_cum"].shift(1, fill_value=0).astype(int)
    )

    df = df.drop(columns=["num_votes_received", "votes_against_cumulative", "_in_danger", "_in_danger_cum"])

    # Correct votes (player voted for the person who went home):
    votes_with_correct = votes_clean.copy()
    votes_with_correct["is_correct"] = (votes_with_correct["vote_id"] == votes_with_correct["voted_out_id"]).astype(int)
    correct_per_ep = (
        votes_with_correct.groupby(["season", "episode", "castaway_id"])["is_correct"]
        .sum()
        .rename("correct_votes_ep")
        .reset_index()
    )

    df = df.merge(correct_per_ep, on=["season", "episode", "castaway_id"], how="left")
    df["correct_votes_ep"] = df["correct_votes_ep"].fillna(0).astype(int)

    df["_correct_cumulative"] = df.groupby(["season", "castaway_id"])["correct_votes_ep"].cumsum()
    df["correct_votes_cumulative_by_previous_ep"] = (
        df.groupby(["season", "castaway_id"])["_correct_cumulative"].shift(1, fill_value=0).astype(int)
    )

    df = df.drop(columns=["correct_votes_ep", "_correct_cumulative"])

    return df


def add_challenge_features(skel: pd.DataFrame, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Add cumulative challenge features.

    """
    cr = data["Challenge Results"]
    cr = cr[cr["version"] == "US"]
    df = skel.copy()

    # Individual immunity wins per episode
    indiv_imm = cr[
        cr["challenge_type"].isin(["Immunity", "Immunity and Reward"])
        & (cr["outcome_type"] == "Individual")
        & (cr["result"] == "Won")
    ]
    wins_per_ep = (
        indiv_imm.groupby(["season", "episode", "castaway_id"])
        .size()
        .rename("imm_wins_ep")
        .reset_index()
    )

    df = df.merge(wins_per_ep, on=["season", "episode", "castaway_id"], how="left")
    df["imm_wins_ep"] = df["imm_wins_ep"].fillna(0).astype(int)

    df = df.sort_values(["season", "castaway_id", "episode"]).reset_index(drop=True)

    df["_imm_wins_cumulative"] = df.groupby(["season", "castaway_id"])["imm_wins_ep"].cumsum()
    df["individual_immunity_wins"] = (
        df.groupby(["season", "castaway_id"])["_imm_wins_cumulative"]
        .shift(1, fill_value=0)
        .astype(int)
    )

    df = df.drop(columns=["imm_wins_ep", "_imm_wins_cumulative"])

    return df

def add_confessional_features(skel: pd.DataFrame, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Add confessional / screen time features.

    feature ideas:
    - confessionals_cumulative: total confessional count up to previous episode
    - confessionals_last_3_eps: confessionals in the last 3 episodes (momentum)
    - [x] confessional_share_last_ep: player's share of total confessionals in previous episode
    """
    conf = data["Confessionals"]
    conf = conf[conf["version"] == "US"]
    df = skel.copy()

    # Compute each player's share of confessionals per episode
    ep_totals = conf.groupby(["season", "episode"])["confessional_count"].transform("sum")
    conf_share = conf[["season", "episode", "castaway_id"]].copy()
    conf_share["confessional_share"] = conf["confessional_count"] / ep_totals

    df = df.merge(conf_share, on=["season", "episode", "castaway_id"], how="left")
    df["confessional_share"] = df["confessional_share"].fillna(0)

    df = df.sort_values(["season", "castaway_id", "episode"]).reset_index(drop=True)

    df["confessional_share_last_ep"] = (
        df.groupby(["season", "castaway_id"])["confessional_share"]
        .shift(1, fill_value=0)
    )

    rolling_mean = df.groupby(["season", "castaway_id"])["confessional_share"].transform(
        lambda x: x.rolling(3, min_periods=1).mean()
    )
    df["confessional_share_rolling_3"] = (
        rolling_mean.groupby([df["season"], df["castaway_id"]]).shift(1, fill_value=0)
    )

    expanding_mean = df.groupby(["season", "castaway_id"])["confessional_share"].transform(
        lambda x: x.expanding(min_periods=1).mean()
    )
    df["confessional_share_cumulative"] = (
        expanding_mean.groupby([df["season"], df["castaway_id"]]).shift(1, fill_value=0)
    )

    df = df.drop(columns=["confessional_share"])

    return df


def _build_holding_periods(advantage_events: pd.DataFrame) -> pd.DataFrame:
    """For each advantage in a season, walk through events to determine who holds it and when.

    Returns a DataFrame with columns:
        season, advantage_id, castaway_id, start_ep, end_ep

    Each row represents one continuous holding period. The holder has the advantage
    during episodes [start_ep, end_ep] inclusive.

    Handles transfers (Received closes the previous holder's period),
    beware advantages (second gain by same player is a no-op), and
    unknown future event types (warned, treated as neutral).
    """
    known = _GAIN_EVENTS | _LOSS_EVENTS | _NEUTRAL_EVENTS
    unknown = set(advantage_events["event"].unique()) - known
    if unknown:
        warnings.warn(f"Unknown advantage events (treated as neutral): {unknown}")

    periods: list[dict] = []

    for (season, adv_id), group in advantage_events.groupby(["season", "advantage_id"]):
        group = group.sort_values("episode")
        current_holder = None
        start_ep = None

        for _, row in group.iterrows():
            if row["event"] in _GAIN_EVENTS:
                if current_holder is not None and current_holder != row["castaway_id"]:
                    periods.append({
                        "season": season, "advantage_id": adv_id,
                        "castaway_id": current_holder,
                        "start_ep": start_ep, "end_ep": row["episode"],
                    })
                    
                # If the current holder is not the same as the new holder, update the current holder and start episode
                if current_holder != row["castaway_id"]:
                    current_holder = row["castaway_id"]
                    start_ep = row["episode"]

            elif row["event"] in _LOSS_EVENTS:
                if current_holder is not None:
                    periods.append({
                        "season": season, "advantage_id": adv_id,
                        "castaway_id": current_holder,
                        "start_ep": start_ep, "end_ep": row["episode"],
                    })
                    current_holder = None
                    start_ep = None

    return pd.DataFrame(periods)


def add_advantage_features(skel: pd.DataFrame, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Add advantage features: has_advantage (binary) and advantages_held (count).

    Uses _build_holding_periods to track each advantage's lifecycle through
    gain/loss/transfer events, then counts how many advantages each player holds
    at each episode.
    """
    am = data["Advantage Movement"]
    us_am = am[am["version"] == "US"].copy()
    df = skel.copy()

    periods = _build_holding_periods(us_am)

    if periods.empty:
        df["advantages_held"] = 0
        df["has_advantage"] = 0
        return df

    # Expand holding periods to episode-level: for each period, find which skeleton
    # episodes fall within [start_ep, end_ep]
    episode_keys = df[["season", "episode", "castaway_id"]].drop_duplicates()
    holdings = episode_keys.merge(periods, on=["season", "castaway_id"], how="inner")
    holdings = holdings[
        (holdings["episode"] >= holdings["start_ep"])
        & (holdings["episode"] <= holdings["end_ep"])
    ]

    adv_counts = (
        holdings.groupby(["season", "episode", "castaway_id"])
        .size()
        .rename("advantages_held")
        .reset_index()
    )

    df = df.merge(adv_counts, on=["season", "episode", "castaway_id"], how="left")
    df["advantages_held"] = df["advantages_held"].fillna(0).astype(int)
    df["has_advantage"] = (df["advantages_held"] > 0).astype(int)

    return df


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

    print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} cols")
    print(f"Columns: {df.columns.tolist()}")
    print()

    # Verify: no duplicate (season, episode, castaway_id)
    dupes = df.duplicated(subset=["season", "episode", "castaway_id"]).sum()
    print(f"Duplicate (season, episode, castaway_id) rows: {dupes}")

    # Verify: eliminations per episode
    elim_per_ep = df.groupby(["season", "episode"])["eliminated_this_episode"].sum()
    print(f"Eliminations per episode — mean: {elim_per_ep.mean():.2f}, "
          f"min: {elim_per_ep.min()}, max: {elim_per_ep.max()}")
    print(f"Episodes with 0 eliminations: {(elim_per_ep == 0).sum()}")
    print(f"Episodes with 1 elimination: {(elim_per_ep == 1).sum()}")
    print(f"Episodes with 2+ eliminations: {(elim_per_ep >= 2).sum()}")

    # Verify: static features
    print(f"\nAge — null: {df['age'].isna().sum()}, mean: {df['age'].mean():.1f}")
    # print(f"Gender — null: {df['gender'].isna().sum()}")
    print(f"Gender Male: {df['gender_Male'].sum()}  ")
    print(f"Gender Non-binary: {df['gender_Non-binary'].sum()}")

    # Verify: vote features
    print(f"\nVotes against (cumulative by prev ep) — mean: {df['votes_against_cumulative_by_previous_ep'].mean():.2f}, "
          f"max: {df['votes_against_cumulative_by_previous_ep'].max()}")

    print(f"\nSample rows (season 20, episode 1):")
    print(df[(df["season"] == 20) & (df["episode"] == 1)].to_string())

