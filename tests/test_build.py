"""Tests for feature building logic."""

import warnings
import pandas as pd
import pytest
from src.features.build import _build_holding_periods, build_modeling_table
from src.load import load_data


def _make_events(rows: list[tuple]) -> pd.DataFrame:
    """Build an advantage_events DataFrame from (season, advantage_id, episode, castaway_id, event) tuples."""
    return pd.DataFrame(rows, columns=["season", "advantage_id", "episode", "castaway_id", "event"])


class TestBuildHoldingPeriods:

    def test_found_and_played(self):
        events = _make_events([
            (41, 1, 3, "PlayerA", "Found"),
            (41, 1, 7, "PlayerA", "Played"),
        ])
        result = _build_holding_periods(events)

        assert len(result) == 1
        assert result.iloc[0]["start_ep"] == 3
        assert result.iloc[0]["end_ep"] == 7

    def test_beware_does_not_double_count(self):
        events = _make_events([
            (49, 1, 2, "PlayerA", "Found (beware)"),
            (49, 1, 4, "PlayerA", "Found"),
            (49, 1, 8, "PlayerA", "Played"),
        ])
        result = _build_holding_periods(events)

        assert len(result) == 1
        assert result.iloc[0]["start_ep"] == 2

    def test_transfer_closes_previous_holder(self):
        events = _make_events([
            (45, 1, 1, "PlayerA", "Found"),
            (45, 1, 9, "PlayerB", "Received"),
            (45, 1, 11, "PlayerB", "Played"),
        ])
        result = _build_holding_periods(events)

        assert len(result) == 2
        a = result[result["castaway_id"] == "PlayerA"].iloc[0]
        b = result[result["castaway_id"] == "PlayerB"].iloc[0]
        assert a["end_ep"] == 9
        assert b["start_ep"] == 9
        assert b["end_ep"] == 11

    def test_found_but_never_lost_emits_no_period(self):
        events = _make_events([
            (50, 1, 3, "PlayerA", "Found"),
        ])
        result = _build_holding_periods(events)
        assert len(result) == 0

    def test_unknown_event_warns_and_treated_as_neutral(self):
        events = _make_events([
            (55, 1, 1, "PlayerA", "Found"),
            (55, 1, 3, "PlayerA", "Stolen"),
            (55, 1, 5, "PlayerA", "Played"),
        ])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = _build_holding_periods(events)
            assert len(w) == 1
            assert "Stolen" in str(w[0].message)

        assert len(result) == 1
        assert result.iloc[0]["end_ep"] == 5


# --- Invariant tests on the full pipeline output ---

@pytest.fixture(scope="module")
def modeling_table():
    """Load real data and build the full modeling table once for all invariant tests."""
    data = load_data()
    return build_modeling_table(data)


class TestPipelineInvariants:

    def test_no_duplicate_rows(self, modeling_table):
        dupes = modeling_table.duplicated(subset=["season", "episode", "castaway_id"]).sum()
        assert dupes == 0

    def test_cumulative_votes_never_decrease(self, modeling_table):
        col = "votes_against_cumulative_by_previous_ep"
        for _, group in modeling_table.groupby(["season", "castaway_id"]):
            vals = group.sort_values("episode")[col].values
            assert all(vals[i] <= vals[i + 1] for i in range(len(vals) - 1))

    def test_cumulative_votes_start_at_zero(self, modeling_table):
        """Every player's first episode in a season should have 0 cumulative votes."""
        first_eps = modeling_table.sort_values("episode").groupby(["season", "castaway_id"]).first()
        assert (first_eps["votes_against_cumulative_by_previous_ep"] == 0).all()
        assert (first_eps["correct_votes_cumulative_by_previous_ep"] == 0).all()
        assert (first_eps["times_in_danger"] == 0).all()

    def test_has_advantage_is_binary(self, modeling_table):
        assert modeling_table["has_advantage"].isin([0, 1]).all()

    def test_advantages_held_non_negative(self, modeling_table):
        assert (modeling_table["advantages_held"] >= 0).all()

    def test_target_has_no_nulls(self, modeling_table):
        assert modeling_table["eliminated_this_episode"].isna().sum() == 0
