"""Split geometry: the small-train default, the refusal to fit on most of the data, and the
embargo actually removing bars."""

from __future__ import annotations

import pytest

from holdout_first.splits import Split, fraction_split, walk_forward_periods


def test_default_train_fraction_holds_out_the_majority() -> None:
    split = fraction_split(100)
    assert split.train_length == 30
    assert split.test_length == 70
    assert split.test_length > split.train_length


def test_fraction_split_produces_contiguous_ranges_in_time_order() -> None:
    split = fraction_split(100, 0.25)
    assert split == Split(train_start=0, train_stop=25, test_start=25, test_stop=100, gap=0)
    assert split.train_slice == slice(0, 25)
    assert split.test_slice == slice(25, 100)


def test_fraction_split_refuses_to_fit_on_more_than_half_the_data() -> None:
    with pytest.raises(ValueError, match="allow_large_train"):
        fraction_split(100, 0.70)


def test_fraction_split_allows_a_large_train_only_via_the_escape_hatch() -> None:
    split = fraction_split(100, 0.70, allow_large_train=True)
    assert split.train_length == 70
    assert split.test_length == 30


def test_fraction_split_accepts_exactly_one_half() -> None:
    split = fraction_split(100, 0.5)
    assert split.train_length == 50


@pytest.mark.parametrize("train_fraction", [0.0, 1.0, -0.1, 1.5])
def test_fraction_split_rejects_fractions_outside_the_open_unit_interval(
    train_fraction: float,
) -> None:
    with pytest.raises(ValueError, match="strictly inside"):
        fraction_split(100, train_fraction, allow_large_train=True)


def test_embargo_removes_exactly_the_requested_number_of_bars() -> None:
    plain = fraction_split(100, 0.30)
    embargoed = fraction_split(100, 0.30, gap=7)
    assert embargoed.train_stop == plain.train_stop
    assert embargoed.test_start - embargoed.train_stop == 7
    assert embargoed.test_length == plain.test_length - 7
    assert embargoed.train_length + 7 + embargoed.test_length == 100


def test_embargo_bars_belong_to_neither_side() -> None:
    split = fraction_split(50, 0.40, gap=4)
    train = set(range(split.train_start, split.train_stop))
    test = set(range(split.test_start, split.test_stop))
    assert train.isdisjoint(test)
    assert len(train | test) == 50 - 4


def test_offset_shifts_every_index_without_changing_lengths() -> None:
    base = fraction_split(60, 0.30, gap=3)
    shifted = fraction_split(60, 0.30, gap=3, offset=1000)
    assert shifted.train_start == base.train_start + 1000
    assert shifted.test_stop == base.test_stop + 1000
    assert shifted.train_length == base.train_length
    assert shifted.test_length == base.test_length


def test_fraction_split_rejects_a_gap_that_leaves_no_held_out_bars() -> None:
    with pytest.raises(ValueError, match="embargo"):
        fraction_split(20, 0.30, gap=14)


def test_fraction_split_rejects_a_train_fraction_that_rounds_to_nothing() -> None:
    with pytest.raises(ValueError, match="empty training range"):
        fraction_split(10, 0.05)


@pytest.mark.parametrize("bad", [-1, 0, 1])
def test_fraction_split_rejects_series_too_short_to_split(bad: int) -> None:
    with pytest.raises(ValueError, match="at least 2"):
        fraction_split(bad)


def test_fraction_split_rejects_negative_gap() -> None:
    with pytest.raises(ValueError, match="gap must be non-negative"):
        fraction_split(100, gap=-1)


def test_fraction_split_rejects_non_integer_n() -> None:
    with pytest.raises(TypeError, match="n must be an int"):
        fraction_split(100.0)  # type: ignore[arg-type]


def test_fraction_split_rejects_non_integer_gap() -> None:
    with pytest.raises(TypeError, match="gap must be an int"):
        fraction_split(100, gap=1.5)  # type: ignore[arg-type]


def test_fraction_split_rejects_non_integer_offset() -> None:
    with pytest.raises(TypeError, match="offset must be an int"):
        fraction_split(100, offset=1.5)  # type: ignore[arg-type]


def test_fraction_split_rejects_negative_offset() -> None:
    with pytest.raises(ValueError, match="offset must be non-negative"):
        fraction_split(100, offset=-1)


def test_walk_forward_blocks_are_disjoint_and_chronological() -> None:
    splits = walk_forward_periods(100, 4, 0.25)
    assert len(splits) == 4
    covered: set[int] = set()
    previous_stop = 0
    for split in splits:
        assert split.train_start >= previous_stop
        indices = set(range(split.train_start, split.train_stop)) | set(
            range(split.test_start, split.test_stop)
        )
        assert covered.isdisjoint(indices)
        covered |= indices
        previous_stop = split.test_stop
    assert covered == set(range(100))


def test_walk_forward_distributes_the_remainder_across_blocks() -> None:
    splits = walk_forward_periods(101, 3, 0.30)
    spans = [split.test_stop - split.train_start for split in splits]
    assert sum(spans) == 101
    assert max(spans) - min(spans) <= 1


def test_walk_forward_applies_the_embargo_inside_every_block() -> None:
    splits = walk_forward_periods(300, 3, 0.30, gap=5)
    assert all(split.test_start - split.train_stop == 5 for split in splits)
    assert all(split.gap == 5 for split in splits)


def test_walk_forward_of_one_period_matches_a_plain_split() -> None:
    assert walk_forward_periods(100, 1, 0.30) == [fraction_split(100, 0.30)]


def test_walk_forward_inherits_the_large_train_refusal() -> None:
    with pytest.raises(ValueError, match="allow_large_train"):
        walk_forward_periods(300, 3, 0.60)


def test_walk_forward_names_the_period_that_is_too_short() -> None:
    with pytest.raises(ValueError, match="period 0 spans bars"):
        walk_forward_periods(30, 5, 0.30, gap=5)


@pytest.mark.parametrize("n_periods", [0, -3])
def test_walk_forward_rejects_non_positive_period_counts(n_periods: int) -> None:
    with pytest.raises(ValueError, match="n_periods must be at least 1"):
        walk_forward_periods(100, n_periods)


def test_walk_forward_rejects_non_integer_period_count() -> None:
    with pytest.raises(TypeError, match="n_periods must be an int"):
        walk_forward_periods(100, 3.0)  # type: ignore[arg-type]
