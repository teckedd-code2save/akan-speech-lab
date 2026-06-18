from scripts.build_waxal_viewer_manifest import select_row_indices


def test_seeded_random_sampling_is_reproducible_and_spread_out():
    first = select_row_indices(total=1_000, limit=5, mode="random", seed=42)
    second = select_row_indices(total=1_000, limit=5, mode="random", seed=42)

    assert first == second
    assert len(first) == 5
    assert len(set(first)) == 5
    assert first != list(range(5))


def test_sequential_sampling_honors_start_row_and_bounds():
    assert select_row_indices(total=100, limit=3, mode="sequential", start_row=20) == [20, 21, 22]
    assert select_row_indices(total=5, limit=3, mode="sequential", start_row=99) == [2, 3, 4]
