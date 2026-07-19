from ml.feature_selection.selector_merger import SelectorMerger

def test_merger_union() -> None:
    merger = SelectorMerger(strategy="union")
    selections = {
        "sel1": ["feat1", "feat2"],
        "sel2": ["feat2", "feat3"]
    }
    merged = merger.merge(selections)
    assert merged == ["feat1", "feat2", "feat3"]

def test_merger_intersection() -> None:
    merger = SelectorMerger(strategy="intersection")
    selections = {
        "sel1": ["feat1", "feat2"],
        "sel2": ["feat2", "feat3"]
    }
    merged = merger.merge(selections)
    assert merged == ["feat2"]

def test_merger_voting() -> None:
    # Threshold = 2
    merger = SelectorMerger(strategy="voting", voting_threshold=2)
    selections = {
        "sel1": ["feat1", "feat2", "feat3"],
        "sel2": ["feat2", "feat3", "feat4"],
        "sel3": ["feat3", "feat4", "feat5"]
    }
    merged = merger.merge(selections)
    # feat2 (2 votes), feat3 (3 votes), feat4 (2 votes) should be selected
    # feat1 and feat5 have 1 vote, below threshold 2
    assert merged == ["feat2", "feat3", "feat4"]
