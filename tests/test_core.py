from src import retrieve
from src.fetch_rulings import clean_subject, description
from src.hs import load_hs
from src.metrics import hit_at_k


def test_hs_has_all_subheadings():
    hs = load_hs()
    assert len(hs) == 5613
    assert hs["code"].str.len().eq(6).all()
    assert hs["doc"].str.count(" > ").eq(2).all()


def test_clean_subject_strips_boilerplate():
    assert clean_subject("RE: The tariff classification of a Halloween decoration from China.") == "Halloween decoration"
    assert clean_subject("Country of origin determination for sheets") is None


def test_description_removes_answer():
    text = ("Dear Sir: This is in reply to your request for a binding ruling. The item is a cotton T-shirt. "
            "The applicable subheading for the T-shirt will be 6109.10.0012, HTSUS.")
    d = description(text)
    assert "6109" not in d and "subheading" not in d.lower()
    assert "cotton T-shirt" in d


def test_hit_at_k_levels():
    assert hit_at_k("610910", ["610990", "620520"], 1, 4)
    assert not hit_at_k("610910", ["610990", "620520"], 1, 6)
    assert hit_at_k("610910", ["620520", "610910"], 2, 6)


def test_rrf_prefers_items_ranked_high_by_both():
    import numpy as np
    a = np.array([3.0, 2.0, 1.0])
    b = np.array([1.0, 3.0, 2.0])
    assert retrieve.rrf(a, b).argmax() == 1


def test_bm25_finds_obvious_heading():
    hs = load_hs()
    top = [hs["code"].iat[i] for i in retrieve.search("coffee, roasted, not decaffeinated", "bm25", 5)]
    assert any(c.startswith("0901") for c in top)
