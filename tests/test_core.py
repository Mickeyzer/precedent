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


def test_agent_list_subheadings_tool():
    from src.agent import run_tool
    out = run_tool("list_subheadings", {"heading": "6109"})
    assert out["heading"] == "6109"
    assert {"610910", "610990"} <= {s["hs6"] for s in out["subheadings"]}
    assert "error" in run_tool("list_subheadings", {"heading": "0000"})


def test_agent_unknown_tool_is_reported_not_raised():
    from src.agent import run_tool
    assert "error" in run_tool("delete_everything", {})


def test_prompt_groups_candidates_by_heading():
    from src.classify import build_prompt
    p = build_prompt("cotton T-shirt", ["610910", "610990", "620520"], [])
    assert p.count("Heading 6109") == 1 and "Heading 6205" in p
    assert "610990:" in p
