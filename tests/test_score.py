import math
import random

import pytest

from opendragon.score import (
    DEL,
    INS,
    OK,
    SUB,
    align,
    corpus_errors,
    format_alignment,
    latency_stats,
    name_accuracy,
    normalise,
    percentile,
    real_time_factor,
    tokens,
    wer,
    word_errors,
)

# ----------------------------------------------------------------- normalise


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Hello, World!", "hello world"),
        ("  lots   of\tspace\n", "lots of space"),
        ("Don't stop.", "don't stop"),
        ("Don’t stop.", "don't stop"),  # curly apostrophe
        ("'quoted' words", "quoted words"),
        ("the users' files", "the users files"),  # trailing possessive apostrophe dropped
        ("e-mail re-run", "e mail re run"),
        ("Whangārei", "whangarei"),
        ("Café naïve Zoë", "cafe naive zoe"),
        ("SIOBHAN", "siobhan"),
        ("What?! No... (really)", "what no really"),
        ("semi;colon:colon", "semi colon colon"),
        ("under_score", "under score"),
        ("", ""),
        ("...", ""),
        ("' ' '", ""),
        ("o'clock rock'n'roll", "o'clock rock'n'roll"),
    ],
)
def test_normalise(raw, expected):
    assert normalise(raw) == expected


def test_normalise_is_idempotent():
    s = "Siobhan's Kubernetes cluster—in Whangārei, ‘obviously’!"
    assert normalise(normalise(s)) == normalise(s)


def test_tokens():
    assert tokens("Hi, Ngaio.") == ["hi", "ngaio"]


# ----------------------------------------------------------------- align / WER


def test_identical():
    r = word_errors("the cat sat", "The cat, sat.")
    assert (r.hits, r.substitutions, r.deletions, r.insertions) == (3, 0, 0, 0)
    assert r.wer == 0.0


def test_one_substitution():
    r = word_errors("the cat sat", "the bat sat")
    assert (r.substitutions, r.deletions, r.insertions) == (1, 0, 0)
    assert r.wer == pytest.approx(1 / 3)
    assert [p.op for p in r.alignment] == [OK, SUB, OK]


def test_one_deletion():
    r = word_errors("the cat sat", "the sat")
    assert (r.substitutions, r.deletions, r.insertions) == (0, 1, 0)
    assert r.alignment[1].op == DEL and r.alignment[1].ref == "cat" and r.alignment[1].hyp is None


def test_one_insertion():
    r = word_errors("the cat sat", "the fat cat sat")
    assert (r.substitutions, r.deletions, r.insertions) == (0, 0, 1)
    assert r.ref_words == 3
    assert r.alignment[1].op == INS and r.alignment[1].hyp == "fat"


def test_empty_hypothesis_is_all_deletions():
    r = word_errors("one two three", "")
    assert r.deletions == 3 and r.wer == 1.0


def test_empty_reference():
    assert word_errors("", "").wer == 0.0
    assert math.isinf(word_errors("", "hello").wer)


def test_wer_can_exceed_one():
    assert wer("hi", "oh hi there friend") == pytest.approx(3.0)


def test_mixed_errors():
    # ref: a b c d e f   hyp: a x c e f g  -> sub b->x, del d, ins g
    r = word_errors("a b c d e f", "a x c e f g")
    assert (r.substitutions, r.deletions, r.insertions) == (1, 1, 1)
    assert r.errors == 3 and r.ref_words == 6


def test_tie_prefers_substitution():
    # Three substitutions and (sub+del+ins) both cost 3; substitutions win the tie.
    r = word_errors("a b c d e", "a x c e f")
    assert r.errors == 3 and r.substitutions == 3


def test_classic_example():
    # A well-known textbook pair: 6 reference words.
    r = word_errors("who is there is it you", "is there it is")
    assert r.errors == 3


def test_alignment_reconstructs_both_sides():
    ref, hyp = "a b c d e f".split(), "a c d x f g".split()
    al = align(ref, hyp)
    assert [p.ref for p in al if p.ref is not None] == ref
    assert [p.hyp for p in al if p.hyp is not None] == hyp


def _edit_distance(a, b):
    """Independent reference implementation (two-row DP)."""
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def test_random_against_reference_implementation():
    rng = random.Random(1234)
    vocab = list("abcde")
    for _ in range(500):
        a = [rng.choice(vocab) for _ in range(rng.randint(0, 9))]
        b = [rng.choice(vocab) for _ in range(rng.randint(0, 9))]
        al = align(a, b)
        assert sum(p.op != OK for p in al) == _edit_distance(a, b)
        assert [p.ref for p in al if p.ref is not None] == a
        assert [p.hyp for p in al if p.hyp is not None] == b
        for p in al:
            if p.op == OK:
                assert p.ref == p.hyp
            elif p.op == SUB:
                assert p.ref != p.hyp


def test_corpus_wer_pools_counts():
    pairs = [("one two three four", "one two three four"), ("hi", "bye")]
    r = corpus_errors(pairs)
    assert r.ref_words == 5 and r.errors == 1
    assert r.wer == pytest.approx(1 / 5)  # not the mean of 0 and 1


def test_as_dict_keys():
    d = word_errors("a b", "a c").as_dict()
    assert d["substitutions"] == 1 and d["wer"] == 0.5 and d["ref_words"] == 2


def test_format_alignment():
    s = format_alignment(word_errors("a b c d", "a x c d e").alignment)
    assert s == "a [b->x] c d [+e]"
    assert format_alignment(word_errors("a b", "a").alignment) == "a [-b]"


# ----------------------------------------------------------------- names


NAMES = ["Siobhan", "Ngaio", "Kubernetes", "Whangārei", "Saoirse Ronan"]


def test_name_accuracy_basic():
    pairs = [
        ("Ask Siobhan about Kubernetes.", "ask siobhan about cooper nettie's"),
        ("Ngaio is driving to Whangarei", "nye oh is driving to whangarei"),
    ]
    r = name_accuracy(pairs, NAMES)
    assert r.per_name["Siobhan"] == (1, 1)
    assert r.per_name["Kubernetes"] == (0, 1)
    assert r.per_name["Ngaio"] == (0, 1)
    assert r.per_name["Whangārei"] == (1, 1)  # macron folded on both sides
    assert r.per_name["Saoirse Ronan"] == (0, 0)
    assert r.correct == 2 and r.total == 4 and r.accuracy == 0.5


def test_name_repeated_in_sentence():
    r = name_accuracy([("Siobhan told Siobhan", "siobhan told shivon")], ["Siobhan"])
    assert r.per_name["Siobhan"] == (1, 2)


def test_name_possessive_is_a_different_token():
    # "Siobhan's" is not the bare name; this keeps scoring strict and predictable.
    r = name_accuracy([("Siobhan's laptop", "siobhan's laptop")], ["Siobhan"])
    assert r.total == 0


def test_multiword_name():
    names = ["Saoirse Ronan"]
    assert name_accuracy([("met Saoirse Ronan today", "met saoirse ronan today")], names).accuracy == 1.0
    assert name_accuracy([("met Saoirse Ronan today", "met sorcha ronan today")], names).accuracy == 0.0
    assert name_accuracy([("met Saoirse Ronan today", "met saoirse a ronan today")], names).accuracy == 0.0
    # Insertion just outside the name doesn't count against it.
    assert name_accuracy([("met Saoirse Ronan today", "met saoirse ronan uh today")], names).accuracy == 1.0


def test_name_accuracy_none_when_absent():
    r = name_accuracy([("plain sentence", "plain sentence")], ["Ngaio"])
    assert r.accuracy is None
    assert r.as_dict()["accuracy"] is None


# ----------------------------------------------------------------- speed


def test_percentile_matches_numpy():
    np = pytest.importorskip("numpy")
    rng = random.Random(7)
    for _ in range(50):
        vals = [rng.uniform(0, 1000) for _ in range(rng.randint(1, 40))]
        for q in (0, 5, 50, 90, 95, 99, 100):
            assert percentile(vals, q) == pytest.approx(float(np.percentile(vals, q)))


def test_percentile_errors():
    with pytest.raises(ValueError):
        percentile([], 50)
    with pytest.raises(ValueError):
        percentile([1.0], 101)


def test_latency_stats():
    s = latency_stats([100, 200, 300, 400, 1000])
    assert s["count"] == 5
    assert s["median"] == 300
    assert s["max"] == 1000
    assert s["mean"] == 400
    assert s["p95"] == pytest.approx(880)


def test_latency_stats_empty():
    assert latency_stats([])["median"] is None


def test_real_time_factor():
    assert real_time_factor([1.0, 2.0], [10.0, 20.0]) == pytest.approx(0.1)
    with pytest.raises(ValueError):
        real_time_factor([1.0], [0.0])
