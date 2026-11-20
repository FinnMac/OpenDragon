"""Scoring: word error rate, name accuracy and speed statistics.

Plain-English summary
---------------------
* **Normalisation** makes the reference text (what you were asked to say) and the
  recogniser's output comparable: lower-case, accents removed, punctuation removed,
  but apostrophes inside words kept ("don't" stays "don't").
* **Word error rate (WER)** lines the two word sequences up with the fewest edits
  (a "Levenshtein alignment") and counts three kinds of mistake:
    - substitution: a word was heard as a different word,
    - deletion: a word was dropped,
    - insertion: an extra word appeared.
  WER = (substitutions + deletions + insertions) / number of reference words.
* **Name accuracy** looks only at a list of hard words (Siobhan, Ngaio,
  Kubernetes, ...) and reports the share recognised exactly. WER hides these,
  because they are rare.
* **Latency statistics** summarise how long things took: median, 95th
  percentile (the slow end), maximum, and the real-time factor
  (processing time / audio length; below 1 means faster than real time).
"""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Iterable, Sequence

# Every kind of apostrophe or single quote we might meet, mapped to a plain one.
_APOSTROPHES = "’‘ʼʹ`´′"
_APOS_TABLE = str.maketrans({c: "'" for c in _APOSTROPHES})

# Anything that isn't a letter, a digit, an apostrophe or whitespace becomes a space.
_NON_WORD = re.compile(r"[^\w\s']|_", re.UNICODE)


def normalise(text: str) -> str:
    """Return ``text`` in a canonical form for scoring.

    Lower-case, accents and macrons removed (``Whangārei`` -> ``whangarei``),
    punctuation replaced by spaces (so ``e-mail`` -> ``e mail``), apostrophes kept
    only *inside* words (``don't`` stays, ``'quoted'`` -> ``quoted``), and
    whitespace collapsed.
    """
    text = text.translate(_APOS_TABLE)
    # Split accented letters into base letter + accent mark, then drop the marks.
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.casefold()
    text = _NON_WORD.sub(" ", text)
    words = []
    for w in text.split():
        w = w.strip("'")
        if w:
            words.append(w)
    return " ".join(words)


def tokens(text: str) -> list[str]:
    """Normalise ``text`` and split it into words."""
    return normalise(text).split()


# --------------------------------------------------------------------------- #
# Alignment and word error rate
# --------------------------------------------------------------------------- #

# Operation codes in an alignment.
OK, SUB, DEL, INS = "ok", "sub", "del", "ins"


@dataclass(frozen=True)
class AlignedPair:
    """One step of an alignment.

    ``ref`` is ``None`` for an insertion, ``hyp`` is ``None`` for a deletion.
    """

    op: str
    ref: str | None
    hyp: str | None


@dataclass
class WerResult:
    """Counts from aligning one or more reference/hypothesis pairs."""

    ref_words: int = 0
    hits: int = 0
    substitutions: int = 0
    deletions: int = 0
    insertions: int = 0
    alignment: list[AlignedPair] = field(default_factory=list)

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def wer(self) -> float:
        """Word error rate as a fraction (0.15 means 15%).

        With an empty reference, WER is 0 if the hypothesis is empty too and
        infinity otherwise (any output is wrong, but there is nothing to divide by).
        """
        if self.ref_words == 0:
            return 0.0 if self.insertions == 0 else math.inf
        return self.errors / self.ref_words

    def __add__(self, other: "WerResult") -> "WerResult":
        return WerResult(
            ref_words=self.ref_words + other.ref_words,
            hits=self.hits + other.hits,
            substitutions=self.substitutions + other.substitutions,
            deletions=self.deletions + other.deletions,
            insertions=self.insertions + other.insertions,
            alignment=self.alignment + other.alignment,
        )

    def as_dict(self) -> dict:
        return {
            "ref_words": self.ref_words,
            "hits": self.hits,
            "substitutions": self.substitutions,
            "deletions": self.deletions,
            "insertions": self.insertions,
            "errors": self.errors,
            "wer": self.wer,
        }


def align(ref: Sequence[str], hyp: Sequence[str]) -> list[AlignedPair]:
    """Align two word lists with the minimum number of edits (Levenshtein).

    All three edit types cost 1. When several alignments have the same cost,
    a match/substitution is preferred, then a deletion, then an insertion, which
    gives the most natural-looking alignment for speech output.
    """
    n, m = len(ref), len(hyp)
    # cost[i][j] = edits to turn ref[:i] into hyp[:j]
    cost = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        cost[i][0] = i
    for j in range(1, m + 1):
        cost[0][j] = j
    for i in range(1, n + 1):
        ri = ref[i - 1]
        row, prev = cost[i], cost[i - 1]
        for j in range(1, m + 1):
            diag = prev[j - 1] + (0 if ri == hyp[j - 1] else 1)
            row[j] = min(diag, prev[j] + 1, row[j - 1] + 1)

    # Walk back from the bottom-right corner.
    out: list[AlignedPair] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            same = ref[i - 1] == hyp[j - 1]
            if cost[i][j] == cost[i - 1][j - 1] + (0 if same else 1):
                out.append(AlignedPair(OK if same else SUB, ref[i - 1], hyp[j - 1]))
                i, j = i - 1, j - 1
                continue
        if i > 0 and cost[i][j] == cost[i - 1][j] + 1:
            out.append(AlignedPair(DEL, ref[i - 1], None))
            i -= 1
        else:
            out.append(AlignedPair(INS, None, hyp[j - 1]))
            j -= 1
    out.reverse()
    return out


def word_errors(ref: str, hyp: str) -> WerResult:
    """Normalise both texts, align them and count the errors."""
    alignment = align(tokens(ref), tokens(hyp))
    res = WerResult(alignment=alignment)
    for p in alignment:
        if p.op == OK:
            res.hits += 1
        elif p.op == SUB:
            res.substitutions += 1
        elif p.op == DEL:
            res.deletions += 1
        else:
            res.insertions += 1
    res.ref_words = res.hits + res.substitutions + res.deletions
    return res


def wer(ref: str, hyp: str) -> float:
    """Word error rate for a single sentence, as a fraction."""
    return word_errors(ref, hyp).wer


def corpus_errors(pairs: Iterable[tuple[str, str]]) -> WerResult:
    """Pool the counts over many (reference, hypothesis) pairs.

    Corpus WER is total errors / total reference words, *not* the average of
    per-sentence WERs (which would over-weight short sentences).
    """
    total = WerResult()
    for ref, hyp in pairs:
        total = total + word_errors(ref, hyp)
    return total


# --------------------------------------------------------------------------- #
# Name accuracy
# --------------------------------------------------------------------------- #


@dataclass
class NameResult:
    """How often each hard word was recognised exactly."""

    per_name: dict[str, tuple[int, int]] = field(default_factory=dict)  # name -> (correct, total)

    @property
    def correct(self) -> int:
        return sum(c for c, _ in self.per_name.values())

    @property
    def total(self) -> int:
        return sum(t for _, t in self.per_name.values())

    @property
    def accuracy(self) -> float | None:
        """Share of name occurrences recognised exactly; ``None`` if none occurred."""
        return None if self.total == 0 else self.correct / self.total

    def as_dict(self) -> dict:
        return {
            "correct": self.correct,
            "total": self.total,
            "accuracy": self.accuracy,
            "per_name": {k: {"correct": c, "total": t} for k, (c, t) in self.per_name.items()},
        }


def _ref_positions(alignment: Sequence[AlignedPair]) -> list[tuple[int, str, str | None]]:
    """For each reference word, its alignment index, op and the word it was heard as."""
    out = []
    for k, p in enumerate(alignment):
        if p.ref is not None:
            out.append((k, p.op, p.hyp))
    return out


def name_accuracy(pairs: Iterable[tuple[str, str]], names: Iterable[str]) -> NameResult:
    """Count how many occurrences of each name in the references were recognised.

    A name may be several words ("Saoirse Ronan"). An occurrence counts as
    correct only if every one of its words is matched exactly in the alignment
    and nothing was inserted between them.
    """
    name_tokens = {}
    for name in names:
        t = tuple(tokens(name))
        if t:
            name_tokens[name] = t
    result = NameResult(per_name={name: (0, 0) for name in name_tokens})

    for ref, hyp in pairs:
        alignment = word_errors(ref, hyp).alignment
        positions = _ref_positions(alignment)
        ref_words = [p.ref for p in alignment if p.ref is not None]
        for name, nt in name_tokens.items():
            L = len(nt)
            for start in range(len(ref_words) - L + 1):
                if tuple(ref_words[start : start + L]) != nt:
                    continue
                span = positions[start : start + L]
                ok = all(op == OK for _, op, _ in span)
                # No insertions inside a multi-word name.
                if ok and L > 1:
                    first, last = span[0][0], span[-1][0]
                    ok = all(alignment[k].op != INS for k in range(first, last + 1))
                c, t = result.per_name[name]
                result.per_name[name] = (c + int(ok), t + 1)
    return result


# --------------------------------------------------------------------------- #
# Speed
# --------------------------------------------------------------------------- #


def percentile(values: Sequence[float], q: float) -> float:
    """The ``q``-th percentile (0-100) with linear interpolation (numpy's default)."""
    if not values:
        raise ValueError("percentile of an empty list")
    if not 0 <= q <= 100:
        raise ValueError("q must be between 0 and 100")
    s = sorted(values)
    pos = (len(s) - 1) * q / 100
    lo = math.floor(pos)
    hi = math.ceil(pos)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def latency_stats(latencies: Sequence[float]) -> dict:
    """Median, 95th percentile, maximum and mean of a list of latencies.

    Units are whatever you pass in (the baseline uses milliseconds).
    """
    vals = [float(v) for v in latencies]
    if not vals:
        return {"count": 0, "median": None, "p95": None, "max": None, "mean": None}
    return {
        "count": len(vals),
        "median": percentile(vals, 50),
        "p95": percentile(vals, 95),
        "max": max(vals),
        "mean": sum(vals) / len(vals),
    }


def real_time_factor(processing_seconds: Sequence[float], audio_seconds: Sequence[float]) -> float:
    """Total processing time / total audio length.

    0.1 means ten times faster than real time; above 1 means the recogniser
    can't keep up with live speech.
    """
    proc = sum(processing_seconds)
    audio = sum(audio_seconds)
    if audio <= 0:
        raise ValueError("total audio length must be positive")
    return proc / audio


def format_alignment(alignment: Sequence[AlignedPair]) -> str:
    """A compact one-line view: correct words plain, errors as [ref->hyp]."""
    parts = []
    for p in alignment:
        if p.op == OK:
            parts.append(p.ref)
        elif p.op == SUB:
            parts.append(f"[{p.ref}->{p.hyp}]")
        elif p.op == DEL:
            parts.append(f"[-{p.ref}]")
        else:
            parts.append(f"[+{p.hyp}]")
    return " ".join(parts)
