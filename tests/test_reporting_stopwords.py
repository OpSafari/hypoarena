"""Reporting verbs carry no entity information and must not dominate matching.

Scientific finding statements differ mostly in how they *report* ("we observed
that", "assays show that") rather than in what they claim. Keeping those words
in the content-token set makes unrelated findings look similar and paraphrases
look different, which is why they are treated as stopwords here. The change is
pinned by tests so it cannot be reverted silently.
"""

from __future__ import annotations

from hypoarena.dedup import jaccard_similarity
from hypoarena.text import REPORTING_WORDS, STOPWORDS, content_tokens, jaccard

PLAIN = "gene G3 is required for metabolite M1 in mouse liver"
REPORTED = "We observed that gene G3 is required for metabolite M1 in mouse liver"
ASSAYED = "Assays in mouse liver show that gene G3 is required for metabolite M1"
UNRELATED = "kinase K1 inhibits protein A in HeLa cells"


def test_reporting_words_are_stopwords() -> None:
    assert "observed" in STOPWORDS
    assert "assays" in REPORTING_WORDS
    assert REPORTING_WORDS < STOPWORDS


def test_reporting_framing_does_not_add_content_tokens() -> None:
    assert content_tokens(PLAIN) == content_tokens(REPORTED)
    assert content_tokens("we do not observe any effect") == ["effect"]


def test_entity_tokens_survive() -> None:
    assert content_tokens(REPORTED) == [
        "gene",
        "g3",
        "required",
        "metabolite",
        "m1",
        "mouse",
        "liver",
    ]


def test_paraphrases_come_closer_than_unrelated_findings() -> None:
    reported = set(content_tokens(REPORTED))
    paraphrase = jaccard(reported, set(content_tokens(ASSAYED)))
    unrelated = jaccard(reported, set(content_tokens(UNRELATED)))
    assert paraphrase > unrelated
    assert paraphrase > 0.6


def test_word_level_similarity_of_reported_paraphrases_is_high() -> None:
    assert jaccard_similarity(REPORTED, ASSAYED, n=1, words=True) > 0.7
    assert jaccard_similarity(REPORTED, UNRELATED, n=1, words=True) < 0.2
