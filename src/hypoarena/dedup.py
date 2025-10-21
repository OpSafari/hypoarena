"""Duplicate and near-duplicate detection for hypotheses and findings.

Four levels are provided, cheapest first: exact matching on normalized text,
character/word n-gram Jaccard, TF-IDF cosine, and MinHash with LSH banding for
large candidate sets. Every function is deterministic and pure; the LSH stage
only *proposes* candidate pairs, and a cluster is never formed without verifying
the pair with the configured similarity metric.

Documented behaviour rather than promises: MinHash estimates Jaccard similarity
with a standard error of about ``1 / sqrt(num_perm)``, and LSH banding trades
false negatives (dissimilar pairs proposed) for false positives (similar pairs
missed) around the ``threshold ≈ (1 / bands) ** (1 / rows)`` inflection.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass

import numpy as np

from hypoarena.errors import (
    DuplicateIdError,
    ValidationError,
)
from hypoarena.ids import (
    content_hash,
)
from hypoarena.text import (
    char_ngrams,
    content_tokens,
    normalize,
    tokenize,
    word_ngrams,
)


def normalized_signature(text: str) -> str:
    """Return the digest of a normalized text.

    Two texts that differ only by case, punctuation or whitespace share a
    signature, which is what makes exact duplicate detection robust to the
    surface variation a paraphrase cluster introduces.
    """
    return content_hash(normalize(text))


def exact_duplicates(texts: Mapping[str, str]) -> tuple[tuple[str, ...], ...]:
    """Group identifiers whose normalized texts are identical.

    Groups with a single member are omitted, and both the members and the groups
    are sorted so the result never depends on dictionary order.
    """
    groups: dict[str, list[str]] = {}
    for identifier, text in texts.items():
        groups.setdefault(normalized_signature(text), []).append(identifier)
    return tuple(
        sorted(tuple(sorted(group)) for group in groups.values() if len(group) > 1)
    )


def duplicate_rate(texts: Mapping[str, str]) -> float:
    """Return the fraction of items that share text with another item."""
    if not texts:
        return 0.0
    duplicated = sum(len(group) for group in exact_duplicates(texts))
    return duplicated / len(texts)


def shingles(text: str, n: int, *, words: bool = False) -> frozenset[str]:
    """Return the character or word n-gram set of a normalized text.

    Character shingles are the default because they survive word-order changes
    better than single tokens; word shingles (``words=True``) are what the TF-IDF
    stage uses. Short texts fall back to the whole normalized string so that very
    short inputs still compare equal to themselves.
    """
    if n < 1:
        raise ValidationError("shingle size must be >= 1", n=n)
    if words:
        return frozenset(word_ngrams(tokenize(text), n))
    return frozenset(char_ngrams(text, n))


def jaccard_similarity(
    left: str,
    right: str,
    *,
    n: int = 3,
    words: bool = False,
    content_only: bool = False,
) -> float:
    """Return the Jaccard index of two texts' shingle sets.

    Two empty texts score ``1.0`` (they are the same nothing); an empty text
    against a non-empty one scores ``0.0``.
    """
    first = shingle_set(left, n, words=words, content_only=content_only)
    second = shingle_set(right, n, words=words, content_only=content_only)
    if not first and not second:
        return 1.0
    if not first or not second:
        return 0.0
    return len(first & second) / len(first | second)


def similarity_matrix(
    texts: Sequence[str], *, n: int = 3, words: bool = False
) -> list[list[float]]:
    """Return the pairwise Jaccard matrix of a list of texts."""
    sets = [shingles(text, n, words=words) for text in texts]
    matrix: list[list[float]] = []
    for first in sets:
        row = []
        for second in sets:
            if not first and not second:
                row.append(1.0)
            elif not first or not second:
                row.append(0.0)
            else:
                row.append(len(first & second) / len(first | second))
        matrix.append(row)
    return matrix


def term_counts(text: str, ngram_size: int) -> dict[str, int]:
    """Count word n-grams (or single tokens when ``ngram_size`` is 1)."""
    if ngram_size < 1:
        raise ValidationError("ngram_size must be >= 1", ngram_size=ngram_size)
    terms = (
        tokenize(text) if ngram_size == 1 else word_ngrams(tokenize(text), ngram_size)
    )
    counts: dict[str, int] = {}
    for term in terms:
        counts[term] = counts.get(term, 0) + 1
    return counts


class TfidfVectorizer:
    """Term frequency / inverse document frequency vectors over word n-grams.

    ``idf`` uses the smoothed form ``log((1 + n_docs) / (1 + df)) + 1`` so a term
    present in every document still contributes (with weight 1) instead of
    vanishing. Rows are L2-normalized, which turns a dot product into a cosine
    similarity. ``min_document_frequency`` drops rare terms, the usual defence
    against identifiers and typos dominating the vocabulary.
    """

    def __init__(self, ngram_size: int = 1, min_document_frequency: int = 1) -> None:
        if ngram_size < 1:
            raise ValidationError("ngram_size must be >= 1", ngram_size=ngram_size)
        if min_document_frequency < 1:
            raise ValidationError(
                "min_document_frequency must be >= 1",
                min_document_frequency=min_document_frequency,
            )
        self.ngram_size = ngram_size
        self.min_document_frequency = min_document_frequency
        self.vocabulary_: tuple[str, ...] = ()
        self.idf_: np.ndarray = np.zeros(0, dtype=np.float64)
        self.fitted_ = False

    def fit(self, documents: Sequence[str]) -> TfidfVectorizer:
        """Learn the vocabulary and inverse document frequencies."""
        frequencies: dict[str, int] = {}
        for document in documents:
            for term in set(term_counts(document, self.ngram_size)):
                frequencies[term] = frequencies.get(term, 0) + 1
        kept = sorted(
            term
            for term, count in frequencies.items()
            if count >= self.min_document_frequency
        )
        total = len(documents)
        self.vocabulary_ = tuple(kept)
        self.idf_ = np.array(
            [
                math.log((1 + total) / (1 + frequencies[term])) + 1.0
                for term in self.vocabulary_
            ],
            dtype=np.float64,
        )
        self.fitted_ = True
        return self

    def transform(self, documents: Sequence[str]) -> np.ndarray:
        """Return L2-normalized TF-IDF rows for ``documents``."""
        if not self.fitted_:
            raise ValidationError("vectorizer must be fitted before transforming")
        index = {term: position for position, term in enumerate(self.vocabulary_)}
        matrix = np.zeros((len(documents), len(self.vocabulary_)), dtype=np.float64)
        for row, document in enumerate(documents):
            for term, count in term_counts(document, self.ngram_size).items():
                position = index.get(term)
                if position is not None:
                    matrix[row, position] = count * self.idf_[position]
        return l2_normalize(matrix)

    def fit_transform(self, documents: Sequence[str]) -> np.ndarray:
        """Fit on ``documents`` and return their vectors."""
        return self.fit(documents).transform(documents)


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Scale each row to unit length, leaving zero rows untouched."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    safe = np.where(norms == 0.0, 1.0, norms)
    return matrix / safe


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    """Return the cosine of two equally shaped vectors.

    Two zero vectors score ``1.0`` and a zero vector against a non-zero one
    scores ``0.0``, matching the convention used by :func:`jaccard_similarity`.
    """
    if left.shape != right.shape:
        raise ValidationError(
            "cosine needs equally shaped vectors",
            left=list(left.shape),
            right=list(right.shape),
        )
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm == 0.0 and right_norm == 0.0:
        return 1.0
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return float(np.dot(left, right) / (left_norm * right_norm))


def cosine_matrix(vectors: np.ndarray) -> np.ndarray:
    """Return the pairwise cosine matrix of L2-normalized rows."""
    normalized = l2_normalize(vectors)
    return normalized @ normalized.T


EMPTY_SIGNATURE_VALUE = 0
MAX_HASH = (1 << 61) - 1


def hash_shingle(shingle: str, permutation: int) -> int:
    """Return a deterministic 61-bit hash of one shingle under one permutation."""
    if permutation < 0:
        raise ValidationError("permutation index must be >= 0", permutation=permutation)
    return int(content_hash([permutation, shingle], length=16), 16) % MAX_HASH


def minhash_signature(
    shingle_set: frozenset[str], num_perm: int, *, seed: int = 0
) -> tuple[int, ...]:
    """Return the MinHash signature of a shingle set.

    Slot ``i`` holds the smallest hash of any shingle under permutation ``i``;
    the seed shifts the permutation family so independent runs can be compared.
    An empty set yields a signature of zeros, and two empty signatures compare
    equal — the same convention the exact similarities use.
    """
    if num_perm < 1:
        raise ValidationError("num_perm must be >= 1", num_perm=num_perm)
    if not shingle_set:
        return tuple([EMPTY_SIGNATURE_VALUE] * num_perm)
    return tuple(
        min(hash_shingle(shingle, seed * 4096 + index) for shingle in shingle_set)
        for index in range(num_perm)
    )


def text_signature(
    text: str,
    num_perm: int,
    *,
    n: int = 3,
    words: bool = False,
    content_only: bool = False,
    seed: int = 0,
) -> tuple[int, ...]:
    """Return the MinHash signature of a text's shingles."""
    return minhash_signature(
        shingle_set(text, n, words=words, content_only=content_only),
        num_perm,
        seed=seed,
    )


def minhash_similarity(left: Sequence[int], right: Sequence[int]) -> float:
    """Return the fraction of agreeing slots, an estimate of Jaccard similarity."""
    if len(left) != len(right):
        raise ValidationError(
            "signatures must have the same length",
            left=len(left),
            right=len(right),
        )
    if not left:
        raise ValidationError("signatures must not be empty")
    agreeing = sum(
        1 for first, second in zip(left, right, strict=True) if first == second
    )
    return agreeing / len(left)


def signature_standard_error(num_perm: int) -> float:
    """Return the standard error of a MinHash estimate: ``1 / sqrt(num_perm)``.

    With 64 permutations the estimate is typically within about 0.125 of the true
    Jaccard index (one standard error); 256 permutations tighten that to 0.0625.
    Callers comparing near a threshold should size ``num_perm`` from this bound
    rather than guessing.
    """
    if num_perm < 1:
        raise ValidationError("num_perm must be >= 1", num_perm=num_perm)
    return 1.0 / math.sqrt(num_perm)


def lsh_rows(num_perm: int, bands: int) -> int:
    """Return the rows per band, requiring ``num_perm`` to divide evenly."""
    if bands < 1:
        raise ValidationError("bands must be >= 1", bands=bands)
    if num_perm < 1:
        raise ValidationError("num_perm must be >= 1", num_perm=num_perm)
    if num_perm % bands:
        raise ValidationError(
            "num_perm must be divisible by bands", num_perm=num_perm, bands=bands
        )
    return num_perm // bands


def lsh_bands(signature: Sequence[int], bands: int) -> tuple[tuple[int, ...], ...]:
    """Split a signature into ``bands`` row tuples used as bucket keys."""
    rows = lsh_rows(len(signature), bands)
    return tuple(
        tuple(signature[index * rows : (index + 1) * rows]) for index in range(bands)
    )


def candidate_pairs(
    signatures: Mapping[str, tuple[int, ...]], bands: int
) -> set[tuple[str, str]]:
    """Return the pairs that share at least one band bucket.

    LSH is a *filter*: every returned pair still has to be verified with the
    configured similarity metric before it may join a cluster. The set contains
    sorted identifier pairs, so downstream iteration is deterministic.
    """
    buckets: dict[tuple[int, tuple[int, ...]], list[str]] = {}
    for identifier, signature in signatures.items():
        for band_index, band in enumerate(lsh_bands(signature, bands)):
            buckets.setdefault((band_index, band), []).append(identifier)
    pairs: set[tuple[str, str]] = set()
    for members in buckets.values():
        if len(members) < 2:
            continue
        ordered = sorted(members)
        for index, first in enumerate(ordered):
            for second in ordered[index + 1 :]:
                pairs.add((first, second))
    return pairs


def candidate_probability(similarity: float, num_perm: int, bands: int) -> float:
    """Return the probability that a pair becomes an LSH candidate.

    For similarity ``s``, ``b`` bands of ``r`` rows the standard S-curve applies:
    ``1 - (1 - s**r)**b``. This is the formula behind the documented
    false-positive/false-negative trade-off — pairs well above the inflection are
    almost always proposed, pairs well below almost never are, and the transition
    is where threshold tuning matters.
    """
    if not 0.0 <= similarity <= 1.0:
        raise ValidationError(
            "similarity must lie within [0, 1]", similarity=similarity
        )
    rows = lsh_rows(num_perm, bands)
    return 1.0 - (1.0 - similarity**rows) ** bands


def lsh_inflection(num_perm: int, bands: int) -> float:
    """Return the similarity at which the S-curve is steepest.

    ``(1 / bands) ** (1 / rows)`` is the usual rule of thumb for the threshold a
    banding configuration approximates; choosing ``bands`` so this value matches
    the intended threshold is how the trade-off is controlled.
    """
    rows = lsh_rows(num_perm, bands)
    return (1.0 / bands) ** (1.0 / rows)


DEDUP_METHODS = ("exact", "jaccard", "tfidf", "minhash")
DEFAULT_NUM_PERM = 128
DEFAULT_BANDS = 32
DEFAULT_THRESHOLD = 0.8


@dataclass(frozen=True)
class DedupConfig:
    """Which similarity metric to use and how aggressively to cluster.

    ``threshold`` is the verified similarity a pair must reach to be merged. For
    the ``minhash`` method, ``bands`` controls the LSH filter: a higher band
    count proposes more candidates (fewer false negatives, more verification
    work) and :func:`lsh_inflection` gives the similarity the configuration
    approximates.
    """

    method: str = "minhash"
    threshold: float = DEFAULT_THRESHOLD
    ngram_size: int = 3
    word_ngram_size: int = 1
    num_perm: int = DEFAULT_NUM_PERM
    bands: int = DEFAULT_BANDS
    min_document_frequency: int = 1
    use_lsh: bool = True
    shingle_unit: str = "char"
    content_only: bool = False
    seed: int = 0

    def __post_init__(self) -> None:
        if self.method not in DEDUP_METHODS:
            raise ValidationError(
                "unknown dedup method", method=self.method, allowed=list(DEDUP_METHODS)
            )
        if not 0.0 < self.threshold <= 1.0:
            raise ValidationError(
                "threshold must lie within (0, 1]", threshold=self.threshold
            )
        for name in ("ngram_size", "word_ngram_size", "num_perm", "bands"):
            if getattr(self, name) < 1:
                raise ValidationError(
                    f"{name} must be >= 1", **{name: getattr(self, name)}
                )
        if self.min_document_frequency < 1:
            raise ValidationError(
                "min_document_frequency must be >= 1",
                min_document_frequency=self.min_document_frequency,
            )
        if self.shingle_unit not in SHINGLE_UNITS:
            raise ValidationError(
                "unknown shingle_unit",
                shingle_unit=self.shingle_unit,
                allowed=list(SHINGLE_UNITS),
            )
        if self.use_lsh and self.num_perm % self.bands:
            raise ValidationError(
                "num_perm must be divisible by bands when LSH is used",
                num_perm=self.num_perm,
                bands=self.bands,
            )

    @property
    def uses_words(self) -> bool:
        """True when shingles are word n-grams rather than character n-grams."""
        return self.shingle_unit == "word"

    @property
    def rows(self) -> int:
        """Rows per LSH band for this configuration."""
        return lsh_rows(self.num_perm, self.bands)

    def inflection(self) -> float:
        """Return the similarity this banding approximates."""
        return lsh_inflection(self.num_perm, self.bands)

    def fingerprint(self) -> str:
        """Return a digest of the configuration for run metadata."""
        return content_hash(asdict(self))


@dataclass(frozen=True)
class DuplicateCluster:
    """A set of identifiers judged to restate the same content."""

    members: tuple[str, ...]
    representative: str
    method: str
    similarities: tuple[tuple[str, str, float], ...] = ()

    def __post_init__(self) -> None:
        if len(self.members) < 2:
            raise ValidationError(
                "a duplicate cluster needs at least two members",
                count=len(self.members),
            )
        if list(self.members) != sorted(set(self.members)):
            raise ValidationError(
                "cluster members must be sorted and unique", members=list(self.members)
            )
        if self.representative not in self.members:
            raise ValidationError(
                "cluster representative must be a member",
                representative=self.representative,
            )

    @property
    def size(self) -> int:
        """Number of members."""
        return len(self.members)

    def contains(self, identifier: str) -> bool:
        """True when an identifier belongs to this cluster."""
        return identifier in self.members

    def pairs(self) -> tuple[tuple[str, str], ...]:
        """Return every unordered member pair, sorted."""
        return tuple(
            (first, second)
            for index, first in enumerate(self.members)
            for second in self.members[index + 1 :]
        )

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-ready view used by reports and artifacts."""
        return {
            "representative": self.representative,
            "method": self.method,
            "size": self.size,
            "members": list(self.members),
            "similarities": [
                {"left": left, "right": right, "score": round(score, 6)}
                for left, right, score in self.similarities
            ],
        }


@dataclass(frozen=True)
class ClusterMetrics:
    """Pair-based precision and recall of a clustering against planted truth."""

    clusters: int
    members: int
    pairs_found: int
    pairs_expected: int
    true_positives: int
    false_positives: int
    false_negatives: int

    @property
    def precision(self) -> float:
        """Fraction of discovered pairs that really are duplicates."""
        if not self.pairs_found:
            return 1.0
        return self.true_positives / self.pairs_found

    @property
    def recall(self) -> float:
        """Fraction of planted pairs that were discovered."""
        if not self.pairs_expected:
            return 1.0
        return self.true_positives / self.pairs_expected

    @property
    def f1(self) -> float:
        """Harmonic mean of precision and recall."""
        precision, recall = self.precision, self.recall
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    def as_dict(self) -> dict[str, float]:
        """Return a JSON-ready view for reports."""
        return {
            "clusters": self.clusters,
            "members": self.members,
            "pairs_found": self.pairs_found,
            "pairs_expected": self.pairs_expected,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
        }


def cluster_metrics(
    clusters: Sequence[DuplicateCluster], truth: Sequence[Iterable[str]]
) -> ClusterMetrics:
    """Score a clustering against planted groups, counting identifier pairs.

    Pair counting is used instead of cluster counting because a run that splits
    one planted group in two and another that merges two groups are different
    failures, and only the pair view distinguishes them.
    """
    found = {pair for cluster in clusters for pair in cluster.pairs()}
    expected: set[tuple[str, str]] = set()
    for group in truth:
        members = tuple(sorted(set(group)))
        for index, first in enumerate(members):
            for second in members[index + 1 :]:
                expected.add((first, second))
    true_positives = len(found & expected)
    return ClusterMetrics(
        clusters=len(clusters),
        members=sum(cluster.size for cluster in clusters),
        pairs_found=len(found),
        pairs_expected=len(expected),
        true_positives=true_positives,
        false_positives=len(found - expected),
        false_negatives=len(expected - found),
    )


def union_groups(
    identifiers: Sequence[str], pairs: Iterable[tuple[str, str]]
) -> list[list[str]]:
    """Group identifiers by an equivalence relation given as pairs.

    Union-find with path compression; groups are returned sorted by their first
    member and members are sorted, so the output never depends on pair order.
    """
    parent = {identifier: identifier for identifier in identifiers}

    def find(identifier: str) -> str:
        while parent[identifier] != identifier:
            parent[identifier] = parent[parent[identifier]]
            identifier = parent[identifier]
        return identifier

    for left, right in pairs:
        first, second = find(left), find(right)
        if first != second:
            parent[max(first, second)] = min(first, second)
    groups: dict[str, list[str]] = {}
    for identifier in identifiers:
        groups.setdefault(find(identifier), []).append(identifier)
    return [sorted(group) for group in sorted(groups.values(), key=min)]


class DuplicateFinder:
    """Finds duplicate groups with the metric chosen in the configuration.

    For ``minhash`` the LSH filter proposes candidates and every candidate is
    then verified with the actual MinHash estimate, so a band collision alone
    never creates a cluster. ``exact`` skips verification because normalized
    equality is already exact.
    """

    def __init__(self, config: DedupConfig | None = None) -> None:
        self.config = config or DedupConfig()

    def report(self, texts: Mapping[str, str]) -> DedupReport:
        """Run :meth:`find` and bundle the outcome with this configuration."""
        return DedupReport(self.find(texts), self.config, len(texts))

    def find(self, texts: Mapping[str, str]) -> tuple[DuplicateCluster, ...]:
        """Return duplicate clusters, sorted by representative."""
        identifiers = sorted(texts)
        if len(identifiers) < 2:
            return ()
        method = self.config.method
        if method == "exact":
            verified = {
                (left, right)
                for group in exact_duplicates(texts)
                for index, left in enumerate(group)
                for right in group[index + 1 :]
            }
            scores: dict[tuple[str, str], float] = {pair: 1.0 for pair in verified}
        elif method == "tfidf":
            verified, scores = self._verify_tfidf(identifiers, texts)
        else:
            verified, scores = self._verify_signatures(identifiers, texts)
        groups = union_groups(identifiers, sorted(verified))
        clusters = []
        for members in groups:
            if len(members) < 2:
                continue
            similarities = tuple(
                (left, right, scores[(left, right)])
                for index, left in enumerate(members)
                for right in members[index + 1 :]
                if (left, right) in scores
            )
            clusters.append(
                DuplicateCluster(
                    members=tuple(members),
                    representative=members[0],
                    method=method,
                    similarities=similarities,
                )
            )
        return tuple(sorted(clusters, key=lambda cluster: cluster.representative))

    def _candidate_pairs(
        self, identifiers: Sequence[str], texts: Mapping[str, str]
    ) -> set[tuple[str, str]]:
        """Return the pairs worth comparing under the configured method."""
        if self.config.method != "minhash" or not self.config.use_lsh:
            return {
                (left, right)
                for index, left in enumerate(identifiers)
                for right in identifiers[index + 1 :]
            }
        signatures = {
            identifier: text_signature(
                texts[identifier],
                self.config.num_perm,
                n=self.config.ngram_size,
                words=self.config.uses_words,
                content_only=self.config.content_only,
                seed=self.config.seed,
            )
            for identifier in identifiers
        }
        return candidate_pairs(signatures, self.config.bands)

    def _verify_signatures(
        self, identifiers: Sequence[str], texts: Mapping[str, str]
    ) -> tuple[set[tuple[str, str]], dict[tuple[str, str], float]]:
        """Verify candidate pairs with Jaccard or MinHash similarity."""
        verified: set[tuple[str, str]] = set()
        scores: dict[tuple[str, str], float] = {}
        if self.config.method == "jaccard":
            for left, right in self._candidate_pairs(identifiers, texts):
                score = jaccard_similarity(
                    texts[left],
                    texts[right],
                    n=self.config.ngram_size,
                    words=self.config.uses_words,
                    content_only=self.config.content_only,
                )
                if score >= self.config.threshold:
                    verified.add((left, right))
                    scores[(left, right)] = score
            return verified, scores
        signatures = {
            identifier: text_signature(
                texts[identifier],
                self.config.num_perm,
                n=self.config.ngram_size,
                words=self.config.uses_words,
                content_only=self.config.content_only,
                seed=self.config.seed,
            )
            for identifier in identifiers
        }
        for left, right in self._candidate_pairs(identifiers, texts):
            score = minhash_similarity(signatures[left], signatures[right])
            if score >= self.config.threshold:
                verified.add((left, right))
                scores[(left, right)] = score
        return verified, scores

    def _verify_tfidf(
        self, identifiers: Sequence[str], texts: Mapping[str, str]
    ) -> tuple[set[tuple[str, str]], dict[tuple[str, str], float]]:
        """Verify every pair with cosine similarity over TF-IDF vectors."""
        vectorizer = TfidfVectorizer(
            ngram_size=self.config.word_ngram_size,
            min_document_frequency=self.config.min_document_frequency,
        )
        vectors = vectorizer.fit_transform([texts[item] for item in identifiers])
        verified: set[tuple[str, str]] = set()
        scores: dict[tuple[str, str], float] = {}
        for index, left in enumerate(identifiers):
            for right_index in range(index + 1, len(identifiers)):
                right = identifiers[right_index]
                score = cosine_similarity(vectors[index], vectors[right_index])
                if score >= self.config.threshold:
                    verified.add((left, right))
                    scores[(left, right)] = score
        return verified, scores


@dataclass(frozen=True)
class NoveltyVerdict:
    """Whether a candidate text is new relative to what was already seen."""

    is_novel: bool
    nearest: str | None
    similarity: float
    method: str

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-ready view for audit trails."""
        return {
            "is_novel": self.is_novel,
            "nearest": self.nearest,
            "similarity": round(self.similarity, 6),
            "method": self.method,
        }


class NoveltyGuard:
    """Tracks seen texts and rejects candidates that restate one of them.

    The guard is what keeps an evolution loop from rediscovering the same
    hypothesis under a new identifier: :meth:`admit` only records a candidate
    when it is novel, so the seen set never grows with duplicates.
    """

    def __init__(self, config: DedupConfig | None = None) -> None:
        self.config = config or DedupConfig()
        self._seen: dict[str, str] = {}

    @property
    def size(self) -> int:
        """How many distinct texts have been admitted."""
        return len(self._seen)

    def has(self, identifier: str) -> bool:
        """True when an identifier was already observed."""
        return identifier in self._seen

    def observe(self, identifier: str, text: str) -> None:
        """Record a text unconditionally."""
        if identifier in self._seen:
            raise DuplicateIdError(identifier, "novelty entry")
        self._seen[identifier] = text

    def check(self, text: str) -> NoveltyVerdict:
        """Return the closest seen text and whether the candidate is novel."""
        if not self._seen:
            return NoveltyVerdict(True, None, 0.0, self.config.method)
        best_id: str | None = None
        best_score = -1.0
        for identifier, seen in sorted(self._seen.items()):
            score = self._similarity(text, seen)
            if score > best_score:
                best_id, best_score = identifier, score
        return NoveltyVerdict(
            is_novel=best_score < self.config.threshold,
            nearest=best_id,
            similarity=max(0.0, best_score),
            method=self.config.method,
        )

    def admit(self, identifier: str, text: str) -> NoveltyVerdict:
        """Check a candidate and record it when it is novel."""
        verdict = self.check(text)
        if verdict.is_novel:
            self.observe(identifier, text)
        return verdict

    def _similarity(self, left: str, right: str) -> float:
        """Compare two texts with the configured metric."""
        method = self.config.method
        if method == "exact":
            return (
                1.0
                if normalized_signature(left) == normalized_signature(right)
                else 0.0
            )
        if method == "jaccard":
            return jaccard_similarity(
                left,
                right,
                n=self.config.ngram_size,
                words=self.config.uses_words,
                content_only=self.config.content_only,
            )
        if method == "tfidf":
            vectorizer = TfidfVectorizer(ngram_size=self.config.word_ngram_size)
            vectors = vectorizer.fit_transform([left, right])
            return cosine_similarity(vectors[0], vectors[1])
        return minhash_similarity(
            text_signature(
                left,
                self.config.num_perm,
                n=self.config.ngram_size,
                words=self.config.uses_words,
                content_only=self.config.content_only,
                seed=self.config.seed,
            ),
            text_signature(
                right,
                self.config.num_perm,
                n=self.config.ngram_size,
                words=self.config.uses_words,
                content_only=self.config.content_only,
                seed=self.config.seed,
            ),
        )


SHINGLE_UNITS = ("char", "word")


def content_shingles(text: str, n: int, *, words: bool = True) -> frozenset[str]:
    """Return shingles built from content tokens only.

    Function and reporting words are dropped first, so two findings that state
    the same relation with different framing ("we observed that ...", "assays
    show that ...") produce nearly the same shingle set. This is the mode that
    makes paraphrase clusters separable; plain shingles keep every token and are
    the better choice when surface form matters (near-identical text detection).
    """
    tokens = content_tokens(text)
    if words:
        return frozenset(word_ngrams(tokens, n))
    return frozenset(char_ngrams(" ".join(tokens), n))


def shingle_set(
    text: str, n: int, *, words: bool = False, content_only: bool = False
) -> frozenset[str]:
    """Return the shingle set for one text under the given options."""
    if content_only:
        return content_shingles(text, n, words=words)
    return shingles(text, n, words=words)


@dataclass(frozen=True)
class DedupReport:
    """The outcome of one deduplication pass, with the configuration used."""

    clusters: tuple[DuplicateCluster, ...]
    config: DedupConfig
    total: int

    def __post_init__(self) -> None:
        if self.total < 0:
            raise ValidationError("total must be >= 0", total=self.total)
        seen: set[str] = set()
        for cluster in self.clusters:
            overlap = seen & set(cluster.members)
            if overlap:
                raise ValidationError(
                    "a text appears in more than one cluster",
                    members=sorted(overlap),
                )
            seen.update(cluster.members)

    @property
    def duplicated(self) -> int:
        """How many texts belong to some cluster."""
        return sum(cluster.size for cluster in self.clusters)

    @property
    def duplicate_rate(self) -> float:
        """Fraction of examined texts that were grouped as duplicates."""
        return self.duplicated / self.total if self.total else 0.0

    def duplicate_ids(self) -> tuple[str, ...]:
        """Return every identifier that belongs to a cluster, sorted."""
        return tuple(
            sorted(member for cluster in self.clusters for member in cluster.members)
        )

    def cluster_of(self, identifier: str) -> DuplicateCluster | None:
        """Return the cluster containing an identifier, if any."""
        for cluster in self.clusters:
            if cluster.contains(identifier):
                return cluster
        return None

    def metrics_against(self, truth: Sequence[Iterable[str]]) -> ClusterMetrics:
        """Score this pass against planted clusters."""
        return cluster_metrics(self.clusters, truth)

    def signature(self) -> str:
        """Return a digest over the clusters and the configuration."""
        return content_hash(
            {
                "config": self.config.fingerprint(),
                "total": self.total,
                "clusters": [cluster.as_dict() for cluster in self.clusters],
            }
        )

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-ready summary for reports and run metadata."""
        return {
            "total": self.total,
            "clusters": len(self.clusters),
            "duplicated": self.duplicated,
            "duplicate_rate": round(self.duplicate_rate, 4),
            "method": self.config.method,
            "threshold": self.config.threshold,
            "config_fingerprint": self.config.fingerprint(),
            "signature": self.signature(),
        }
