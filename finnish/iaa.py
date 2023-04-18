from __future__ import annotations
from pathlib import Path
from typing import Iterator, Sequence

from sklearn.metrics import cohen_kappa_score

import conll_handler


def _align_annotations(
    reference: str | Path,
    annotations: Sequence[str | Path],
    disregard_nonentities: bool = False
) -> Iterator[tuple[str, str]]:
    an1 = conll_handler.read_conll(reference)
    an2 = {}
    for annotation in annotations:
        an2.update(conll_handler.read_conll(annotation))
    for sentence, an2_values in an2.items():
        an1_values = an1[sentence]
        for (t1w, t1l), (t2w, t2l) in zip(an1_values, an2_values):
            assert t1w == t2w
            if disregard_nonentities and t1l == t2l == 'O':
                continue
            yield t1l, t2l


def calculate_basic_iaa(
    reference: str | Path,
    *annotations: str | Path,
    disregard_nonentities: bool = False
) -> float:
    labeldiff = 0
    total = 0
    annotation_list = list(annotations)
    for t1l, t2l in _align_annotations(
        reference, annotation_list, disregard_nonentities
    ):
        if t1l != t2l:
            labeldiff += 1
        total += 1
    return 1 - (labeldiff / total)


def calculate_kappa_iaa(
    reference: str | Path,
    *annotations: str | Path,
    disregard_nonentities: bool = True
) -> float:
    annotation_list = list(annotations)
    label_indices: dict[str, int] = {}
    an1_labels = []
    an2_labels = []

    for t1l, t2l in _align_annotations(
        reference, annotation_list, disregard_nonentities
    ):
        an1_labels.append(label_indices.setdefault(t1l, len(label_indices)))
        an2_labels.append(label_indices.setdefault(t2l, len(label_indices)))

    return cohen_kappa_score(an1_labels, an2_labels)
