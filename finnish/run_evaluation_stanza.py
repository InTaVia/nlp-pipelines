from __future__ import annotations
import json
from typing import Any, Iterator

import numpy as np
import pandas as pd
from nltk.metrics.distance import edit_distance

import conll_handler


IGNORE_BIO = False

EQUIVALENCIES = {'EVENT': 'EVT', 'PRO': 'WORK'}


def find_matches(
    annotations: dict[str, list[tuple[str, str]]], processed: list[dict[str, Any]]
):
    spaceless_annotations = {t.replace(' ', ''): t for t in annotations}
    unmatched_text = []
    matches = {}
    for sentence in processed:
        text = sentence['data']['text']  # ['morpho_syntax']['data'][0]['text']
        spaceless_processed = text.replace(' ', '')

        if match := spaceless_annotations.get(spaceless_processed):
            matches[text] = match
        else:
            unmatched_text.append(text)
    if not unmatched_text:
        return matches
    unmatched_annotations = list(set(annotations) - set(matches.values()))
    dist_matches = _match_by_distance(unmatched_text, unmatched_annotations)
    matches.update(dist_matches)
    return matches


def _match_by_distance(unmatched_text: list[str], unmatched_annotations: list[str]):
    assert len(unmatched_text) == len(unmatched_annotations)
    distances = np.zeros((len(unmatched_text), len(unmatched_text)))
    for ntext, utext in enumerate(unmatched_text):
        for nann, uann in enumerate(unmatched_annotations):
            dist = edit_distance(utext, uann) / max(len(utext), len(uann))
            distances[ntext, nann] = dist
    mins = distances.argmin(1)
    if not all(np.sort(mins) == np.arange(len(unmatched_text))):
        raise Exception('Wrong distance matching')
    matches = {}
    for nt, na in np.ndenumerate(mins):
        text = unmatched_text[nt[0]]
        annotation = unmatched_annotations[na]
        matches[text] = annotation
    return matches


def evaluate_sentence(
    processed_sentence, annotation, ignore_bio=False
) -> Iterator[tuple[str, str]]:
    ner_labels = []
    data = processed_sentence['morpho_syntax']['data']
    for data_num in data:
        for pw in data_num['words']:
            ner_label = pw['MISC']['NamedEntityLabel']
            if ner_label.startswith(('S', 'E')):
                ner_label = f"I{ner_label[1:]}"
            if ignore_bio:
                ner_labels.append(_replace_equivalencies(_remove_bio(ner_label)))
            else:
                ner_labels.append(_replace_equivalencies(ner_label))
    if ignore_bio:
        ann_labels = [_remove_bio(i[1]) for i in annotation]
    else:
        ann_labels = [i[1] for i in annotation]
    if len(ann_labels) != len(ner_labels):
        raise ValueError('The labels do not match')
    for alabel, nlabel in zip(ann_labels, ner_labels):
        relationship = _process_relationship(alabel, nlabel)
        yield alabel, relationship


def evaluate(processed, annotations, matches, ignore_bio=False):
    if ignore_bio:
        alabels = sorted(
            {_remove_bio(tok[1]) for sent in annotations.values() for tok in sent}
        )
    else:
        alabels = sorted({tok[1] for sent in annotations.values() for tok in sent})
    eval_categories = ['TP', 'FP', 'TN', 'FN']
    evaluations_df = pd.DataFrame(0, index=alabels, columns=eval_categories)
    falses = 0
    for item in processed:
        proc_sentence = item['data']
        annotation = annotations[matches[proc_sentence['text']]]
        try:
            for label, relationship in evaluate_sentence(
                proc_sentence, annotation, ignore_bio
            ):
                evaluations_df.loc[label, relationship] += 1
        except ValueError:
            falses += 1
    precision = evaluations_df.eval('TP / (TP + FP)')
    recall = evaluations_df.eval('TP / (TP + FN)')
    f1 = 2 * (precision * recall) / (precision + recall)
    totals = evaluations_df.sum(1)
    print(f"{falses=}")
    return pd.concat(
        [
            precision.rename('Precision'),
            recall.rename('Recall'),
            f1.rename('F1'),
            totals.rename('Total'),
        ],
        axis=1
    )


def _process_relationship(annotation_label, predicted_label) -> str:
    if annotation_label == 'O':
        if predicted_label != 'O':                  # False positive
            return 'FP'
        else:                                       # True negative
            return 'TN'
    else:
        if predicted_label == 'O':                  # False negative
            return 'FN'
        elif predicted_label == annotation_label:   # True positive
            return 'TP'
        else:                                       # False positive
            return 'FP'


def _remove_bio(label: str) -> str:
    if label == 'O':
        return label
    return label.split('-')[1]


def _replace_equivalencies(txt: str, equiv_dict: dict[str, str] = EQUIVALENCIES) -> str:
    for orig, replacement in equiv_dict.items():
        if orig in txt:
            return txt.replace(orig, replacement)
    return txt


if __name__ == '__main__':
    annots = conll_handler.read_conll('annotations/conll/annotator1-I_II_III.conll')
    with open('processed/processed_dataset.json') as f:
        processed = json.load(f)

    matches = find_matches(annots, processed)
    evaluation_df = evaluate(processed, annots, matches, ignore_bio=IGNORE_BIO)
    edf = evaluation_df.drop('O').fillna(0)

    edf = edf.sort_values('F1', ascending=False)
    averages = np.average(
        edf.iloc[:, :3], weights=edf['Total'], axis=0
    )
    avgs = np.hstack((averages, np.array([np.nan])))
    edf.loc['Weighted avg'] = avgs
    print(edf.round(3))
