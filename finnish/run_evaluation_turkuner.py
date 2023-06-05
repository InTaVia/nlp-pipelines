from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
import spacy_alignments as salign
from tqdm import tqdm

import conll_handler
import ner_processors as nerproc


IGNORE_BIO = False
# For some reason stanza+ignore_bio isn't working with this method
CHOSEN_PROCESSOR = 'turku'   # 'turku' or 'stanza'

FOLDER = Path('processed')

PROCESSORS_DICT = {
    'turku': {
        'filepath': FOLDER / 'predicted_entities_turkuner.json',
        'proc': nerproc.TurkuNerProcessor,
        'equivalencies': {
            'PERSON': 'PER',
            'EVENT': 'EVT',
            'FAC': 'LOC',
            'GPE': 'LOC',
            'PRODUCT': 'MISC',
            'WORK_OF_ART': 'WORK',
        }
    },
    'stanza': {
        'filepath': FOLDER / 'predicted_entities_stanza.json',
        'proc': nerproc.StanzaNerProcessor,
        'equivalencies': {'EVENT': 'EVT', 'PRO': 'WORK', 'PERSON': 'PER'},
    }
}


def process_annotations(annotations, processor):
    proc = processor['proc']()
    procfile = processor['filepath']
    if procfile.is_file():
        with procfile.open() as f:
            preds = json.load(f)
            return preds
    results = {}
    for sentence, ents in tqdm(annots.items()):
        entities = proc.process(sentence)
        results[sentence] = entities
    with procfile.open('w') as f:
        json.dump(results, f)
    return results


def evaluate(annotations, predictions, processor, ignore_bio=False):
    if ignore_bio:
        alabels = sorted(
            {_remove_bio(tok[1]) for sent in annotations.values() for tok in sent}
        )
    else:
        alabels = sorted({tok[1] for sent in annotations.values() for tok in sent})
    eval_categories = ['TP', 'FP', 'TN', 'FN']
    evaluations_df = pd.DataFrame(0, index=alabels, columns=eval_categories)
    falses = 0
    splits = 0

    for sentence, ents in annots.items():
        turku_ents = predictions[sentence]
        if len(ents) != len(turku_ents):
            ents = _align_annotations(ents, turku_ents)
            if ents is None:
                falses += 1
                continue
            splits += 1
        for (aent, alabel), (tent, tlabel) in zip(ents, turku_ents):
            if IGNORE_BIO:
                alabel = _remove_bio(alabel)
                tlabel = _remove_bio(tlabel)
            tlabel = _replace_equivalencies(tlabel, processor['equivalencies'])
            relationship = _process_relationship(alabel, tlabel)
            evaluations_df.loc[alabel, relationship] += 1
    precision = evaluations_df.eval('TP / (TP + FP)')
    recall = evaluations_df.eval('TP / (TP + FN)')
    f1 = 2 * (precision * recall) / (precision + recall)
    totals = evaluations_df.sum(1)
    print(f"{falses=}")
    print(f"{splits=}")
    return pd.concat(
        [
            precision.rename('Precision'),
            recall.rename('Recall'),
            f1.rename('F1'),
            totals.rename('Total'),
        ],
        axis=1
    )


def _align_annotations(annot_entities, pred_entities):
    atokens = [a[0] for a in annot_entities]
    ptokens = [p[0] for p in pred_entities]
    a2p, p2a = salign.get_alignments(atokens, ptokens)
    if len(p2a) < len(a2p):
        return None
    new_annot_ents = []
    for alig, (form, label) in zip(a2p, annot_entities):
        if len(alig) == 1:
            new_annot_ents.append((form, label))
            continue
        words = [ptokens[i] for i in alig]
        new_labels = _multiply_label(label, len(alig))
        for word, nlab in zip(words, new_labels):
            new_annot_ents.append((word, nlab))
    assert len(new_annot_ents) == len(pred_entities)
    return new_annot_ents


def _multiply_label(label, n):
    if label.startswith(('O', 'I')):
        return [label] * n
    bioless_part = label.split('-')[1]
    return [label] + [f"I-{bioless_part}"] * (n - 1)


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


def _replace_equivalencies(txt: str, equiv_dict: dict[str, str]) -> str:
    for orig, replacement in equiv_dict.items():
        if orig in txt:
            return txt.replace(orig, replacement)
    return txt


if __name__ == '__main__':
    annots = conll_handler.read_conll('annotations/conll/annotator1-I_II_III.conll')

    processor = PROCESSORS_DICT[CHOSEN_PROCESSOR]
    preds = process_annotations(annots, processor)
    evaluation_df = evaluate(annots, preds, processor, ignore_bio=IGNORE_BIO)
    edf = evaluation_df.drop('O').fillna(0)

    edf = edf.sort_values('F1', ascending=False)
    averages = np.average(
        edf.iloc[:, :3], weights=edf['Total'], axis=0
    )
    avgs = np.hstack((averages, np.array([np.nan])))
    edf.loc['Weighted avg'] = avgs
    print(edf.round(3))
