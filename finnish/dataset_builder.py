"""Handles the original .csv files in the orig folder"""

from __future__ import annotations
import csv
from pathlib import Path
from string import punctuation
from typing import Iterator, Optional

punctset = set(punctuation)
punct = punctset - {'('}
punct2 = punctset - {')'}


HERE = Path(__file__).parent
DATASET_FOLDER = HERE / 'dataset'
SOURCE_FOLDER = HERE / 'orig'


def _yield_csv_sentences(
    datafile: Path | str,
    limit: Optional[int] = None,
    label: bool = False,
    offset: int = -1
) -> Iterator[str]:
    if isinstance(datafile, str):
        datafile = Path(datafile)
    with datafile.open() as f:
        reader = csv.DictReader(f, delimiter=';')
        sentence = ''
        current_index = -1
        sentence_index = 0
        for row in reader:
            if not row['ord']:
                continue
            if limit and sentence_index >= limit:
                return
            indx = int(row['ord'])
            word = row['word']
            if label:
                l1 = row['1']
                l2 = row['2']
                if l1 != '0':
                    if l2 != '0' and l2 != l1:
                        ner_label = f"{l1}|{l2}"
                    else:
                        ner_label = l1
                    word += f"[{ner_label}]"
            if indx < current_index:
                current_index = indx
                if sentence_index >= offset:
                    yield sentence.replace('( ', '(') + '\n'
                sentence_index += 1
                sentence = word
            else:
                sentence += word if word in punct else f" {word}"
            current_index = indx
        if sentence:
            if sentence_index >= offset:
                yield sentence.replace('( ', '(')


def _get_csv_labels(datafile=None):
    if isinstance(datafile, str):
        datafiles = [datafile]
    else:
        datafiles = list(Path(__file__).parent.glob('*.csv'))

    labels = set()
    for dfile in datafiles:
        with open(dfile) as f:
            reader = csv.DictReader(f, delimiter=';')
            labels = labels.union({row['1'] for row in reader})
    return list({lab.split('-')[-1] for lab in labels if lab != '0'} - {''})


def _write_sentences(
    datafile: Path | str,
    outfile: Optional[Path | str] = None,
    limit: Optional[int] = None,
    offset: int = -1
) -> None:
    dfile = Path(datafile)

    if outfile:
        name = Path(outfile)
    else:
        name = Path(f"{dfile.stem}-sentences.txt")
    with name.open('w') as out:
        for sentence in _yield_csv_sentences(dfile, limit=limit, offset=offset):
            out.write(sentence)


def build_dataset():
    dset1 = DATASET_FOLDER / 'evaluation_dataset_I.txt'
    dset2 = DATASET_FOLDER / 'evaluation_dataset_II.txt'
    dset3 = DATASET_FOLDER / 'evaluation_dataset_III.txt'
    source1 = SOURCE_FOLDER / 'BS_evaluation_dataset.csv'
    source2 = SOURCE_FOLDER / 'BS_evaluation_dataset_200.csv'
    DATASET_FOLDER.mkdir(exist_ok=True, parents=True)

    _write_sentences(datafile=source2, outfile=dset1, limit=100)
    _write_sentences(datafile=source2, outfile=dset2, offset=100)
    _write_sentences(datafile=source1, outfile=dset3)
