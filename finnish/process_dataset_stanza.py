"""Based on the code for Dutch by José Angel Daza
https://github.com/InTaVia/nlp-pipelines/blob/main/dutch/nl_text_to_json.py"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Callable, Any

import stanza
from tqdm import tqdm

import utils_nlp as unlp


def process_doc(doc: Path, nlp: Callable) -> list[dict[str, Any]]:
    processed_doc = []
    text = doc.read_text()
    for sentence in tqdm(text.split('\n')):
        sentence = sentence.strip()
        if not sentence:
            continue
        nlp_dict = {}
        stanza_dict = unlp.run_stanza(sentence, nlp)
        nlp_dict['input_text'] = sentence
        nlp_dict['token_objs'] = stanza_dict['token_objs']
        nlp_dict['entities'] = stanza_dict['entities']
        response = unlp.nlp_to_dict(nlp_dict)
        processed_doc.append(response)
    return processed_doc


if __name__ == '__main__':
    FOLDER = Path('processed')
    FOLDER.mkdir(exist_ok=True)

    nlp = stanza.Pipeline(
        lang='fi',
        processors='tokenize,lemma,pos,depparse,ner',
        download_method=stanza.DownloadMethod.REUSE_RESOURCES,
    )

    dataset = list(Path('./dataset').glob('*.txt'))

    results = []
    for doc in dataset:
        results.extend(process_doc(doc, nlp))

    with (FOLDER / 'processed_dataset.json').open('w') as f:
        json.dump(results, f, indent=2)
