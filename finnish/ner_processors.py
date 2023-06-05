from __future__ import annotations
from typing import NamedTuple

import requests
import stanza


class NamedEntity(NamedTuple):
    text: str
    label: str
    start_char: int
    end_char: int


class TurkuNerProcessor:
    UNWANTED = {
        'NORP',
        'LAW',
        'DATE',
        'CARDINAL',
        'ORDINAL',
        'PERCENT',
        'QUANTITY',
        'LANGUAGE',
        'TIME',
        'MONEY'
    }

    def __repr__(self) -> str:
        return 'turku-ner'

    def process(self, text: str) -> list[tuple[str, str]]:
        request = requests.get('http://127.0.0.1:8080', params={'text': text})
        reqtext = request.text
        ner_results, tokens_indices = self._process_tokens(text, reqtext)
        return ner_results

    def _process_tokens(
        self, text: str, requested_text: str, offset: int = 0
    ) -> tuple[list[tuple[str, str]], list[tuple[str, int]]]:
        ner_results = tuple(r.split('\t') for r in requested_text.split('\n') if r)
        ner_words = tuple(r[0] for r in ner_results)
        tokens_indices = []
        indx = 0
        for token in ner_words:
            indx = text.find(token, indx)
            tokens_indices.append((token, indx + offset))
            indx += len(token)
        return ner_results, tokens_indices


class StanzaNerProcessor:
    stanza_nlp = stanza.Pipeline(
        'fi', processors='tokenize,ner', download_method=stanza.DownloadMethod.REUSE_RESOURCES
    )

    def __repr__(self) -> str:
        return 'stanza-ner'

    def process(self, text) -> list[tuple[str, str]]:
        doc = self.stanza_nlp(text)
        results = []
        for token in doc.iter_tokens():
            results.append((token.text, token.ner))
        return results
