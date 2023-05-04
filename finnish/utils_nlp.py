"""Based on the code for Dutch by José Angel Daza
https://github.com/InTaVia/nlp-pipelines/blob/main/dutch/utils_nlp.py"""

from __future__ import annotations
from typing import TypeVar, Any, Union, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
from collections import defaultdict

import stanza


StanzaPipeline = TypeVar('StanzaPipeline')


@dataclass
class TokenJSON:
    ID: int
    FORM: str
    LEMMA: str
    UPOS: str
    XPOS: str
    HEAD: int
    DEPREL: str
    DEPS: Optional[str]
    MISC: Optional[dict[str, Any]] = None
    FEATS: Optional[dict[str, str]] = None


def nlp_to_dict(nlp_dict: dict[str, Any]) -> dict[str, Any]:
    sentencized, token_objs = defaultdict(list), []
    tokens = []
    for tok in nlp_dict['token_objs']:
        sentencized[tok['sent_id']].append(tok)
        tokens.append(tok['text'])
    for sent_id, sentence in sentencized.items():
        sent_text = ' '.join(tokens)
        token_objs.append(
            {
                'paragraph': None,
                'sentence': None,
                'text': sent_text,
                'words': [asdict(nlp_token2json_token(tok)) for tok in sentence],
            }
        )

    return {
        'status': '200',
        'data': {
            'text': nlp_dict['input_text'],
            'morpho_syntax': {'model': 'stanza_fi', 'data': token_objs},
            'entities': nlp_dict.get('entities', []),
            'tokenization': {'model': 'stanza_fi', 'data': tokens}
        },
        'service': 'stanza_fi',
        'timestamp': str(datetime.date(datetime.now()))
    }


def nlp_token2json_token(nlp_token: dict[str, Any]):
    return TokenJSON(
        ID=nlp_token['id'],
        FORM=nlp_token['text'],
        LEMMA=nlp_token['lemma'],
        UPOS=nlp_token['upos'],
        XPOS=nlp_token['xpos'],
        HEAD=nlp_token['dep_head'],
        DEPREL=nlp_token['dep_rel'],
        DEPS=None,
        FEATS=process_feats(nlp_token['morph']),
        MISC={
            'SpaceAfter': nlp_token['space_after'],
            'StartChar': nlp_token['start_char'],
            'EndChar': nlp_token['end_char'],
            'NamedEntityLabel': nlp_token['ner_iob'],
        },
    )


def run_stanza(text: Union[str, list[list[str]]], nlp: Callable) -> dict:
    doc = nlp(text)

    stanza_info, stanza_tokens, stanza_sents = [], [], []
    stanza_entities = []

    charstarts2token = {}
    charends2token = {}
    tot_toks = 0

    doc_level_token_id = 0
    for sent_ix, s in enumerate(doc.sentences):
        sent_words = []
        tok_ents = fix_ner_entities(s.tokens)
        #  tok_ents_alt = pick_ner_entities(s.tokens)
        #  if tok_ents != tok_ents_alt:
        #      import ipdb; ipdb.set_trace()

        sentence_tokens = []
        for tok in s.words:
            sentence_tokens.append(tok)
            charstarts2token[tok.start_char] = tot_toks
            charends2token[tok.end_char] = tot_toks + 1
            tot_toks += 1

        stanza_entities += [
            {
                'method': 'stanza_fi',
                'text': ent.text,
                'label': ent.type,
                'start': ent.start_char,
                'end': ent.end_char,
                'start_token': charstarts2token[ent.start_char],
                'end_token': charends2token[ent.end_char],
            }
            for ent in s.ents
        ]

        shifted_sentence = sentence_tokens + ['</END>']
        for ix, (tok, next_tok) in enumerate(
            zip(sentence_tokens, shifted_sentence[1:])
        ):
            sent_words.append(tok.text)
            obj = {
                'id': doc_level_token_id,
                'text': tok.text,
                'lemma': tok.lemma,
                'upos': tok.upos,
                'xpos': tok.xpos,
                'morph': tok.feats,
                'dep_head': tok.head,
                'dep_rel': tok.deprel,
                'ner_iob': tok_ents[ix],
                'start_char': tok.start_char,
                'end_char': tok.end_char,
                'space_after': (
                    False if next_tok != '</END>'
                        and tok.end_char == next_tok.start_char
                    else True
                ),
                'sent_id': sent_ix,
                'is_sent_start': True if ix == 0 else False,
                'is_sent_end': False,
            }
            stanza_info.append(obj)
            stanza_tokens.append(tok.text)
            doc_level_token_id += 1
        # The last char of a sentence needs some manual inspection
        # to properly set the space_after and is_sent_end!
        if len(stanza_info) > 0:
            stanza_info[-1]['is_sent_end'] = True
            if tok.end_char < len(text):
                lookahead_char = text[tok.end_char]
                if lookahead_char != ' ':
                    stanza_info[-1]['space_after'] = False
            else:
                stanza_info[-1]['space_after'] = False
        stanza_sents.append(' '.join(sent_words))
    return {
        'stanza_doc': doc,
        'sentences': stanza_sents,
        'tokens': stanza_tokens,
        'token_objs': stanza_info,
        'entities': stanza_entities,
    }


def process_feats(feats: Optional[str]) -> Optional[dict[str, str]]:
    if not feats:
        return None
    results = {}
    for feat in feats.split('|'):
        key, value = feat.split('=')
        results[key] = value
    return results

def fix_ner_entities(tokens: list[stanza.models.common.doc.Tokens]):
    """Transforms the ner labelling model to BIO"""
    ner_labels = []
    prev_label = 'O'
    for tok in tokens:
        ner_label = tok.ner
        # handling cases where a token is divided into multiple words
        # e.g. jottei -> jotta + ei
        tok_len = len(tok.words)
        if tok_len > 1:
            ner_labels.extend([ner_label] * tok_len)
        if (
            ner_label.startswith('S')
            or (prev_label == 'O' and ner_label.startswith(('I')))
        ):
            ner_label = f"B{ner_label[1:]}"
        elif ner_label.startswith('E'):
            ner_label = f"I{ner_label[1:]}"
        ner_labels.append(ner_label)
        prev_label = ner_label
    return ner_labels
