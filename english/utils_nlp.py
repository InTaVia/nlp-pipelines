import re
from typing import TypeVar, Dict, Any, List
from dataclasses import dataclass, asdict
from collections import defaultdict
Converter = TypeVar('Converter')
NafParser = TypeVar('NafParser')
SpacyLanguage = TypeVar('SpacyLanguage')
SpacyDoc = TypeVar('SpacyDoc')


@dataclass 
class TokenJSON:
    ID: int
    FORM: str
    LEMMA: str
    UPOS: str
    XPOS: str
    HEAD: int
    DEPREL: str
    DEPS: str
    MISC: List[str] = None
    FEATS: Dict[str, str] = None



def preprocess_and_clean_text(text: str) -> str:
    clean_text = re.sub(r'[\r\n]+', " ", text)
    clean_text = re.sub(r'"', ' " ', clean_text)
    clean_text = re.sub(r'[\s]+', " ", clean_text)
    return clean_text


def run_spacy(text: str, nlp: SpacyLanguage, nlp_processor: str = 'spacy') -> Dict:
    doc = nlp(text)
    spacy_info, spacy_tokens, spacy_sents = [], [], []
    spacy_ents = []
    for sent_ix, sent in enumerate(doc.sents):
        spacy_sents.append(" ".join([t.text for t in sent]))
        shifted_sentence = list(sent) + ['</END>']
        for tok_ix, (tok, next_tok) in enumerate(zip(sent, shifted_sentence[1:])):
            spacy_tokens.append(tok.text)
            obj = {'id': tok.i, 
                    'text': tok.text, 
                    'lemma': tok.lemma_, 
                    'upos': tok.pos_, 
                    'xpos': tok.tag_,
                    'dep_head': tok.head.i,
                    'dep_rel': tok.dep_,
                    'ner_iob': tok.ent_iob_,
                    'ner_type': tok.ent_type_,
                    'morph': tok.morph.to_dict(),
                    'start_char': tok.idx, 
                    'end_char': tok.idx + len(tok.text),
                    'space_after': False if tok_ix < len(sent)-1 and tok.idx + len(tok.text) == next_tok.idx else True,
                    'like_url': tok.like_url,
                    'like_email': tok.like_email,
                    'is_oov': tok.is_oov,
                    'is_alpha': tok.is_alpha,
                    'is_punct': tok.is_punct,
                    'sent_id': sent_ix,
                    'is_sent_start': tok.is_sent_start,
                    'is_sent_end': tok.is_sent_end
                    }
            spacy_info.append(obj)
        # The last char of a sentence needs some manual inspecion to properly set the space_after and is_sent_end!
        if len(spacy_info) > 0:
            if obj['end_char'] < len(text):
                lookahead_char = text[obj['end_char']]
                if lookahead_char != " ":
                    spacy_info[-1]['space_after'] = False
            else:
                spacy_info[-1]['space_after'] = False
    if doc.ents:
        for ent in doc.ents:
            spacy_ents.append({'ID': None, 'surfaceForm': ent.text, 'category': ent.label_.upper(), 'locationStart': doc[ent.start].idx, 'locationEnd': doc[ent.end].idx, 
                                'tokenStart': ent.start, 'tokenEnd': ent.end, 'method': nlp_processor})

    return {'spacy_doc': doc,'sentences':spacy_sents, 'tokens': spacy_tokens, 'token_objs': spacy_info, 'entities': spacy_ents}


def nlp_to_dict(nlp_dict: Dict[str, Any]) -> Dict[str, Any]:
    sentencized, token_objs = defaultdict(list), []
    for tok in nlp_dict['token_objs']:
        sentencized[tok['sent_id']].append(tok)
    for sent_id, sentence in sentencized.items():
        sent_text = " ".join([tok['text'] for tok in sentence])
        token_objs.append({
            "paragraph": None,
            "sentence": sent_id,
            "text": sent_text,
            "words": [asdict(nlp_token2json_token(tok)) for tok in sentence]
        })


    return {
        'status': '200',
        'data': {
            'text': nlp_dict['input_text'],
            'morphology': token_objs,
            'entities': nlp_dict['entities'],
            'time_exps': [],
            'events': []
        }
    }


def nlp_token2json_token(nlp_token: Dict[str, Any]):
    return TokenJSON(
        ID=nlp_token['id'],
        FORM=nlp_token['text'],
        LEMMA=nlp_token['lemma'],
        UPOS=nlp_token['upos'],
        XPOS=nlp_token['xpos'],
        HEAD=nlp_token['dep_head'],
        DEPREL=nlp_token['dep_rel'],
        DEPS=None,
        FEATS=nlp_token['morph'],
        MISC={
            'SpaceAfter': nlp_token['space_after'],
            'StartChar': nlp_token['start_char'],
            'EndChar': nlp_token['end_char']
        }
    )




def create_naf_object(text: str, naf_name: str, naf_converter: Converter) -> NafParser:
    """Reads in a Text, uses SpaCy as an NLP Pipeline and returns the annotations on NAF Format

    Args:
        text (str): The text to be parsed and strcuctured. Annotations added: 

    Returns:
        NafParser: The NAF object containing the SpaCy English annotations
    """
    naf_name = naf_name.lower().replace(" ", "_") + ".naf"
    naf = naf_converter.process_text(text, naf_name, out_path=None)
    return naf
