import re
from typing import TypeVar, Dict, Any, List, Tuple, Union
from dataclasses import dataclass, asdict
from collections import defaultdict
from bs4 import BeautifulSoup


InfoExtractor = TypeVar('InfoExtractor')
ClasslaPipeline = TypeVar('ClasslaPipeline')

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


## Functions to NLP Process

def preprocess_and_clean_text(text: str) -> str:
    clean_text = re.sub(r'[\r\n]+', " ", text)
    clean_text = re.sub(r'"', ' " ', clean_text)
    clean_text = re.sub(r'[\s]+', " ", clean_text)
    return clean_text


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
            'entities': nlp_dict.get('entities', []),
            'time_expressions': nlp_dict.get('time_expressions', []),
            'semantic_roles': nlp_dict.get('semantic_roles', [])
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


def run_classla(text: Union[str, List[List[str]]], nlp: ClasslaPipeline) -> Dict:

    doc = nlp(text)

    classla_info, classla_tokens, classla_sents = [], [], []
    
    charstarts2token = {}
    charends2token = {}
    tot_toks = 0

    tok_ents = []
    doc_char_index = 0
    spaces_after = []
    document_tokens = []
    for sent_ix, s in enumerate(doc.sentences):
        sent_words = [tok.text for tok in s.tokens]
        tok_ents += [tok.ner for tok in s.tokens]
        classla_sents.append(" ".join(sent_words))
        sentence_tokens = []
        for tok in s.words:
            sentence_tokens.append(tok)
            charstarts2token[doc_char_index] = tot_toks 
            charends2token[doc_char_index + len(tok.text)] = tot_toks
            tot_toks += 1
            if tok.misc:
                doc_char_index += len(tok.text) # "SpaceAfter=No"
                spaces_after.append(False)
            else:
                doc_char_index += len(tok.text) + 1
                spaces_after.append(True)
        document_tokens.append(sentence_tokens)

    token2charstart = {v:k for k,v in charstarts2token.items()}
    token2charends = {v:k for k,v in charends2token.items()}

    classla_entities = []

    doc_level_token_id = 0
    for sent_ix, sent_toks in enumerate(document_tokens):
        for ix, tok in enumerate(sent_toks):
            obj = {'id': doc_level_token_id,
                    'text': tok.text, 
                    'lemma': tok.lemma, 
                    'upos': tok.upos, 
                    'xpos': tok.xpos,
                    'morph': tok.feats,
                    'dep_head': tok.head,
                    'dep_rel': tok.deprel,
                    'ner_iob': tok_ents[doc_level_token_id],
                    'start_char': token2charstart[doc_level_token_id], 
                    'end_char': token2charends[doc_level_token_id],
                    'space_after': spaces_after[doc_level_token_id],
                    'sent_id': sent_ix
                    }
            classla_info.append(obj)
            classla_tokens.append(tok.text)
            doc_level_token_id += 1
    
    classla_entities = bio2entities(classla_info)
        
    return {'classla_doc': doc, 'sentences':classla_sents, 'tokens': classla_tokens, 'token_objs': classla_info, 'entities': classla_entities}

def bio2entities(token_objs: Dict[str, Any]) -> List:
    sentence_entities = []
    ent_tokens, ent_indices, ent_label = [], [], ""
    for ix, token in enumerate(token_objs):
        bio_tag = token['ner_iob']
        if bio_tag.startswith("B-"):
            if len(ent_label) > 0:
                sentence_entities.append({'ID': None, 'surfaceForm': " ".join([t['text'] for t in ent_tokens]), 'category': ent_label.upper(), 
                                        'locationStart': ent_tokens[0]['start_char'], 'locationEnd': ent_tokens[-1]['end_char'], 
                                        'tokenStart': ent_indices[0], 'tokenEnd': ent_indices[-1]+1, 'method': 'classla'})
                ent_indices = []
                ent_tokens = []
            ent_label = bio_tag[2:]
            ent_tokens.append(token)
            ent_indices.append(ix)
        elif bio_tag.startswith("I-"):
            ent_tokens.append(token)
            ent_indices.append(ix)
        elif bio_tag == "O" and len(ent_label) > 0:
            sentence_entities.append({'ID': None, 'surfaceForm': " ".join([t['text'] for t in ent_tokens]), 'category': ent_label.upper(), 
                                        'locationStart': ent_tokens[0]['start_char'], 'locationEnd': ent_tokens[-1]['end_char'], 
                                        'tokenStart': ent_indices[0], 'tokenEnd': ent_indices[-1]+1, 'method': 'classla'})
            ent_label = ""
            ent_indices = []
            ent_tokens = []
    
    return sentence_entities