import re
from typing import TypeVar, Dict, Any, List, Tuple, Union
from dataclasses import dataclass, asdict
from collections import defaultdict
from python_heideltime import Heideltime
from bs4 import BeautifulSoup

from flair.data import Sentence
from flair.models import SequenceTagger
from flair.tokenization import SegtokSentenceSplitter

Converter = TypeVar('Converter')
NafParser = TypeVar('NafParser')
InfoExtractor = TypeVar('InfoExtractor')
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
    DEPS: str
    MISC: List[str] = None
    FEATS: Dict[str, str] = None


@dataclass
class SRL_Argument:
    predicate: Tuple[int, str] # (token_index, surface_form) in the text
    text: str # argument text
    label: str # argument label
    start: int # token index where the argument starts
    end: int # token index where the argument ends


@dataclass
class SRL_Output:
    tokens: List[str]
    predicates: List[Tuple[int, str]] # (token_index, surface_form) in the text
    arg_labels: List[List[str]] # each internal list has Bio Labels corresponding to the predicates by position in the list
    pred_arg_struct: Dict[int, SRL_Argument]


## Functions to NLP Process

def preprocess_and_clean_text(text: str) -> str:
    clean_text = re.sub(r'[\r\n]+', " ", text)
    clean_text = re.sub(r'"', ' " ', clean_text)
    clean_text = re.sub(r'[\s]+', " ", clean_text)
    return clean_text


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


def run_stanza(text: Union[str, List[List[str]]], nlp: StanzaPipeline) -> Dict:

    doc = nlp(text)

    stanza_info, stanza_tokens, stanza_sents = [], [], []
    stanza_entities = []
    
    charstarts2token = {}
    charends2token = {}
    tot_toks = 0

    doc_level_token_id = 0
    for sent_ix, s in enumerate(doc.sentences):
        sent_words = []
        tok_ents = [tok.ner for tok in s.tokens]

        sentence_tokens = []
        for tok in s.words:
            sentence_tokens.append(tok)
            charstarts2token[tok.start_char] = tot_toks 
            charends2token[tok.end_char] = tot_toks + 1
            tot_toks += 1

        stanza_entities += [{'method': 'stanza_nl', 'text': ent.text, 'label': ent.type, 'start': ent.start_char, 'end': ent.end_char, 'start_token': charstarts2token[ent.start_char], 'end_token': charends2token[ent.end_char]} for ent in s.ents]
        
        shifted_sentence = sentence_tokens + ['</END>']
        for ix, (tok, next_tok) in enumerate(zip(sentence_tokens, shifted_sentence[1:])):
            sent_words.append(tok.text)
            try:
                srl_info = (tok.srl, tok.frame)
            except:
                srl_info = (None, None)
            obj = {'id': doc_level_token_id,
                    'text': tok.text, 
                    'lemma': tok.lemma, 
                    'upos': tok.upos, 
                    'xpos': tok.xpos,
                    'morph': tok.feats,
                    'dep_head': tok.head,
                    'dep_rel': tok.deprel,
                    'ner_iob': tok_ents[ix],
                    'srl': srl_info[0],
                    'frame': srl_info[1],
                    'start_char': tok.start_char, 
                    'end_char': tok.end_char,
                    'space_after': False if next_tok != '</END>' and tok.end_char == next_tok.start_char else True,
                    'sent_id': sent_ix,
                    'is_sent_start': True if ix == 0 else False,
                    'is_sent_end': False
                    }
            stanza_info.append(obj)
            stanza_tokens.append(tok.text)
            doc_level_token_id += 1
        # The last char of a sentence needs some manual inspecion to properly set the space_after and is_sent_end!
        if len(stanza_info) > 0:
            stanza_info[-1]['is_sent_end'] = True
            if tok.end_char < len(text):
                lookahead_char = text[tok.end_char]
                if lookahead_char != " ":
                    stanza_info[-1]['space_after'] = False
            else:
                stanza_info[-1]['space_after'] = False
        stanza_sents.append(" ".join(sent_words))
    return {'stanza_doc': doc, 'sentences':stanza_sents, 'tokens': stanza_tokens, 'token_objs': stanza_info, 'entities': stanza_entities}


def run_flair(text: str, tagger: SequenceTagger, splitter: SegtokSentenceSplitter) -> Dict:
    if splitter:
        sentences = splitter.split(text)
        tagger.predict(sentences)
        texts, ner = [], []
        for sentence in sentences:
            tagged_ents = []
            for entity in sentence.get_spans('ner'):
                tagged_ents.append({"text": entity.text, "start": entity.start_position, "end": entity.end_position, "entity": entity.get_label("ner").value, "score": entity.get_label("ner").score})
            ner.append(tagged_ents)
            texts.append(sentence.to_original_text())
        return {'tagged_ner':  ner, 'sentences': texts}
    else:
        sentence = Sentence(text)
        tagger.predict(sentence)
        tagged_ents = []
        for entity in sentence.get_spans('ner'):
            tagged_ents.append({"text": entity.text, "start": entity.start_position, "end": entity.end_position, "entity": entity.get_label("ner").value, "score": entity.get_label("ner").score})
        return {'tagged_ner': tagged_ents, 'sentences': [sentence.to_tokenized_string()]}


## Functions to Add JSON Layers (via Script or API)

def add_json_heideltime(text: str, heideltime_parser: Heideltime) -> List[Dict[str, Any]]:
    # Get Time Expressions
    xml_timex3 = heideltime_parser.parse(text)
    # Map the TIMEX Nodes into the Raw String Character Offsets
    soup  = BeautifulSoup(xml_timex3, 'xml')
    root = soup.find('TimeML')
    span_end = 0
    timex_all = []
    try:
        for timex in root.find_all('TIMEX3'):
            span_begin = span_end + root.text[span_end:].index(timex.text) - 1
            span_end = span_begin + len(timex.text)
            timex_dict = {'ID': timex.get('tid'), 'category': timex.get('type'), 'value': timex.get('value'), 'surfaceForm': timex.text, 'locationStart': span_begin, 'locationEnd': span_end, 'method': 'HeidelTime'}
            timex_all.append(timex_dict)
        return timex_all
    except:
        return []


def add_json_flair_ner(flair_output: Dict[str, Any]) -> List[Dict[str, Any]]:
    ner_all = []
    print(flair_output)
    doc_offset = 0
    for i, sent_objs in enumerate(flair_output['tagged_ner']):
        for ner_obj in sent_objs:
            ner_all.append({'ID': i, 
                    'category': ner_obj['entity'], 
                    'surfaceForm': ner_obj['text'], 
                    'locationStart': doc_offset + ner_obj['start'], 
                    'locationEnd': doc_offset + ner_obj['end'], 
                    'method': 'flair_ner-dutch-large'
                    })
        doc_offset += len(flair_output['sentences'][i])
    return ner_all