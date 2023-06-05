import re
from typing import TypeVar, Dict, Any, List, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict

from allennlp.predictors import Predictor

from python_heideltime import Heideltime
from bs4 import BeautifulSoup

Converter = TypeVar('Converter')
NafParser = TypeVar('NafParser')
SpacyLanguage = TypeVar('SpacyLanguage')
SpacyDoc = TypeVar('SpacyDoc')
InfoExtractor = TypeVar('InfoExtractor')

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
            'semantic_roles': nlp_dict.get('semantic_roles', []),
            'coreference': nlp_dict.get('coreference', [])
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


def run_spacy(text: str, nlp: SpacyLanguage, nlp_processor: str = 'spacy') -> Dict:
    doc = nlp(text)
    spacy_info, spacy_tokens, spacy_sents = [], [], []
    original_sents = []
    spacy_ents = []
    for sent_ix, sent in enumerate(doc.sents):
        spacy_sents.append(" ".join([t.text for t in sent]))
        original_sents.append(sent)
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
            spacy_ents.append({'ID': None, 'surfaceForm': ent.text, 'category': ent.label_.upper(), 'locationStart': doc[ent.start].idx, 'locationEnd': doc[ent.start].idx + len(ent.text), 
                                'tokenStart': ent.start, 'tokenEnd': ent.end, 'method': nlp_processor})

    return {'spacy_doc': doc, 'sentences':spacy_sents, 'sentences_untokenized': original_sents,'tokens': spacy_tokens, 'token_objs': spacy_info, 'entities': spacy_ents}


def allennlp_srl(text: str, srl_predictor: Predictor) -> SRL_Output:
    output = srl_predictor.predict(text)

    simplified_output = SRL_Output(output['words'], predicates=[], arg_labels=[], pred_arg_struct={})

    for verb_obj in output['verbs']:
        # print(f"\tVERB: {verb_obj['verb']} | ARGS: {verb_obj['tags']}")
        predicate_index, predicate_arguments = 0, []
        arg_tokens, arg_indices, arg_label = [], [], ""
        for ix, bio_tag in enumerate(verb_obj['tags']):
            if bio_tag == "B-V":
                predicate_index = ix
                if len(arg_label) > 0:
                    predicate_arguments.append(SRL_Argument((predicate_index, verb_obj['verb']), " ".join(arg_tokens), arg_label, start=arg_indices[0], end=arg_indices[-1]))
                    arg_label = ""
                    arg_indices = []
                    arg_tokens = []
            elif bio_tag.startswith("B-"):
                if len(arg_label) > 0:
                    predicate_arguments.append(SRL_Argument((predicate_index, verb_obj['verb']), " ".join(arg_tokens), arg_label, start=arg_indices[0], end=arg_indices[-1]))
                    arg_indices = []
                    arg_tokens = []
                arg_label = bio_tag[2:]
                arg_tokens.append(output['words'][ix])
                arg_indices.append(ix)
            elif bio_tag.startswith("I-"):
                arg_tokens.append(output['words'][ix])
                arg_indices.append(ix)
            elif bio_tag == "O" and len(arg_label) > 0:
                predicate_arguments.append(SRL_Argument((predicate_index, verb_obj['verb']), " ".join(arg_tokens), arg_label, start=arg_indices[0], end=arg_indices[-1]))
                arg_label = ""
                arg_indices = []
                arg_tokens = []

        simplified_output.predicates.append((predicate_index, verb_obj['verb']))
        simplified_output.arg_labels.append(verb_obj['tags'])
        simplified_output.pred_arg_struct[predicate_index] = predicate_arguments
        
    return simplified_output


def allennlp_ner(text: str, ner_predictor: Predictor, text_init_offset: int = 0) -> Tuple[List, List]:
    output = ner_predictor.predict(text)
    tokenized_sentence = output['words']
    sentence_entities = []
    ent_tokens, ent_indices, ent_label = [], [], ""
    for ix, bio_tag in enumerate(output["tags"]):
        if bio_tag.startswith("U-"):
            sentence_entities.append({'ID': None, 'surfaceForm': output["words"][ix], 'category': bio_tag[2:].upper(), 'locationStart': None, 'locationEnd': None, 
                                'tokenStart': text_init_offset + ix, 'tokenEnd': text_init_offset + ix + 1, 'method': 'allennlp_2.9.0'})
            ent_label = ""
            ent_indices = []
            ent_tokens = []
        elif bio_tag.startswith("B-"):
            if len(ent_label) > 0:
                sentence_entities.append({'ID': None, 'surfaceForm': " ".join(ent_tokens), 'category': ent_label.upper(), 'locationStart': None, 'locationEnd': None, 
                                'tokenStart': text_init_offset + ent_indices[0], 'tokenEnd': text_init_offset + ent_indices[-1]+1, 'method': 'allennlp_2.9.0'})
                ent_indices = []
                ent_tokens = []
            ent_label = bio_tag[2:]
            ent_tokens.append(output['words'][ix])
            ent_indices.append(ix)
        elif bio_tag.startswith("I-") or bio_tag.startswith("L-"):
            ent_tokens.append(output['words'][ix])
            ent_indices.append(ix)
        elif bio_tag == "O" and len(ent_label) > 0:
            sentence_entities.append({'ID': None, 'surfaceForm': " ".join(ent_tokens), 'category': ent_label.upper(), 'locationStart': None, 'locationEnd': None, 
                                'tokenStart': text_init_offset + ent_indices[0], 'tokenEnd': text_init_offset + ent_indices[-1]+1, 'method': 'allennlp_2.9.0'})
            ent_label = ""
            ent_indices = []
            ent_tokens = []
    
    return tokenized_sentence, sentence_entities


def allennlp_coref(text: str, coref_predictor: Predictor) -> Tuple[List[str], Dict[str, Any]]:
    allennlp_output = coref_predictor.predict(text)
    tokens = allennlp_output["document"]
    clusters = defaultdict(list)
    for cluster_id, cluster_elems in enumerate(allennlp_output["clusters"]):
        for j, (start, end) in enumerate(cluster_elems):
            labeled_span = " ".join(tokens[start:end+1])
            clusters[cluster_id].append({'ID': f"{cluster_id}_{j}",'surfaceForm': labeled_span, 'locationStart': None, 'locationEnd': None, 
                                            'tokenStart': start, 'tokenEnd': end+1, 'method': 'allennlp_2.9.0'}) 
    return tokens, clusters


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


def add_json_srl_allennlp(sentences: List[str], srl_predictor: Predictor, token_objects: List[Dict]) -> List[Dict[str, Any]]:
    structured_layer = []
    doc_token_offset = 0
    for sentence in sentences:
        srl_output = allennlp_srl(sentence, srl_predictor)
        for i, (predicate_pos, arguments) in enumerate(srl_output.pred_arg_struct.items()):
            pred_obj = {'predicateID': str(i), 
                        'locationStart': token_objects[doc_token_offset + predicate_pos]['start_char'], 
                        'tokenStart': doc_token_offset + predicate_pos,
                        'tokenEnd': doc_token_offset + predicate_pos + 1,
                        'locationEnd': token_objects[doc_token_offset + predicate_pos]['end_char'],
                        'surfaceForm': token_objects[doc_token_offset + predicate_pos]['text'],
                        'predicateLemma': token_objects[doc_token_offset + predicate_pos]['lemma'],
                        'arguments': [],
                        'method': 'allennlp_2.9.0'
                        }
            for arg in arguments:
                pred_obj['arguments'].append({
                    'argumentID': f"{i}_{arg.label}",
                    'surfaceForm': arg.text,
                    'tokenStart': doc_token_offset + arg.start,
                    'tokenEnd': doc_token_offset + arg.end+1,
                    'locationStart': token_objects[doc_token_offset + arg.start]['start_char'],
                    'locationEnd': token_objects[doc_token_offset + arg.end]['end_char'],
                    'category': arg.label
                })
            if len(pred_obj['arguments']) > 0:
                structured_layer.append(pred_obj)
        # Update Document Offset
        doc_token_offset += len(srl_output.tokens)
    return structured_layer


def add_json_ner_allennlp(sentences: List[str], ner_predictor: Predictor, token_objects: List[Dict]) -> List[Dict[str, Any]]:
    doc_entities = []
    doc_token_offset = 0
    for sentence in sentences:
        # Get Sentencewise NER
        tokenized, sentence_ner = allennlp_ner(sentence, ner_predictor, doc_token_offset)
        # Fix Sentence Offsets to fit Document
        for entity in sentence_ner:
            doc_tok_start = entity['tokenStart']
            doc_tok_end = entity['tokenEnd']
            entity['locationStart'] = token_objects[doc_tok_start]['start_char']
            entity['locationEnd'] = token_objects[doc_tok_end - 1]['end_char']
            # Append to Document-level list fo entities
            doc_entities.append(entity)
        # Update Document Offset
        doc_token_offset += len(tokenized)

    return doc_entities


def add_json_coref_allennlp(sentences: List[str], coref_predictor: Predictor, token_objects: List[Dict]) -> List[Dict[str, Any]]:
    doc_limit = 400 # We only take the first 400 tokens of the document to find coreferences there
    doc_truncated = [] 
    fake_paragraph_len = 0
    for sentence in sentences:
        fake_paragraph_len += len(sentence.split())
        if fake_paragraph_len <= doc_limit:
            doc_truncated.append(sentence)
        else:
            break
    # Get Coreference for the whole Document
    doc_truncated = " ".join(doc_truncated)
    tokenized, doc_clusters = allennlp_coref(doc_truncated, coref_predictor)

    try:
        assert len(tokenized) == len(token_objects)
        for _, entities in doc_clusters.items():
            for entity in entities:
                tok_start = entity['tokenStart']
                tok_end = entity['tokenEnd']
                entity['locationStart'] = token_objects[tok_start]['start_char']
                entity['locationEnd'] = token_objects[tok_end - 1]['end_char']
    except:
        pass

    return doc_clusters