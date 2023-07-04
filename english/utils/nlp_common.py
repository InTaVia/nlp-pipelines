import re, os, json
from typing import TypeVar, Dict, Any, List, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter


Converter = TypeVar('Converter')
NafParser = TypeVar('NafParser')
SpacyLanguage = TypeVar('SpacyLanguage')
SpacyDoc = TypeVar('SpacyDoc')


@dataclass 
class TokenJSON:
    ID: int
    FORM: str
    LEMMA: str = None
    UPOS: str = None
    XPOS: str = None
    HEAD: int = None
    DEPREL: str = None
    DEPS: str = None
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
    clean_text = text
    # Get rid of everything after "See Also" Section
    match = re.search("== See also ==", clean_text)
    if match: clean_text = clean_text[:match.start()]
    # Get rid of everything after "References" Section
    match = re.search("== References ==", clean_text)
    if match: clean_text = clean_text[:match.start()]
    # Eliminate section titles and subtitles
    clean_text = re.sub(r"=+\s.+?\s=+", "\n", clean_text) 
    # Just keep ONE empty line between paragraphs
    # clean_text = re.sub(r'[\r\n]{2,}', "\n\n", clean_text)
    # clean_text = re.sub(r'\n{2,}', "\n\n", clean_text)  
    # Just keep ONE space between words (also erases the new lines!)
    clean_text = re.sub(r'[\s]+', " ", clean_text) 
    return clean_text


def get_char_offsets_from_tokenized(text: str, tokenized: List[str]):
    leftover_text = text
    global_offset = 0
    char_init_offsets = {}
    for i, tok in enumerate(tokenized):
        tok_ix = leftover_text.index(tok)
        char_init_offsets[i] = tok_ix + global_offset
        global_offset += len(tok)
        leftover_text = text[global_offset:]
    
    return char_init_offsets

# text = "Maya Angelou ( (listen) AN-jə-loh; born Marguerite Annie Johnson; April 4, 1928 – May 28, 2014) was an American memoirist, poet, and civil rights activist."
# tokens = ['Maya', 'Angelou', '(', '(', 'listen', ')', 'AN', '-', 'jə', '-', 'loh', ';', 'born', 'Marguerite', 'Annie', 'Johnson', ';', 'April', '4', ',', '1928', '–', 'May', '28', ',', '2014', ')', 'was', 'an', 'American', 'memoirist', ',', 'poet', ',', 'and', 'civil', 'rights', 'activist', '.']

# offset_dict = get_char_offsets_from_tokenized(text, tokens)
# print(offset_dict)
# for tok_ix, offset in offset_dict.items():
#     token_text = tokens[tok_ix]
#     token_char_init = offset
#     token_char_end = offset+len(token_text)
#     print(token_text, '--->' ,text[token_char_init:token_char_end])


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


def nlp_to_dict(nlp_dict: Dict[str, Any], basic_model_name: str) -> Dict[str, Any]:
    return {
        'status': '200',
        'data': {
            'text': nlp_dict['input_text'],
            'tokenization': {f"{basic_model_name}": [tok['text'] for tok in nlp_dict['token_objs']]},
            'morphology': {f"{basic_model_name}": add_morphosyntax(nlp_dict['token_objs'])},
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


def create_nlp_template(text: str, filepath: str = None) -> Tuple[Dict[str, Any], bool]:
    is_from_file = False
    default_response = {
            'text': text,
            'tokenization': {},
            'morphology': {},
            'entities': [],
            'time_expressions': [],
            'semantic_roles': [],
            'coreference': []
        }
    if not filepath:
        return default_response, is_from_file
    elif os.path.exists(filepath):
        is_from_file = True
        with open(filepath) as f:
            intavia_dict = json.load(open(filepath))
            return intavia_dict['data'], is_from_file
    else:
        return default_response, is_from_file

def reconstruct_original_sentence(token_list: List[Dict]) -> str:
    sent_text = ""
    for tok in token_list:
        if tok['space_after']:
            sent_text += tok['text'] + " "
        else:
            sent_text += tok['text']
    return sent_text

def add_morphosyntax(token_objects: List[Dict]):
    sentencized, morpho_syntax = defaultdict(list), []
    for tok in token_objects:
        sentencized[tok['sent_id']].append(tok)
    for sent_id, sentence in sentencized.items():
        sent_text = reconstruct_original_sentence(sentence)
        morpho_syntax.append({
            "paragraph": 0,
            "sentence": sent_id,
            "text": sent_text,
            "words": [asdict(nlp_token2json_token(tok)) for tok in sentence]
            })
    return morpho_syntax


def run_spacy(text: str, nlp: SpacyLanguage, nlp_processor: str = 'spacy') -> Dict:
    doc = nlp(text)
    spacy_info, spacy_tokens, spacy_sents = [], [], []
    original_sents = []
    spacy_ents = []
    for sent_ix, sent in enumerate(doc.sents):
        spacy_sents.append(" ".join([t.text for t in sent]))
        original_sents.append(sent.text)
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
        for end_id, ent in enumerate(doc.ents):
            spacy_ents.append({'ID': f"ent_{end_id}_spacy", 'surfaceForm': ent.text, 'category': ent.label_.upper(), 'locationStart': doc[ent.start].idx, 'locationEnd': doc[ent.start].idx + len(ent.text), 
                                'tokenStart': ent.start, 'tokenEnd': ent.end, 'method': nlp_processor})

    return {'spacy_doc': doc, 'sentences':spacy_sents, 'sentences_untokenized': original_sents,'tokens': spacy_tokens, 'token_objs': spacy_info, 'entities': spacy_ents}


def merge_frames_srl(srl_roles: List[Dict[str, Any]], frame_list: List[Dict[str, Any]]):
    stats = []
    frame_spans, second_chance = {}, {}
    for obj in frame_list:
        # print(f"{obj['locationStart']}_{obj['locationEnd']}", f"{obj['sentenceID']}_{obj['surfaceForm']}")
        frame_spans[f"{obj['locationStart']}_{obj['locationEnd']}"] = obj['predicateSense']
        second_chance[f"{obj['sentenceID']}_{obj['surfaceForm']}"] = obj['predicateSense']
    # Then Iterate Flair Frame files i) access their "entities" ii) match the corresponding AllenNLP iii) augment the pred_args
    for proposition in srl_roles:
        prop_key = f"{proposition['locationStart']}_{proposition['locationEnd']}"
        second_chance_key = f"{proposition['sentenceID']}_{proposition['surfaceForm']}"
        sense_match = frame_spans.get(prop_key)
        # print(prop_key, second_chance_key)
        if sense_match:
            proposition['predicateSense'] = sense_match
            # print(f"MATCH! {proposition['surfaceForm']} ---> {sense_match}")
            stats.append("match")
        elif second_chance.get(second_chance_key):
            proposition['predicateSense'] = second_chance.get(second_chance_key)
            stats.append("fuzzy_match")
            # print(f"FUZZY MATCH! {proposition['surfaceForm']} ({prop_key}) ---> {second_chance.get(second_chance_key)}")
        else:
            # print(f"NON MATCH! {proposition['surfaceForm']} ({prop_key}) ---> {sense_match}")
            stats.append("non_match")
    print(Counter(stats).most_common())
    print("-----------------------------------")
    return srl_roles