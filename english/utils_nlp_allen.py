import re, os, json
from typing import TypeVar, Dict, Any, List, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict

from allennlp.predictors import Predictor

from utils_nlp_common import SRL_Output, SRL_Argument


Converter = TypeVar('Converter')
NafParser = TypeVar('NafParser')
InfoExtractor = TypeVar('InfoExtractor')


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