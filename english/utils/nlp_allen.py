import re, os, json
from typing import TypeVar, Dict, Any, List, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict

from allennlp.predictors import Predictor

from utils.nlp_common import SRL_Output, SRL_Argument, get_char_offsets_from_tokenized


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


def allennlp_ner(text: str, ner_predictor: Predictor, char_init_offset: int = 0, token_init_offset: int = 0, sent_index: int = 0) -> Tuple[List, List]:
    output = ner_predictor.predict(text)
    tokenized_sentence = output['words']
    char_offset_dict = get_char_offsets_from_tokenized(text, tokenized_sentence)
    
    sentence_entities = []
    ent_tokens, ent_indices, ent_label = [], [], ""
    for ix, bio_tag in enumerate(output["tags"]):
        if bio_tag.startswith("U-"):
            location_start = char_init_offset + char_offset_dict[ix]
            location_end = location_start + len(output["words"][ix])
            sentence_entities.append({'ID': None, 'sentenceID': sent_index, 'surfaceForm': output["words"][ix], 'category': bio_tag[2:].upper(), 
                                      'locationStart': location_start, 'locationEnd': location_end, 'tokenStart': token_init_offset + ix, 'tokenEnd': token_init_offset + ix + 1,
                                        'sentenceTokenStart': ix, 'sentenceTokenEnd': ix+1, 'method': 'allennlp_2.9.0'})
            ent_label = ""
            ent_indices = []
            ent_tokens = []
        elif bio_tag.startswith("B-"):
            if len(ent_label) > 0:
                location_start = char_init_offset + char_offset_dict[ent_indices[0]]
                location_end = location_start + len(" ".join(ent_tokens))
                sentence_entities.append({'ID': None, 'sentenceID': sent_index, 'surfaceForm': " ".join(ent_tokens), 'category': ent_label.upper(), 
                                          'locationStart': location_start, 'locationEnd': location_end, 'tokenStart': token_init_offset + ent_indices[0], 'tokenEnd': token_init_offset + ent_indices[-1]+1, 
                                            'sentenceTokenStart': ent_indices[0], 'sentenceTokenEnd': ent_indices[-1]+1, 'method': 'allennlp_2.9.0'})
                ent_indices = []
                ent_tokens = []
            ent_label = bio_tag[2:]
            ent_tokens.append(output['words'][ix])
            ent_indices.append(ix)
        elif bio_tag.startswith("I-") or bio_tag.startswith("L-"):
            ent_tokens.append(output['words'][ix])
            ent_indices.append(ix)
        elif bio_tag == "O" and len(ent_label) > 0:
            location_start = char_init_offset + char_offset_dict[ent_indices[0]]
            location_end = location_start + len(" ".join(ent_tokens))
            sentence_entities.append({'ID': None, 'sentenceID': sent_index, 'surfaceForm': " ".join(ent_tokens), 'category': ent_label.upper(), 
                                      'locationStart': location_start, 'locationEnd': location_end, 'tokenStart': token_init_offset + ent_indices[0], 'tokenEnd': token_init_offset + ent_indices[-1]+1, 
                                        'sentenceTokenStart': ent_indices[0], 'sentenceTokenEnd': ent_indices[-1]+1, 'method': 'allennlp_2.9.0'})
            ent_label = ""
            ent_indices = []
            ent_tokens = []
    
    return tokenized_sentence, sentence_entities


def allennlp_coref(text: str, coref_predictor: Predictor) -> Tuple[List[str], Dict[str, Any]]:
    allennlp_output = coref_predictor.predict(text)
    tokens = allennlp_output["document"]
    char_offset_dict = get_char_offsets_from_tokenized(text, tokens)
    clusters = defaultdict(list)
    for cluster_id, cluster_elems in enumerate(allennlp_output["clusters"]):
        for j, (start, end) in enumerate(cluster_elems):
            labeled_span = " ".join(tokens[start:end+1])
            location_start = char_offset_dict[start]
            location_end = location_start + len(labeled_span)
            clusters[cluster_id].append({'ID': f"{cluster_id}_{j}",'surfaceForm': labeled_span, 'locationStart': location_start, 'locationEnd': location_end, 
                                        'method': 'allennlp_2.9.0'}) # 'tokenStart': start, 'tokenEnd': end+1,
    return tokens, clusters


## Functions to Add JSON Layers (via Script or API)

def add_json_srl_allennlp(sentences: List[str], srl_predictor: Predictor, token_objects: List[Dict]) -> List[Dict[str, Any]]:
    structured_layer = []
    doc_token_offset = 0
    # To coincide with the Pre-Tokenized TokenObjects instead of AllenNLP tokenization
    sentence_dict = defaultdict(int)
    char_starts2token, char_ends2token = {}, {}
    for k, tok in enumerate(token_objects):
        sentence_dict[int(tok['sent_id'])] += 1
        char_starts2token[tok['start_char']] = k
        char_ends2token[tok['end_char']] = k
    sentence_token_lengths = {}
    for sent_id, sent_len in sorted(sentence_dict.items()):
        sentence_token_lengths[sent_id] = sent_len

    char_init_offset = 0
    for i, sentence in enumerate(sentences):
        if len(sentence) == 0: continue
        srl_output = allennlp_srl(sentence, srl_predictor)
        char_offset_dict = get_char_offsets_from_tokenized(sentence, srl_output.tokens)
        for j, (predicate_pos, arguments) in enumerate(srl_output.pred_arg_struct.items()):
            location_start = char_init_offset + char_offset_dict[predicate_pos]
            location_end = location_start + len(srl_output.tokens[predicate_pos])
            tok_start = char_starts2token.get(location_start)
            tok_end = char_ends2token.get(location_end)
            pred_obj = {'predicateID': str(j), 
                        'sentenceID': str(i),
                        #'tokenStart': doc_token_offset + predicate_pos,
                        #'tokenEnd': doc_token_offset + predicate_pos + 1,
                        'locationStart': location_start, 
                        'locationEnd': location_end,
                        #'sentenceTokenStart':predicate_pos,
                        #'sentenceTokenEnd':predicate_pos + 1,
                        'surfaceForm': srl_output.tokens[predicate_pos],
                        'arguments': [],
                        'method': 'allennlp_2.9.0'
                        }
            # if tok_start: pred_obj['tokenStart'] = doc_token_offset + tok_start
            # if tok_end: pred_obj['tokenEnd'] = doc_token_offset + tok_end
            for arg in arguments:
                arg_loc_start = char_init_offset + char_offset_dict[arg.start]
                arg_loc_end = arg_loc_start + len(arg.text)
                pred_obj['arguments'].append({
                    'argumentID': f"{i}_{arg.label}",
                    'surfaceForm': arg.text,
                    #'tokenStart': doc_token_offset + arg.start,
                    #'tokenEnd': doc_token_offset + arg.end+1,
                    'locationStart': arg_loc_start,
                    'locationEnd': arg_loc_end,
                    #'sentenceTokenStart':arg.start,
                    #'sentenceTokenEnd':arg.end+1,
                    'category': arg.label
                })
            if len(pred_obj['arguments']) > 0:
                structured_layer.append(pred_obj)
        # Update Document Offset
        doc_token_offset += sentence_token_lengths.get(i, 0)
        char_init_offset += len(sentence) + 1
    return structured_layer


def add_json_ner_allennlp(sentences: List[str], ner_predictor: Predictor, token_objects: List[Dict]) -> List[Dict[str, Any]]:
    doc_entities = []
    doc_char_offset, doc_token_offset = 0, 0
    # To coincide with the Pre-Tokenized TokenObjects instead of AllenNLP tokenization
    sentence_dict = defaultdict(int)
    for tok in token_objects:
        sentence_dict[int(tok['sent_id'])] += 1
    sentence_token_lengths = []
    for _, sent_len in sorted(sentence_dict.items()):
        sentence_token_lengths.append(sent_len)

    for i, sentence in enumerate(sentences):
        # Get Sentencewise NER
        tokenized, sentence_ner = allennlp_ner(sentence, ner_predictor, doc_char_offset, doc_token_offset, sent_index=i)
        # Fix Sentence Offsets to fit Document
        for j, entity in enumerate(sentence_ner):
            entity['ID'] = f"ent_{i}_{j}_allen"
            # Append to Document-level list fo entities
            doc_entities.append(entity)
        # Update Document Offset
        doc_token_offset += sentence_token_lengths[i]
        doc_char_offset += len(sentence) + 1

    return doc_entities


def add_json_coref_allennlp(sentences: List[str], coref_predictor: Predictor, token_objects: List[Dict]) -> List[Dict[str, Any]]:
    doc_limit = 1000 # We only take the first 400 tokens of the document to find coreferences there
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

    return doc_clusters