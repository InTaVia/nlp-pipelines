from typing import TypeVar, Dict, Any, List, Tuple
import re, os, json
from flair import __version__ as flair_version
from flair.data import Sentence
from flair.nn import Classifier
from flair.splitter import SegtokSentenceSplitter
from dataclasses import dataclass, asdict
from collections import Counter
from tqdm import tqdm


def add_morphosyntax_flair(text: str, splitter: SegtokSentenceSplitter):
    sentences = splitter.split(text)
    morpho_syntax = []
    tokenized_document = []
    doc_offset, tok_offset = 0, 0
    for sent_ix, sentence in enumerate(sentences):
        words = []
        for k, tok in enumerate(sentence.tokens):
            space_after = True if tok.whitespace_after == 1 else False
            end_position = tok.start_position + len(tok.text)
            words.append({"ID": k, "FORM": tok.text, "MISC": {"SpaceAfter": space_after, "StartChar": tok.start_position, "EndChar": end_position}})
        morpho_syntax.append({
                "paragraphID": 0,
                "sentenceID": sent_ix,
                "text": sentence.to_plain_string(),
                "tokenized": sentence.to_tokenized_string(),
                "docCharOffset": doc_offset + 1,
                "docTokenOffset": tok_offset,
                "words": words
            }) 
        tokenized_document.extend([t.text for t in sentence.tokens])
        doc_offset += len(sentence.to_plain_string())
        tok_offset += len(sentence.tokens)

    return morpho_syntax, tokenized_document


def run_flair(sentences: List[Sentence], task: str, flair_models: Dict[str, str], metadata: Dict[str, Any] = None) -> Dict[str, Any]:
    
    entities, entity_ids = [], {}
    relations = []

    if "relations" == task:
        ner_tagger = flair_models["ner"]
        ner_tagger.predict(sentences)
        rel_tagger = flair_models[task]
        rel_tagger.predict(sentences)
    else:
        tagger = flair_models[task]
        tagger.predict(sentences)

    doc_offset, doc_token_offset = 0, 0
    sent_ix = 0
    print(task)
    for sentence in tqdm(sentences):
        # Format Information for NLP Intavia
        # print(sentence.annotation_layers.keys()) # --> dict_keys(['np', 'frame', 'ner', 'relation'])
        if "chunker" == task:
            for chunk_ix, chunk in enumerate(sentence.get_spans("np")):
                token_indices = [t.idx for t in chunk]
                entities.append({
                    "ID": f"chunk_{sent_ix}_{chunk_ix}_flair",
                    "sentenceID": sent_ix,
                    "surfaceForm": chunk.text,
                    "category": chunk.get_label("np").value,
                    "locationStart": doc_offset + chunk.start_position,
                    "locationEnd":  doc_offset + chunk.end_position,
                    "tokenStart": doc_token_offset + token_indices[0]-1,
                    "tokenEnd": doc_token_offset + token_indices[-1],
                    'sentenceTokenStart': token_indices[0]-1,
                    'sentenceTokenEnd': token_indices[-1],
                    "sentenceLocationStart": chunk.start_position,
                    "sentenceLocationEnd":  chunk.end_position,
                    "score": chunk.get_label("np").score,
                    "method": f"flair_{task}_{flair_version}"
                })
        elif "relations" == task:
            # 1) NER
            for ent_ix, entity in enumerate(sentence.get_spans('ner')):
                token_indices = [t.idx for t in entity]
                entities.append({
                    "ID": f"ent_{sent_ix}_{ent_ix}_flair",
                    "sentenceID": sent_ix,
                    "surfaceForm": entity.text,
                    "category": entity.get_label("ner").value,
                    "locationStart": doc_offset + entity.start_position,
                    "locationEnd": doc_offset + entity.end_position,
                    "tokenStart": doc_token_offset + token_indices[0]-1,
                    "tokenEnd": doc_token_offset + token_indices[-1],
                    'sentenceTokenStart': token_indices[0]-1,
                    'sentenceTokenEnd': token_indices[-1],
                    "sentenceLocationStart": entity.start_position,
                    "sentenceLocationEnd":  entity.end_position,
                    "score": entity.get_label("ner").score,
                    "method": f"flair_ner_{flair_version}"
                })
                entity_ids[(entity.start_position, entity.end_position)] = f"ent_{sent_ix}_{ent_ix}_flair"
            # 2) Relation Extraction
            for rel_ix, relation in enumerate(sentence.get_relations('relation')):
                # print(relation.first, relation.tag ,relation.second.annotation_layers)
                # print("---------")
                first_ent_id = entity_ids.get((relation.first.start_position, relation.first.end_position))
                second_ent_id = entity_ids.get((relation.second.start_position, relation.second.end_position))
                relations.append({
                        "relationID": f"rel_{sent_ix}_{rel_ix}_flair",
                        "sentenceID": sent_ix,
                        "subjectID": first_ent_id,
                        "objectID": second_ent_id,
                        "surfaceFormSubj": relation.first.text,
                        "relationValue": relation.tag,
                        "surfaceFormObj": relation.second.text,
                        "score": relation.score,
                        "method": f"flair_{task}_{flair_version}"
                    })
        elif "ner" == task: 
            for ent_ix, entity in enumerate(sentence.get_spans('ner')):
                token_indices = [t.idx for t in entity]
                entities.append({
                    "ID": f"ent_{sent_ix}_{ent_ix}_flair",
                    "sentenceID": sent_ix,
                    "surfaceForm": entity.text,
                    "category": entity.get_label("ner").value,
                    "locationStart": doc_offset + entity.start_position,
                    "locationEnd": doc_offset + entity.end_position,
                    "tokenStart": doc_token_offset + token_indices[0]-1,
                    "tokenEnd": doc_token_offset + token_indices[-1],
                    "sentenceTokenStart": token_indices[0]-1,
                    "sentenceTokenEnd": token_indices[-1],
                    "sentenceLocationStart": entity.start_position,
                    "sentenceLocationEnd":  entity.end_position,
                    "score": entity.get_label("ner").score,
                    "method": f"flair_{task}_{flair_version}"
                })
        elif "frames" == task:
            pred_ix = 0
            for token in sentence:
                label = token.get_label("frame")
                if label.value != "O":
                    entities.append({
                        "predicateID": f"pred_{sent_ix}_{pred_ix}_flair",
                        "sentenceID": sent_ix,
                        "locationStart": doc_offset + token.start_position,
                        "locationEnd": doc_offset + token.end_position,
                        "tokenStart": doc_token_offset + token.idx-1,
                        "tokenEnd": doc_token_offset + token.idx,
                        "sentenceTokenStart": token.idx-1,
                        "sentenceTokenEnd": token.idx,
                        "sentenceLocationStart": token.start_position,
                        "sentenceLocationEnd":  token.end_position,
                        "surfaceForm": token.text,
                        "predicateSense": label.value,
                        "score": label.score,
                        "arguments": [],
                        "method": f"flair_{task}_{flair_version}"
                    })
        elif "linker" == task:
            entity_ids = metadata["entity_ids"]
            link_ix = 0
            for label in sentence.get_labels():
                if label.data_point.tag != "<unk>":
                    entities.append({
                            "linkedID": f"link_{sent_ix}_{link_ix}_flair",
                            "sentenceID": sent_ix,
                            "entityID": entity_ids.get((label.data_point.start_position, label.data_point.end_position)), # LINK to NER IDS
                            "locationStart": doc_offset + label.data_point.start_position,
                            "locationEnd": doc_offset + label.data_point.end_position,
                            "sentenceLocationStart": label.data_point.start_position,
                            "sentenceLocationEnd": label.data_point.end_position,
                            "surfaceForm": label.data_point.text,
                            "wikiTitle": label.data_point.tag,
                            "score": label.data_point.score,
                            "wikiURL": f"https://en.wikipedia.org/wiki/{label.data_point.tag}",
                            "method": f"flair_{task}_{flair_version}"
                        })
                    link_ix += 1
        doc_offset += len(sentence.to_plain_string()) + 1
        doc_token_offset += len(sentence.tokens)
        sent_ix += 1

    return {"tagged_entities": entities, "tagged_relations": relations, "entity_ids": entity_ids}