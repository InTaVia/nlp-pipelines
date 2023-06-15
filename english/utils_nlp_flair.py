from typing import TypeVar, Dict, Any, List, Tuple
import re, os, json
from flair import __version__ as flair_version
from flair.data import Sentence
from flair.nn import Classifier
from flair.splitter import SegtokSentenceSplitter
from dataclasses import dataclass, asdict
from collections import defaultdict


def add_morphosyntax_flair(text: str, splitter: SegtokSentenceSplitter):
    sentences = splitter.split(text)
    morpho_syntax = []
    tokenized_document = []
    doc_offset, tok_offset = 0, 0
    for sent_ix, sentence in enumerate(sentences):
        morpho_syntax.append({
                "paragraphID": 0,
                "sentenceID": sent_ix,
                "text": sentence.to_plain_string(),
                "docCharOffset": doc_offset + 1,
                "docTokenOffset": tok_offset,
                "words": [{"ID": k, "FORM": tok.text} for k, tok in enumerate(sentence.tokens)]
            }) 
        tokenized_document.extend([t.text for t in sentence.tokens])
        doc_offset += len(sentence.to_plain_string())
        tok_offset += len(sentence.tokens)
    
    return morpho_syntax, tokenized_document


def run_flair(sentences: List[Sentence], task: str, flair_models: Dict[str, str], metadata: Dict[str, Any] = None) -> Dict[str, Any]:
    
    entities, entity_ids = [], {}
    relations = []

    if "relations" == task:
        ner_tagger = Classifier.load(flair_models["ner"])
        ner_tagger.predict(sentences)
        rel_tagger = Classifier.load(flair_models[task])
        rel_tagger.predict(sentences)
    else:
        tagger = Classifier.load(flair_models[task])
        tagger.predict(sentences)

    for sent_ix, sentence in enumerate(sentences):
        # Format Information for NLP Intavia
        # print(sentence.annotation_layers.keys()) # --> dict_keys(['np', 'frame', 'ner', 'relation'])
        if "chunker" == task:
            for chunk_ix, chunk in enumerate(sentence.get_spans("np")):
                token_indices = [t.idx for t in chunk]
                entities.append({
                    "ID": f"chunk_{sent_ix}_{chunk_ix}",
                    "sentenceID": sent_ix,
                    "surfaceForm": chunk.text,
                    "category": chunk.get_label("np").value,
                    "locationStart": chunk.start_position,
                    "locationEnd":  chunk.end_position,
                    "tokenStart": token_indices[0]-1,
                    "tokenEnd": token_indices[-1],
                    "score": chunk.get_label("np").score,
                    "method": f"flair_{flair_models[task]}_{flair_version}"
                })
        elif "relations" == task:
            # 1) NER
            for ent_ix, entity in enumerate(sentence.get_spans('ner')):
                token_indices = [t.idx for t in entity]
                entities.append({
                    "ID": f"ent_{sent_ix}_{ent_ix}",
                    "sentenceID": sent_ix,
                    "surfaceForm": entity.text,
                    "category": entity.get_label("ner").value,
                    "locationStart": entity.start_position,
                    "locationEnd":  entity.end_position,
                    "tokenStart": token_indices[0]-1,
                    "tokenEnd": token_indices[-1],
                    "score": entity.get_label("ner").score,
                    "method": f"flair_{flair_models['ner']}_{flair_version}"
                })
                entity_ids[(entity.start_position, entity.end_position)] = f"ent_{sent_ix}_{ent_ix}"
            # 2) Relation Extraction
            for rel_ix, relation in enumerate(sentence.get_relations('relation')):
                # print(relation.first, relation.tag ,relation.second.annotation_layers)
                # print("---------")
                first_ent_id = entity_ids.get((relation.first.start_position, relation.first.end_position))
                second_ent_id = entity_ids.get((relation.second.start_position, relation.second.end_position))
                relations.append({
                        "relationID": f"rel_{sent_ix}_{rel_ix}",
                        "sentenceID": sent_ix,
                        "subjectID": first_ent_id,
                        "objectID": second_ent_id,
                        "surfaceFormSubj": relation.first.text,
                        "relationValue": relation.tag,
                        "surfaceFormObj": relation.second.text,
                        "score": relation.score,
                        "method": f"flair_{flair_models[task]}_{flair_version}"
                    })
        elif "ner" == task: 
            for ent_ix, entity in enumerate(sentence.get_spans('ner')):
                token_indices = [t.idx for t in entity]
                entities.append({
                    "ID": f"ent_{sent_ix}_{ent_ix}",
                    "sentenceID": sent_ix,
                    "surfaceForm": entity.text,
                    "category": entity.get_label("ner").value,
                    "locationStart": entity.start_position,
                    "locationEnd":  entity.end_position,
                    "tokenStart": token_indices[0]-1,
                    "tokenEnd": token_indices[-1],
                    "score": entity.get_label("ner").score,
                    "method": f"flair_{flair_models[task]}_{flair_version}"
                })
        elif "frames" == task:
            pred_ix = 0
            for token in sentence:
                label = token.get_label("frame")
                if label.value != "O":
                    entities.append({
                        "predicateID": f"pred_{sent_ix}_{pred_ix}",
                        "sentenceID": sent_ix,
                        "locationStart": token.start_position,
                        "locationEnd": token.end_position,
                        "tokenStart": token.idx-1,
                        "tokenEnd": token.idx,
                        "surfaceForm": token.text,
                        "predicateSense": label.value,
                        "score": label.score,
                        "arguments": [],
                        "method": f"flair_{flair_models[task]}_{flair_version}"
                    })
        elif "linker" == task:
            entity_ids = metadata["entity_ids"]
            link_ix = 0
            for label in sentence.get_labels():
                if label.data_point.tag != "<unk>":
                    entities.append({
                            "linkedID": f"link_{sent_ix}_{link_ix}",
                            "sentenceID": sent_ix,
                            "entityID": entity_ids.get((label.data_point.start_position, label.data_point.end_position)), # LINK to NER IDS
                            "locationStart": label.data_point.start_position,
                            "locationEnd": label.data_point.end_position,
                            "surfaceForm": label.data_point.text,
                            "wikiTitle": label.data_point.tag,
                            "score": label.data_point.score,
                            "wikiURL": f"https://en.wikipedia.org/wiki/{label.data_point.tag}",
                            "method": f"flair_{flair_models[task]}_{flair_version}"
                        })
                    link_ix += 1

    return {"tagged_entities": entities, "tagged_relations": relations, "entity_ids": entity_ids}
