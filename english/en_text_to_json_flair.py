import os, re
import spacy, json
from typing import Dict, Any, List, Tuple, Union

from utils_wiki import get_wikipedia_article, save_wikipedia_page

from flair import __version__ as flair_version
from flair.data import Sentence
from flair.nn import Classifier
from flair.splitter import SegtokSentenceSplitter

def test_english_pipeline_json(query_tests: str = ["Albrecht Dürer"], json_path: str = "./english/data/json/"):
    if not os.path.exists(json_path): os.mkdir(json_path)
    wiki_root = "./english/data/wikipedia/"
    if not os.path.exists(wiki_root): os.mkdir(wiki_root)

    splitter = SegtokSentenceSplitter()
    flair_models = {
        "chunker": "chunk",
        "ner": 'ner-ontonotes-large', # 
        "relations": "relations", # If relations is provided then is not necessary to do NER sepparately!
        "frames": "frame",
        "linker": "linker"
    }


    wiki_matches = 0
    for query in query_tests:
        doc_title = query
        text_file_url = f"{wiki_root}/{query.replace(' ', '_').lower()}.txt"
        
        nlp_dict = {}
        json_name = doc_title.lower().replace(" ", "_") + ".flair.json"
        json_full_path = f"{json_path}/{json_name}"

        # Directly read the text file (if exists) otherwise query Wikipedia
        if os.path.exists(text_file_url):
            print(f"Reading Text directly from file: {wiki_page.title}")
            with open(text_file_url) as f:
                text = f.read()
            wiki_matches += 1
        # elif os.path.exists(json_full_path):
        #     print(f"NLP File Exists! Loading ...")
        else:
            wiki_page = get_wikipedia_article(doc_title)
            if wiki_page:
                print(f"Found a Page: {wiki_page.title}")
                text = wiki_page.content
                wiki_matches += 1
                save_wikipedia_page(wiki_page, text_file_url, include_metadata=True)
            else:
                print(f"Couldn't find {query}!")
        
        # Step 1: Clean Text
        clean_text = preprocess_and_clean_text(text)

        # Step 2: Split Into Sentences
        sentences = splitter.split(clean_text)
        morpho_syntax = {"model": f"flair_{flair_version}", "data": []}
        tokenized_document = []
        doc_offset, tok_offset = 0, 0
        for sent_ix, sentence in enumerate(sentences):
            morpho_syntax['data'].append({
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

        # # Step 3: Run Heideltime
        nlp_dict['time_expressions'] = [] # unlp.add_json_heideltime(clean_text, heideltime_parser)

        # Step 4: Run Flair Taggers
        nlp_dict['phrase_chunks'] = run_flair(sentences, "chunker", flair_models)["tagged_entities"]
        nlp_dict['semantic_roles'] = run_flair(sentences, "frames", flair_models)["tagged_entities"]
        ent_rel_out = run_flair(sentences, "relations", flair_models)
        nlp_dict['entities'] = ent_rel_out["tagged_entities"]
        nlp_dict['relations'] = ent_rel_out["tagged_relations"]
        # Must restart the sentence to erase previous tags
        sentences = splitter.split(clean_text)
        nlp_dict['linked_entities'] = run_flair(sentences, "linker", flair_models, metadata={"entity_ids":ent_rel_out["entity_ids"]})["tagged_entities"]

        # # Step N: Build General Dict
        nlp_dict['input_text'] = clean_text
        response = {
            'status': '200',
            'data': {
                'text': nlp_dict['input_text'],
                'morphology': morpho_syntax,
                'tokenization': tokenized_document
            }
        }
        for key in nlp_dict.keys():
            if key not in response['data']:
                response['data'][key] = nlp_dict[key]
        # Write to Disk
        json.dump(response, open(json_full_path, "w"), indent=2, ensure_ascii=False)

    
    print(f"Found {wiki_matches} out of {len(query_tests)} articles in Wikipedia ({wiki_matches/len(query_tests) * 100:.2f}%)")




def preprocess_and_clean_text(text: str) -> str:
    clean_text = re.sub(r'[\r\n]+', " ", text)
    clean_text = re.sub(r'"', ' " ', clean_text)
    clean_text = re.sub(r'[\s]+', " ", clean_text)
    return clean_text



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


if __name__ == "__main__":
    #query_tests = ["William the Silent", "Albrecht Dürer", "Vincent van Gogh", "Constantijn Huygens", "Baruch Spinoza", "Erasmus of Rotterdam",
                    #"Christiaan Huygens", "Rembrandt van Rijn", "Antoni van Leeuwenhoek", "John von Neumann", "Johan de Witt"]
    test_english_pipeline_json()