import os, re
import spacy, json
from typing import Dict, Any, List, Tuple, Union

from python_heideltime import Heideltime

from utils_wiki import get_wikipedia_article, save_wikipedia_page

from flair import __version__ as flair_version
from flair.splitter import SegtokSentenceSplitter
from utils_nlp_flair import run_flair
from utils_nlp_common import preprocess_and_clean_text, add_json_heideltime


def test_english_pipeline_json(query_tests: str = ["Albrecht Dürer"], json_path: str = "./english/data/json/"):
    if not os.path.exists(json_path): os.mkdir(json_path)
    wiki_root = "./english/data/wikipedia/"
    if not os.path.exists(wiki_root): os.mkdir(wiki_root)

    heideltime_parser = Heideltime()
    heideltime_parser.set_language('ENGLISH')
    heideltime_parser.set_document_type('NARRATIVES')

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
                text = wiki_page.summary
                wiki_matches += 1
                save_wikipedia_page(wiki_page, text_file_url, include_metadata=True)
            else:
                print(f"Couldn't find {query}!")
                break
        
        # Step 1: Clean Text
        clean_text = preprocess_and_clean_text(text)

        # Step 2: Split Into Sentences
        sentences = splitter.split(clean_text)
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

        # # Step 3: Run Heideltime
        nlp_dict['time_expressions'] = add_json_heideltime(clean_text, heideltime_parser)

        # Step 4: Run Flair Taggers
        # nlp_dict['phrase_chunks'] = run_flair(sentences, "chunker", flair_models)["tagged_entities"]
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
                'morphology': {f"flair_{flair_version}": morpho_syntax},
                'tokenization': {f"flair_{flair_version}": tokenized_document}
            }
        }
        for key in nlp_dict.keys():
            if key not in response['data']:
                response['data'][key] = nlp_dict[key]
        # Write to Disk
        json.dump(response, open(json_full_path, "w"), indent=2, ensure_ascii=False)

    
    print(f"Found {wiki_matches} out of {len(query_tests)} articles in Wikipedia ({wiki_matches/len(query_tests) * 100:.2f}%)")



if __name__ == "__main__":
    #query_tests = ["William the Silent", "Albrecht Dürer", "Vincent van Gogh", "Constantijn Huygens", "Baruch Spinoza", "Erasmus of Rotterdam",
                    #"Christiaan Huygens", "Rembrandt van Rijn", "Antoni van Leeuwenhoek", "John von Neumann", "Johan de Witt"]
    test_english_pipeline_json(["Constantijn Huygens"])