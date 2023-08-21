"""
    This script already assumes that the Wikipedia Texts are present in the HardDrive

    EXAMPLE:
        python english/en_text_to_json_flair.py "english/data/wikipedia/top_women/"

"""

import os, re, glob, sys
import json
from typing import Dict, Any, List, Tuple, Union

import spacy
from spacy import __version__ as spacy_version
from spacy.tokens import Doc

from python_heideltime import Heideltime
from utils.nlp_heideltime import add_json_heideltime

from flair import __version__ as flair_version
from flair.splitter import SegtokSentenceSplitter
from flair.nn import Classifier

from utils.utils_wiki import get_wikipedia_article, save_wikipedia_page
from utils.nlp_flair import run_flair
from utils.nlp_common import preprocess_and_clean_text, create_nlp_template, run_spacy, add_morphosyntax



spacy_model = "en_core_web_lg"
spacy_nlp = spacy.load("en_core_web_lg")

heideltime_parser = Heideltime()
heideltime_parser.set_language('ENGLISH')
heideltime_parser.set_document_type('NARRATIVES')

splitter = SegtokSentenceSplitter()
ner_tagger = Classifier.load('ner-ontonotes-large')
rel_tagger = Classifier.load('relations')
frames_tagger = Classifier.load('frame')
linker_tagger = Classifier.load('linker')

flair_models = {
    "chunker": "chunk",
    "ner": ner_tagger, # These are the specific pre-trained models, can be switched...
    "relations": rel_tagger,
    "frames": frames_tagger,
    "linker": linker_tagger
}


def run_flair_pipeline(text: str):
    sentences = splitter.split(text)
    tokenized_document, morpho_syntax = [], []
    doc_offset, tok_offset = -1, 0
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

    # Empty NLP Dict
    nlp_dict, is_from_file = create_nlp_template(text)

    # Add Flair Morphology
    nlp_dict['morpho_syntax'][f"flair_{flair_version}"] = morpho_syntax
    nlp_dict['tokenization'][f"flair_{flair_version}"] = tokenized_document

    # OPTIONAL - Add Also Spacy Morphology
    spacy_dict = run_spacy(text, spacy_nlp)
    nlp_dict['tokenization'][f'spacy_{spacy_model}_{spacy_version}'] = [tok['text'] for tok in spacy_dict['token_objs']],
    nlp_dict['morpho_syntax'][f'spacy_{spacy_model}_{spacy_version}'] = add_morphosyntax(spacy_dict['token_objs'])

    # # Run Heideltime
    nlp_dict['time_expressions'] = add_json_heideltime(text, heideltime_parser)

    # Run Flair Taggers
    frame_list = run_flair(sentences, "frames", flair_models)["tagged_entities"]
    nlp_dict['frames'] = frame_list
    
    ent_rel_out = run_flair(sentences, "relations", flair_models)
    nlp_dict['entities'] += ent_rel_out["tagged_entities"]
    nlp_dict['relations'] = ent_rel_out["tagged_relations"]
    # Must restart the sentence to erase previous tags
    sentences = splitter.split(text)
    nlp_dict['linked_entities'] = run_flair(sentences, "linker", flair_models, metadata={"entity_ids":ent_rel_out["entity_ids"]})["tagged_entities"]
    
    # # Step N: Build General Dict
    nlp_dict['input_text'] = text
    intavia_dict = {
        'status': '200',
        'data': nlp_dict
    }

    return intavia_dict



def test_english_pipeline_json(query_tests: str = ["Albrecht Dürer"], json_path: str = "./english/data/json/"):
    if not os.path.exists(json_path): os.mkdir(json_path)
    wiki_root = "./english/data/wikipedia/"
    if not os.path.exists(wiki_root): os.mkdir(wiki_root)

    wiki_matches = 0
    for query in query_tests:
        doc_title = query
        text_file_url = f"{wiki_root}/{query.replace(' ', '_').lower()}.txt"
        
        
        json_name = doc_title.lower().replace(" ", "_") + ".json"
        json_full_path = f"{json_path}/{json_name}"

        # Directly read the text file (if exists) otherwise query Wikipedia
        if os.path.exists(json_full_path):
            print(f"An NLP file exists for {doc_title}. Reading directly from JSON.")
            nlp_dict, is_from_file = create_nlp_template("", json_full_path)
            # Step 1: Clean Text
            clean_text = nlp_dict['text']
            wiki_matches += 1
        else:
            wiki_page = get_wikipedia_article(doc_title)
            if wiki_page:
                print(f"Found a Page: {wiki_page.title}")
                text = wiki_page.content[:1000]
                # Step 1: Clean Text
                clean_text = preprocess_and_clean_text(text)
                wiki_matches += 1
                save_wikipedia_page(wiki_page, text_file_url, include_metadata=True)
            else:
                print(f"Couldn't find {query}!")
                break

        # Step 2: Run Flair Pipeline
        intavia_dict = run_flair_pipeline(clean_text)

        # Write to Disk
        json.dump(intavia_dict, open(json_full_path, "w"), indent=2, ensure_ascii=False)

    
    print(f"Found {wiki_matches} out of {len(query_tests)} articles in Wikipedia ({wiki_matches/len(query_tests) * 100:.2f}%)")


def process_wiki_files(root_dir: str):
    ix = 0
    for person_file in glob.glob(f"{root_dir}/*.txt"):
        ix += 1
        person_name = os.path.basename(person_file)[:-4]
        print(ix, person_file, person_name)
        text_filename = person_file
        json_nlp_filename = f"{root_dir}/{person_name.replace(' ', '_').lower()}.nlp.flair.json"
        if not os.path.exists(json_nlp_filename):
            # Load Wikipedia Text from File
            with open(text_filename) as f:
                text = f.read()
                # text = text[:1000]
                text = preprocess_and_clean_text(text)
            # Run Pipeline
            intavia_dict = run_flair_pipeline(text)
            # Write to Disk
            json.dump(intavia_dict, open(json_nlp_filename, "w"), indent=2, ensure_ascii=False)
        else:
            print("NLP File already exists. Skipping...")

if __name__ == "__main__":
    # query_tests = ["William the Silent", "Albrecht Duerer", "Vincent van Gogh", "Constantijn Huygens", "Baruch Spinoza", "Erasmus of Rotterdam",
    #                 "Christiaan Huygens", "Rembrandt van Rijn", "Antoni van Leeuwenhoek", "John von Neumann", "Johan de Witt"]
    # test_english_pipeline_json(["Albrecht Duerer"])
    
    if len(sys.argv) == 2:
        root_dir = sys.argv[1]
        if os.path.exists(os.path.dirname(root_dir)):
            process_wiki_files(root_dir)
        else:
            print(f"Cannot find Path {root_dir}")
    else:
        print("USAGE: python en_wikipedia_to_json.py <root_folder_with_biographies>")