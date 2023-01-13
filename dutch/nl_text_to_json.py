import os
import spacy, json
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict
import stanza
from python_heideltime import Heideltime
from flair.models import SequenceTagger
from flair.tokenization import SegtokSentenceSplitter

from utils_wiki import get_wikipedia_article
import utils_nlp as unlp



def test_dutch_pipeline_json(query_tests: str = ["Antoni van Leeuwenhoek"], json_path: str = "./dutch/data/json/"):
    nlp = stanza.Pipeline(lang="nl", processors="tokenize,lemma,pos,depparse,ner")

    heideltime_parser = Heideltime()
    heideltime_parser.set_language('DUTCH')
    heideltime_parser.set_document_type('NARRATIVES')

    flair_tagger = SequenceTagger.load("flair/ner-dutch-large")
    flair_splitter = SegtokSentenceSplitter()

    wiki_matches = 0
    for query in query_tests:
        doc_title = query
        wiki_page = get_wikipedia_article(doc_title, language="nl")
        if wiki_page:
            print(f"Found a Page: {wiki_page.title}")
            nlp_dict = {}
            text = wiki_page.summary
            wiki_matches += 1
            json_name = doc_title.lower().replace(" ", "_") + ".json"
            # Step 1: Clean Text
            clean_text = unlp.preprocess_and_clean_text(text)
            # Step 2: Basic Processing with SpaCy
            stanza_dict = unlp.run_stanza(clean_text, nlp)
            # Step 3: Run HeidelTime
            nlp_dict['time_expressions'] = unlp.add_json_heideltime(clean_text, heideltime_parser)
            # Step 4: Run Flair for Better NER
            flair_output = unlp.run_flair(clean_text, flair_tagger, flair_splitter)
            flair_entities = unlp.add_json_flair_ner(flair_output)
            # Step N: Build General Dict
            nlp_dict['input_text'] = clean_text
            nlp_dict['token_objs'] = stanza_dict['token_objs']
            nlp_dict['entities'] = stanza_dict['entities']
            nlp_dict['entities'] += flair_entities
            response = unlp.nlp_to_dict(nlp_dict)
            # Write to Disk
            if not os.path.exists(json_path): os.mkdir(json_path)
            json.dump(response, open(f"{json_path}/{json_name}", "w"), indent=2)
        else:
            print(f"Couldn't find {query}!")

    print(f"Found {wiki_matches} out of {len(query_tests)} articles in Wikipedia ({wiki_matches/len(query_tests) * 100:.2f}%)")


if __name__ == "__main__":
    query_tests = ["William the Silent", "Albercht Durer", "Vincent van Gogh", "Constantijn Huygens", "Baruch Spinoza", "Erasmus of Rotterdam",
                   "Christiaan Huygens", "Rembrandt van Rijn", "Antoni van Leeuwenhoek", "John von Neumann", "Johan de Witt"]
    test_dutch_pipeline_json()