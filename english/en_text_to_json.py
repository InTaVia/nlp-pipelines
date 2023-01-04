import os
import spacy, json
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict
from allennlp.predictors import Predictor

from python_heideltime import Heideltime
from bs4 import BeautifulSoup

from utils_wiki import get_wikipedia_article
import utils_nlp as unlp


def test_english_pipeline_json(query_tests: str = ["William the Silent"], json_path: str = "./english/data/json/"):
    nlp = spacy.load("en_core_web_sm")
    srl_predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz")
    ner_predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/ner-elmo.2021-02-12.tar.gz")

    heideltime_parser = Heideltime()
    heideltime_parser.set_language('ENGLISH')
    heideltime_parser.set_document_type('NARRATIVES')

    wiki_matches = 0
    for query in query_tests:
        doc_title = query
        wiki_page = get_wikipedia_article(doc_title)
        if wiki_page:
            print(f"Found a Page: {wiki_page.title}")
            nlp_dict = {}
            text = wiki_page.summary
            wiki_matches += 1
            json_name = doc_title.lower().replace(" ", "_") + ".json"
            # Step 1: Clean Text
            clean_text = unlp.preprocess_and_clean_text(text)
            # Step 2: Basic Processing with SpaCy
            spacy_dict = unlp.run_spacy(clean_text, nlp)
            # Step 3: Run HeidelTime
            nlp_dict['time_expressions'] = unlp.add_json_heideltime(clean_text, heideltime_parser)
            # Step 4: Run AllenNLP SRL
            nlp_dict['semantic_roles'] = unlp.add_json_srl_allennlp(spacy_dict['sentences'], srl_predictor, spacy_dict['token_objs'])
            # Step N: Build General Dict
            nlp_dict['input_text'] = clean_text
            nlp_dict['token_objs'] = spacy_dict['token_objs']
            nlp_dict['entities'] = spacy_dict['entities']
            nlp_dict['entities'] += unlp.add_json_ner_allennlp(spacy_dict['sentences'], ner_predictor, spacy_dict['token_objs'])
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
    test_english_pipeline_json(query_tests)