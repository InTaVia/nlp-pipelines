"""
    This script already assumes that the Wikipedia Texts are present in the HardDrive

    EXAMPLE:
        python english/en_text_to_json.py "english/data/wikipedia/top_women/"

"""
import os, glob, sys
import spacy, json
from typing import List
from spacy import __version__ as spacy_version
from allennlp.predictors import Predictor

from python_heideltime import Heideltime

from utils.utils_wiki import get_wikipedia_article
import utils.nlp_common as unlp
import utils.nlp_allen as anlp
from utils.nlp_heideltime import add_json_heideltime

# Initialize Models just ONCE Globally
spacy_model = "en_core_web_sm"
nlp = spacy.load("en_core_web_sm")
srl_predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz")
ner_predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/ner-elmo.2021-02-12.tar.gz")
coref_predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2021.03.10.tar.gz")

heideltime_parser = Heideltime()
heideltime_parser.set_language('ENGLISH')
heideltime_parser.set_document_type('NARRATIVES')



def run_allennlp_pipeline(text: str):
    nlp_dict = {}
    # Step 1: Clean Text ??
    clean_text = unlp.preprocess_and_clean_text(text)
    # Step 2: Basic Processing with SpaCy
    spacy_dict = unlp.run_spacy(clean_text, nlp)
    # Step 3: Run HeidelTime
    nlp_dict['time_expressions'] = add_json_heideltime(clean_text, heideltime_parser)
    # Step 4: Run AllenNLP SRL
    nlp_dict['semantic_roles'] = anlp.add_json_srl_allennlp(spacy_dict['sentences'], srl_predictor, spacy_dict['token_objs'])
    # Step 5: Run AllenNLP Coref
    nlp_dict['coreference'] = anlp.add_json_coref_allennlp(spacy_dict['sentences'], coref_predictor, spacy_dict['token_objs'])
    # Step N: Build General Dict
    nlp_dict['input_text'] = clean_text
    nlp_dict['token_objs'] = spacy_dict['token_objs']
    nlp_dict['entities'] = spacy_dict['entities']
    nlp_dict['entities'] += anlp.add_json_ner_allennlp(spacy_dict['sentences'], ner_predictor, spacy_dict['token_objs'])
    response = unlp.nlp_to_dict(nlp_dict, basic_model_name='spacy') # f'spacy_{spacy_model}_{spacy_version}'
    return response


def test_english_pipeline_json(query_tests: List[str] = ["William the Silent"], output_json_path: str = "./english/data/json/"):
    wiki_matches = 0
    for query in query_tests:
        doc_title = query
        wiki_page = get_wikipedia_article(doc_title)
        if wiki_page:
            print(f"Found a Page: {wiki_page.title}")
            text = wiki_page.summary # BEWARE! Since this is a test we only process the SUMMARY from each Article
            wiki_matches += 1
        else:
            print(f"Couldn't find {query}!")
            text = None
        
        if text:
            json_name = doc_title.lower().replace(" ", "_") + ".json"
            intavia_dict = run_allennlp_pipeline(doc_title, text)
            # Write to Disk
            if not os.path.exists(output_json_path): os.mkdir(output_json_path)
            json.dump(intavia_dict, open(f"{output_json_path}/{json_name}", "w"), indent=2, ensure_ascii=False)
        
    print(f"Found {wiki_matches} out of {len(query_tests)} articles in Wikipedia ({wiki_matches/len(query_tests) * 100:.2f}%)")


def process_wiki_files(root_dir: str):
    for person_file in glob.glob(f"{root_dir}/*.txt"):
        person_name = os.path.basename(person_file)[:-4]
        print(person_file, person_name)
        text_filename = person_file
        json_nlp_filename = f"{root_dir}/{person_name.replace(' ', '_').lower()}.nlp.json"
        # Load Wikipedia Text from File
        with open(text_filename) as f:
            text = f.read()
            text = text[:1000]
            text = unlp.preprocess_and_clean_text(text)
        # Run Pipeline
        intavia_dict = run_allennlp_pipeline(text)
        # Write to Disk
        json.dump(intavia_dict, open(json_nlp_filename, "w"), indent=2, ensure_ascii=False)
        exit()


if __name__ == "__main__":
    # query_tests = ["William the Silent", "Albrecht Duerer", "Vincent van Gogh", "Constantijn Huygens", "Baruch Spinoza", "Erasmus of Rotterdam",
    #                 "Christiaan Huygens", "Rembrandt van Rijn", "Antoni van Leeuwenhoek", "John von Neumann", "Johan de Witt"]
    # test_english_pipeline_json(["Albrecht Duerer"])

    if len(sys.argv) == 2:
        root_dir = sys.argv[1]
        if os.path.exists(os.path.dirname(root_dir)):
            process_wiki_files(root_dir)
    else:
        print("USAGE: python en_wikipedia_to_json.py <root_folder_with_biographies>")

    