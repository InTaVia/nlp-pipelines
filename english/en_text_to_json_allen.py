"""
    This script already assumes that the Wikipedia Texts are present in the HardDrive

    EXAMPLE:
        python english/en_text_to_json_allen.py --from_text "english/data/wikipedia/top_women/"

        python english/en_text_to_json_allen.py --from_flair_json --path "english/data/wikipedia/top_women/"

"""
import os, glob, sys
import spacy, json
import argparse
from typing import List, Dict, Any
from spacy import __version__ as spacy_version
from allennlp.predictors import Predictor

from utils.utils_wiki import get_wikipedia_article
import utils.nlp_common as unlp
import utils.nlp_allen as anlp

# Initialize Models just ONCE Globally
spacy_model = "en_core_web_lg"
nlp = spacy.load("en_core_web_lg")
srl_predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz")
ner_predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/ner-elmo.2021-02-12.tar.gz")
coref_predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2021.03.10.tar.gz")

# Heideltime Functions
from python_heideltime import Heideltime
from utils.nlp_heideltime import add_json_heideltime
heideltime_parser = Heideltime()
heideltime_parser.set_language('ENGLISH')
heideltime_parser.set_document_type('NARRATIVES')



def run_allennlp_pipeline(text: str, nlp_dict: Dict[str, Any] = {}):
    # Step 1: Clean Text
    clean_text = unlp.preprocess_and_clean_text(text)
    if len(nlp_dict) == 0:
        from_scratch = True
        nlp_dict['input_text'] = clean_text
        # Step 2: Basic Processing with SpaCy
        spacy_dict = unlp.run_spacy(clean_text, nlp)
        token_objs = spacy_dict['token_objs']
        nlp_dict['token_objs'] = token_objs
        nlp_dict['entities'] = spacy_dict['entities']
        # Step 3: Run HeidelTime
        nlp_dict['time_expressions'] = add_json_heideltime(clean_text, heideltime_parser)
        sentences = spacy_dict['sentences']
    else:
        from_scratch = False
        sentences, sent_tokenized, token_objs = [], [], []
        flair_versions = [v for v in nlp_dict['morphology'].keys() if 'flair_' in v]
        if len(flair_versions)>0:
            for sent_ix, sent_obj in enumerate(nlp_dict['morphology'][flair_versions[0]]):
                sentences.append(sent_obj['text'])
                sent_tokenized.append(sent_obj['tokenized'])
                for tok in sent_obj['words']:
                    token_objs.append({'sent_id': sent_ix, 'text': tok['FORM'], 'lemma': tok['FORM'], 
                                       'start_char': tok['MISC']['StartChar'], 'end_char': tok['MISC']['EndChar'], 'space_after': tok['MISC']['SpaceAfter']})
    # Step 4: Run AllenNLP SRL
    nlp_dict['semantic_roles'] = anlp.add_json_srl_allennlp(sentences, srl_predictor, token_objs)
    # Step 5: Run AllenNLP Coref
    nlp_dict['coreference'] = anlp.add_json_coref_allennlp(sentences, coref_predictor, token_objs)
    # Step N: Run AllenNLP Entities
    nlp_dict['entities'] += anlp.add_json_ner_allennlp(sentences, ner_predictor, token_objs)

    if from_scratch:
        response = unlp.nlp_to_dict(nlp_dict, basic_model_name='spacy') # f'spacy_{spacy_model}_{spacy_version}'
    else:
        response = {'status': '200', 'data': nlp_dict}

    return response


def test_english_pipeline_json(query_tests: List[str], output_json_path: str = "./english/data/json/"):
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


def process_wiki_files(root_dir: str, pre_load_json: bool = False):
    ix = 0

    if pre_load_json:
        list_pattern = f"{root_dir}/*.nlp.flair.json"
        extension_index = -15
    else:
        list_pattern = f"{root_dir}/*.txt"
        extension_index = -4

    for person_file in glob.glob(list_pattern):
        ix += 1
        person_name = os.path.basename(person_file)[:extension_index]
        print(ix, person_file, person_name)
        
        json_nlp_filename = f"{root_dir}/{person_name.replace(' ', '_').lower()}.nlp.json"
        if not os.path.exists(json_nlp_filename):
            if pre_load_json:
                nlp_dict = json.load(open(person_file))["data"]
                text = nlp_dict["text"]
            else:
                # Load Wikipedia Text from File
                with open(person_file) as f:
                    text = f.read()
            # Run Pipeline
            intavia_dict = run_allennlp_pipeline(text, nlp_dict)
            # Write to Disk
            json.dump(intavia_dict, open(json_nlp_filename, "w"), indent=2, ensure_ascii=False)
        else:
            print("NLP File already exists. Skipping...")



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # GENERAL SYSTEM PARAMS
    parser.add_argument('-t', '--from_text', help='This mode looks for an TXT files and processed them from scratch', action='store_true')
    parser.add_argument('-f', '--from_flair_json', help='This mode looks for an NLP JSON (produced by Flair) and adds AllenNLP layers to it, otherwise it looks for TXT', action='store_true')
    parser.add_argument('-p', '--path', help='Path to a directory holding the TXT or JSON files that need to be processed', required=True)

    args = parser.parse_args()
    root_dir = args.path
    
    if args.from_flair_json:
        pre_load_json = True
    else:
        pre_load_json = False

    if args.from_flair_json or args.from_text:
        if os.path.exists(os.path.dirname(root_dir)):
            process_wiki_files(root_dir, pre_load_json)
        else:
            print("USAGE: python en_wikipedia_to_json_allen.py <ARGS> (see script)")
    else:
        print("No valid mode provided. Running Test...")
        # query_tests = ["William the Silent", "Albrecht Duerer", "Vincent van Gogh", "Constantijn Huygens", "Baruch Spinoza", "Erasmus of Rotterdam",
        #             "Christiaan Huygens", "Rembrandt van Rijn", "Antoni van Leeuwenhoek", "John von Neumann", "Johan de Witt"]
        test_english_pipeline_json(["Albrecht Duerer"])

    