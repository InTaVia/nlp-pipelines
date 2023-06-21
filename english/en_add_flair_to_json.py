"""
    This script already assumes that the Wikipedia Texts are present in the HardDrive

    EXAMPLE:
        python english/en_add_flair_to_json.py "english/data/wikipedia/top_women/"

"""
import glob, os, sys
from utils.nlp_common import create_nlp_template, add_morphosyntax, run_spacy, preprocess_and_clean_text
import spacy, json
from spacy import __version__ as spacy_version
# Load Flair Libraries
from flair import __version__ as flair_version
from flair.data import Sentence
from flair.splitter import SegtokSentenceSplitter

from utils.nlp_flair import run_flair, add_morphosyntax_flair


def main(root_dir: str):
    for person_file in glob.glob(f"{root_dir}/*.nlp.json"):
        person_name = os.path.basename(person_file)[:-9]
        text_filename = person_file
        json_nlp_filename = f"{root_dir}/{person_name.replace(' ', '_').lower()}.nlp_with_flair.json"
        process_person(text_filename, json_nlp_filename)
        exit()


def process_person(nlp_filename: str, json_nlp_filename: str):
    # NLP Structured Data
    with open(nlp_filename) as f:
        intavia_dict = json.load(open(nlp_filename))
    
    nlp_dict = intavia_dict["data"]
    text = nlp_dict["text"]

    splitter = SegtokSentenceSplitter()
    flair_models = {
        "chunker": "chunk",
        "ner": 'ner-ontonotes-large',
        "relations": "relations",
        "frames": "frame",
        "linker": "linker"
    }

    morpho, tokenized_doc = add_morphosyntax_flair(text, splitter)
    nlp_dict['tokenization'][f"flair_{flair_version}"] = tokenized_doc
    nlp_dict['morphology'][f"flair_{flair_version}"] = morpho
    
    # Sentence Splitting According to Flair
    # sentences = splitter.split(text)
    
    # Sentence Splitting from SpaCy (spacy vs flair tokenization might still differ)
    sentences = []
    for s in nlp_dict["morphology"]["spacy"]:
        sentences.append(Sentence(s['text']))

    if 'entities' not in nlp_dict: nlp_dict['entities'] = []
    if 'relations' not in nlp_dict: nlp_dict['relations'] = []

    ent_rel_out = run_flair(sentences, "relations", flair_models)
    nlp_dict['entities'] += ent_rel_out["tagged_entities"]
    nlp_dict['relations'] = ent_rel_out["tagged_relations"]

    frames = run_flair(sentences, "frames", flair_models)
    nlp_dict['frames'] = frames

    # Must restart the sentence to erase previous tags
    sentences = splitter.split(text)
    if 'linked_entities' not in nlp_dict: nlp_dict['linked_entities'] = []
    nlp_dict['linked_entities'] += run_flair(sentences, "linker", flair_models, metadata={"entity_ids":ent_rel_out["entity_ids"]})["tagged_entities"]

    intavia_dict = {
            'status': '200',
            'data': nlp_dict
        }

    json.dump(intavia_dict, open(json_nlp_filename, "w"), indent=2, ensure_ascii=False)

if __name__ == "__main__":

    if len(sys.argv) == 2:
        root_dir = sys.argv[1]
        if os.path.exists(os.path.dirname(root_dir)):
            main(root_dir)
    else:
        print("USAGE: python en_wikipedia_to_json.py <root_folder_with_biographies>")
