
import re, os
import spacy, json
from typing import Dict, Any, List, Tuple
from spacy_to_naf.converter import Converter
from allennlp.predictors import Predictor

from dataclasses import dataclass
from nafparserpy.layers.naf_header import LPDependency

from nafparserpy.parser import NafParser
from nafparserpy.layers.text import Wf
from nafparserpy.layers.elements import Span
from nafparserpy.layers.srl import Predicate, Role
from nafparserpy.layers.time_expressions import TimeExpressions, Timex3

from python_heideltime import Heideltime

from utils_wiki import get_wikipedia_article
import utils_nlp as unlp


def main():
    query_tests = ["William the Silent", "Albercht Durer", "Vincent van Gogh", "Constantijn Huygens", "Baruch Spinoza", "Erasmus of Rotterdam",
                    "Christiaan Huygens", "Rembrandt van Rijn", "Antoni van Leeuwenhoek", "John von Neumann", "Johan de Witt"]
    test_english_pipeline_naf(query_tests)


#### BEGIN English Pipeline

def get_naf_sentences(naf: NafParser) -> List[List[Wf]]:
    naf_tokens = naf.get('text')
    sentences, curr_sent = [], []
    shifted_tokens = naf_tokens[1:] + [naf_tokens[-1]]
    for token, next_token in zip(naf_tokens, shifted_tokens):
        curr_sent.append(token)
        if token.sent != next_token.sent:
            sentences.append(curr_sent)
            curr_sent = []
    
    if len(curr_sent) > 0:
        sentences.append(curr_sent)
    
    return sentences


def add_naf_srl_layer(naf: NafParser, srl_predictor: Predictor) -> NafParser:
    naf_sentences = get_naf_sentences(naf)
    srl_doc_info = []
    # Send to AllenNLP Sentence by Sentence to obtain Sentence-wise Predicate-Argument Structures
    for i, sent in enumerate(naf_sentences):
        print(f"===== Sentence {i+1} ======")
        sent_tokens = [wf.text for wf in sent]
        print("NAF TOKENS:", sent_tokens)
        srl_results = unlp.allennlp_srl(" ".join(sent_tokens), srl_predictor)
        allennlp_tokens = srl_results.tokens
        assert len(allennlp_tokens) == len(sent_tokens), f"\nToken Length Mismatch!\nAllenNLP = {allennlp_tokens}\nSpaCy = {sent_tokens}\n"
        srl_doc_info.append(srl_results)

    naf.add_linguistic_processor('srl', "AllenNLP", "2.9.0", lpDependencies=[LPDependency("structured-prediction-srl-bert", "2020.12.15.tar.gz")])

    doc_predicate_counter = 0
    document_offset = 0
    all_predicates = []
    for i, sent_srl_output in enumerate(srl_doc_info):
        for pred_ix, arguments in sent_srl_output.pred_arg_struct.items():
            doc_predicate_counter += 1
            naf_roles = []
            print(f"------ SENT {i} PRED {doc_predicate_counter} OFFSET {document_offset} ------")
            for j, arg in enumerate(arguments):
                print(arg.predicate, arg.label, arg.text)
                naf_role = Role(arg.label, Span.create([f"t{k+1}" for k in range(document_offset+arg.start, document_offset+arg.end+1)]))
                naf_roles.append(naf_role)
            naf_predicate = Predicate(sent_srl_output.tokens[pred_ix], 
                                        span=Span.create([f"t{document_offset+pred_ix+1}"]), 
                                        roles=naf_roles,
                                        )
            all_predicates.append(naf_predicate)
        document_offset += len(sent_srl_output.tokens)
    
    naf.add_layer_from_elements("srl", elements=all_predicates)

    return naf


def add_naf_timexp_layer(naf: NafParser, heideltime_parser: Heideltime) -> NafParser:
    naf_sentences = get_naf_sentences(naf)
    token_ix = 0
    init_spans2token, end_spans2token = {}, {}
    for sent in naf_sentences:
        for wf in sent:
            init_spans2token[int(wf.offset)] = token_ix
            end_spans2token[int(wf.offset) + int(wf.length)] = token_ix
            token_ix += 1
    timex_info = unlp.add_json_heideltime(naf.get('raw').text, heideltime_parser)
    all_timex = []
    for timex_obj in timex_info:
        tokStart = init_spans2token.get(timex_obj['locationStart'])
        tokEnd = end_spans2token.get(timex_obj['locationEnd'])
        if tokStart and tokEnd:
            all_timex.append(Timex3(
                timex_obj['ID'].replace('t', 'x'),
                timex_obj['category'],
                Span.create([f"t{k+1}" for k in range(tokStart, tokEnd+1)])
            ))
    
    naf.add_layer_from_elements("timeExpressions", all_timex)
    return naf



def naf_to_file(naf: NafParser, filepath: str, filename: str) -> NafParser:
    if not filename.endswith(".naf"): filename = filename + ".naf"
    naf.write(f"{filepath}/{filename}")
    return naf



#### END English Pipeline


def test_english_pipeline_naf(query_tests: str = ["William the Silent"], naf_path:str = "./english/data/naf"):

    naf_converter = Converter("en_core_web_sm", add_terms=True, add_deps=True, add_entities=True, add_chunks=True)
    srl_predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz")
    ner_predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/ner-elmo.2021-02-12.tar.gz")

    heideltime_parser = Heideltime()
    heideltime_parser.set_language('ENGLISH')
    heideltime_parser.set_document_type('NARRATIVES')

    wiki_matches = 0

    for query in query_tests:
        print(f"\n------------------\nStarting query for: {query}")
        # Load Text(s) to process...
        wiki_page = get_wikipedia_article(query)
        if wiki_page:
            print(f"Found a Page: {wiki_page.title}")
            text = wiki_page.summary
            wiki_matches += 1
            
            # Apply different patterns to 'clean' the text
            clean_text = unlp.preprocess_and_clean_text(text)
            
            # Process the text using SpaCy to NAF module
            naf_name = query.lower().replace(" ", "_") + ".naf"
            naf = unlp.create_naf_object(clean_text, naf_name, naf_converter)
            
            # Add AllenNLP SRL Layer
            naf = add_naf_srl_layer(naf, srl_predictor)

            # Add Heideltime Layer
            timexps = add_naf_timexp_layer(naf, heideltime_parser)

            # Write to Disk
            if not os.path.exists(naf_path): os.mkdir(naf_path)
            naf_to_file(naf, naf_path, naf_name)
            
        else:
            print(f"Couldn't find {query}!")

    print(f"Found {wiki_matches} out of {len(query_tests)} articles in Wikipedia ({wiki_matches/len(query_tests) * 100:.2f}%)")




if __name__ == "__main__":
    main()