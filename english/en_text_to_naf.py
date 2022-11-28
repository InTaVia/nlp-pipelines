
import re, os

from typing import Dict, Any, List, Tuple
from spacy_to_naf.converter import Converter
from allennlp.predictors import Predictor
from dataclasses import dataclass
from nafparserpy.layers.naf_header import LPDependency

from nafparserpy.parser import NafParser
from nafparserpy.layers.text import Wf
from nafparserpy.layers.elements import Span
from nafparserpy.layers.srl import Predicate, Role

from utils_wiki import get_wikipedia_article


def main():
    query_tests = ["William the Silent", "Albercht Durer", "Albert Einstein", "Octavio Paz", "Barack Obama", "Donald Trump"]
    test_english_pipeline(query_tests)


#### BEGIN English Pipeline
@dataclass
class SRL_Argument:
    predicate: Tuple[int, str] # (token_index, surface_form) in the text
    text: str # argument text
    label: str # argument label
    start: int # token index where the argument starts
    end: int # token index where the argument ends


@dataclass
class SRL_Output:
    tokens: List[str]
    predicates: List[Tuple[int, str]] # (token_index, surface_form) in the text
    arg_labels: List[List[str]] # each internal list has Bio Labels corresponding to the predicates by position in the list
    pred_arg_struct: Dict[int, SRL_Argument]


def preprocess_and_clean_text(text: str) -> str:
    clean_text = re.sub(r'[\r\n]+', " ", text)
    clean_text = re.sub(r'"', ' " ', clean_text)
    clean_text = re.sub(r'[\s]+', " ", clean_text)
    return clean_text


def create_naf_object(text: str, naf_name: str, naf_converter: Converter) -> NafParser:
    """Reads in a Text, uses SpaCy as an NLP Pipeline and returns the annotations on NAF Format

    Args:
        text (str): The text to be parsed and strcuctured. Annotations added: 

    Returns:
        NafParser: The NAF object containing the SpaCy English annotations
    """
    naf_name = naf_name.lower().replace(" ", "_") + ".naf"
    naf = naf_converter.process_text(text, naf_name, out_path=None)
    return naf
    

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
        srl_results = allennlp_srl(" ".join(sent_tokens), srl_predictor)
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


def allennlp_srl(text: str, srl_predictor: Predictor) -> SRL_Output:
    output = srl_predictor.predict(text)
    print("ALLEN TOKENS:", output['words'])

    simplified_output = SRL_Output(output['words'], predicates=[], arg_labels=[], pred_arg_struct={})

    for verb_obj in output['verbs']:
        # print(f"\tVERB: {verb_obj['verb']} | ARGS: {verb_obj['tags']}")
        predicate_index, predicate_arguments = 0, []
        arg_tokens, arg_indices, arg_label = [], [], ""
        for ix, bio_tag in enumerate(verb_obj['tags']):
            if bio_tag == "B-V":
                predicate_index = ix
                if len(arg_label) > 0:
                    predicate_arguments.append(SRL_Argument((predicate_index, verb_obj['verb']), " ".join(arg_tokens), arg_label, start=arg_indices[0], end=arg_indices[-1]))
                    arg_label = ""
                    arg_indices = []
                    arg_tokens = []
            elif bio_tag.startswith("B-"):
                if len(arg_label) > 0:
                    predicate_arguments.append(SRL_Argument((predicate_index, verb_obj['verb']), " ".join(arg_tokens), arg_label, start=arg_indices[0], end=arg_indices[-1]))
                    arg_indices = []
                    arg_tokens = []
                arg_label = bio_tag[2:]
                arg_tokens.append(output['words'][ix])
                arg_indices.append(ix)
            elif bio_tag.startswith("I-"):
                arg_tokens.append(output['words'][ix])
                arg_indices.append(ix)
            elif bio_tag == "O" and len(arg_label) > 0:
                predicate_arguments.append(SRL_Argument((predicate_index, verb_obj['verb']), " ".join(arg_tokens), arg_label, start=arg_indices[0], end=arg_indices[-1]))
                arg_label = ""
                arg_indices = []
                arg_tokens = []

        simplified_output.predicates.append((predicate_index, verb_obj['verb']))
        simplified_output.arg_labels.append(verb_obj['tags'])
        simplified_output.pred_arg_struct[predicate_index] = predicate_arguments
        
    return simplified_output


def naf_to_file(naf: NafParser, filepath: str, filename: str) -> NafParser:
    if not filename.endswith(".naf"): filename = filename + ".naf"
    naf.write(f"{filepath}/{filename}")
    return naf

#### END English Pipeline


def test_english_pipeline(query_tests: str = ["William the Silent"]):

    naf_converter = Converter("en_core_web_sm", add_terms=True, add_deps=True, add_entities=True, add_chunks=True)
    srl_predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz")

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
            clean_text = preprocess_and_clean_text(text)
            
            # Process the text using SpaCy to NAF module
            naf_name = query.lower().replace(" ", "_") + ".naf"
            naf = create_naf_object(clean_text, naf_name, naf_converter)
            
            # Add AllenNLP SRL Layer
            naf = add_naf_srl_layer(naf, srl_predictor)

            # Write to Disk
            naf_path = "english/data/naf"
            if not os.path.exists(naf_path): os.mkdir(naf_path)
            naf_to_file(naf, naf_path, naf_name)
            
        else:
            print(f"Couldn't find it!")

    print(f"Found {wiki_matches} out of {len(query_tests)} articles in Wikipedia ({wiki_matches/len(query_tests) * 100:.2f}%)")



if __name__ == "__main__":
    main()