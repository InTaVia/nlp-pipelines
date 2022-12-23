from flask import Blueprint, request, jsonify
from spacy_to_naf.converter import Converter
from allennlp.predictors import Predictor
import spacy
from lxml import etree
import utils_nlp as unlp

naf_converter = Converter("en_core_web_sm", add_terms=True, add_deps=True, add_entities=True, add_chunks=True)
nlp = spacy.load("en_core_web_sm")
srl_predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz")


en_blueprint = Blueprint('english_api_routes', __name__, url_prefix='/api/en')


@en_blueprint.route('/spacy_nlp', methods=['POST'])
def get_structured_text():
    doc_title = request.form['doc_title']
    text = request.form['text']
    nlp_dict = {}
    json_name = doc_title.lower().replace(" ", "_") + ".json"
    # Step 1: Clean Text
    clean_text = unlp.preprocess_and_clean_text(text)
    # Step 2: Basic Processing with SpaCy
    spacy_dict = unlp.run_spacy(clean_text, nlp)
    # Step 3: Run HeidelTime
    # Step 4: Run AllenNLP SRL
    # Step N: Build General Dict
    nlp_dict['input_text'] = clean_text
    nlp_dict['token_objs'] = spacy_dict['token_objs']
    nlp_dict['entities'] = spacy_dict['entities']
    response = unlp.nlp_to_dict(nlp_dict)
    return jsonify(response)


@en_blueprint.route('/naf_nlp', methods=['POST'])
def get_naf_from_text():
    doc_title = request.form['doc_title']
    text = request.form['text']
    naf_name = doc_title.lower().replace(" ", "_") + ".json"
    # Step 1: Clean Text
    clean_text = unlp.preprocess_and_clean_text(text)
    # Step 2: Create NAF (with SpaCy processor)
    naf = unlp.create_naf_object(clean_text, naf_name, naf_converter)
    # Add AllenNLP SRL Layer
    # naf = unlp.add_naf_srl_layer(naf, srl_predictor)
    tree_bytes = etree.tostring(naf.root, encoding='UTF-8', pretty_print=True, xml_declaration=True)
    tree_str = tree_bytes.decode('utf-8')
    print(type(tree_str))
    return jsonify({
        'naf_content': tree_str
    })