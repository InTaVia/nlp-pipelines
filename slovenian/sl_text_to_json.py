import os, json
import classla
from utils_wiki import get_wikipedia_article
import utils_nlp as unlp



def test_slovenian_pipeline_json(query_tests: str, json_path: str = "./slovenian/data/json/"):
    nlp = classla.Pipeline(lang="sl", processors="tokenize,lemma,pos,depparse,ner")


    wiki_matches = 0
    for query in query_tests:
        doc_title = query
        wiki_page = get_wikipedia_article(doc_title, language="sl")
        if wiki_page:
            print(f"Found a Page: {wiki_page.title}")
            nlp_dict = {}
            text = wiki_page.summary
            wiki_matches += 1
            json_name = doc_title.lower().replace(" ", "_") + ".json"
            # Step 1: Clean Text
            clean_text = unlp.preprocess_and_clean_text(text)
            # Step 2: Basic Processing with CLASSLA
            classla_dict = unlp.run_classla(clean_text, nlp)
            # Step N: Build General Dict
            nlp_dict['input_text'] = clean_text
            nlp_dict['token_objs'] = classla_dict['token_objs']
            nlp_dict['entities'] = classla_dict['entities']
            response = unlp.nlp_to_dict(nlp_dict)
            # Write to Disk
            if not os.path.exists(json_path): os.mkdir(json_path)
            json.dump(response, open(f"{json_path}/{json_name}", "w"), indent=2)
        else:
            print(f"Couldn't find {query}!")

    print(f"Found {wiki_matches} out of {len(query_tests)} articles in Wikipedia ({wiki_matches/len(query_tests) * 100:.2f}%)")


if __name__ == "__main__":
    query_tests = ["Slavoj Žižek"]
    test_slovenian_pipeline_json(query_tests)