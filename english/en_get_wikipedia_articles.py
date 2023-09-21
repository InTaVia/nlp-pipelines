import os, argparse
from utils.utils_wiki import get_wikipedia_article, save_wikipedia_page
from utils.nlp_common import preprocess_and_clean_text


def get_article_from_name(person_name: str):
    wikipedia_title = person_name

    if not os.path.exists("data/wikipedia"): os.makedirs("data/wikipedia")

    wiki_page = get_wikipedia_article(person_name)
    if wiki_page:
        print(f"Found a Page: {wiki_page.title}")
        text = wiki_page.content
        wikipedia_title = wiki_page.title
        text_filename = f"data/wikipedia/{wikipedia_title.replace(' ', '_').lower()}.txt"
        save_wikipedia_page(wiki_page, text_filename, include_metadata=True, include_sections=True)
        print(f"Saved {text_filename}")
    else:
        print(f"Query Failed! Couldn't find {person_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # GENERAL SYSTEM PARAMS
    parser.add_argument('-n', '--name', help='A string with a full name', default=None)
    parser.add_argument('-l', '--list_names_file', help='A structured TXT containing a list of names', default=None)

    args = parser.parse_args()

    if args.name:
        get_article_from_name(args.name)
    
    if args.list_names_file:
        pass # Open, read file and foor loop with get_article() for each name found ...