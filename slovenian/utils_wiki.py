import wikipedia # https://pypi.org/project/wikipedia/
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
import Levenshtein as lev
import re

@dataclass
class RankedArticle:
    wikipage_title: str
    queried_name: str
    lev_similarity: float = -1
    token_overlap: float = -1
    dates_confidence: int = -1


def rank_article_names(wiki_page_names: List[str], original_query_names: List[str], query_restrictions: Dict[str, Any]) -> List[RankedArticle]:
    """Disambiaguate the possible Wikipedia Pages to properly match the Original Query Names
    
    Args:
        wiki_page_names (List[str]): _description_
        original_query_names (List[str]): _description_

    Returns:
        str: The Disambiaguated Wikipedia Page name to query a proper article
    """

    def _token_overlap(str1, str2):
        toks1 = set(str1.split())
        toks2 = set(str2.split())
        return len(toks1.intersection(toks2)) / max(len(toks1),len(toks2))

    distance_matrix: List[RankedArticle] = []
    for wp in wiki_page_names:
        for qn in original_query_names:
            tok_verlap_ratio = _token_overlap(wp.lower(), qn.lower())
            lev_similarity = lev.ratio(wp.lower(), qn.lower())
            if tok_verlap_ratio  > 0 and lev_similarity >= 0.45:
                distance_matrix.append(RankedArticle(wp, qn, lev_similarity, tok_verlap_ratio))
    
    if len(distance_matrix) == 0:
        return []
    else:
        distance_matrix_all = sorted(distance_matrix, key= lambda x: (x.token_overlap, x.lev_similarity), reverse=True)
        print(f"Ordered Options Compund Metric: {distance_matrix_all}")

        only_names, ordered_names = [], []
        for x in distance_matrix_all:
            if x.queried_name not in only_names:
                only_names.append(x.wikipage_title)
                ordered_names.append(x)
        return ordered_names


def get_wikipedia_article(query_str: str, language: str, query_restrictions: Dict[str, Any] = {}) -> wikipedia.WikipediaPage:
    """Get a simple query and return ONE non-ambiugous wikipedia article.
    Args:
        query_str (str): The string to query the Wikipedia API for article title suggestions
        query_restrictions: (Dict[str, Any]): Propertied that bound the search beyond the query string (e.g. birth date)
    Returns:
        wiki_article (str): The content of the wikipedia article
    """

    # Return a list of terms that match our (usually and unintentionally) Fuzzy Term!
    wikipedia.set_lang(language)
    page_names = wikipedia.search(query_str, results=3, suggestion=False)
    print(f"Options: {set(page_names)}")

    # Execute the best ranked queries and match with birth year
    ranked_articles = rank_article_names(page_names, [query_str], query_restrictions)
    print(f"Ranked: {ranked_articles}")

    if len(ranked_articles) > 0:
        article = ranked_articles[0]
        page_name = article.wikipage_title
        # Now that we have a valid page name, we retrieve the Page Object
        try:
            print(f"\nRetrieving page for {page_name}")
            page = wikipedia.page(page_name, pageid=None, auto_suggest=False, redirect=True, preload=False)
            return page
        except:
            return None
    