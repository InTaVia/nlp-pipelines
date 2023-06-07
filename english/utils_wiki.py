import wikipedia # https://pypi.org/project/wikipedia/
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
import Levenshtein as lev
import re, json

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


def get_wikipedia_article(query_str: str, query_restrictions: Dict[str, Any] = {}) -> wikipedia.WikipediaPage:
    """Get a simple query and return ONE non-ambiugous wikipedia article.
    TODO: The challenge is --> How to create a robust search?
    Args:
        query_str (str): The string to query the Wikipedia API for article title suggestions
        query_restrictions: (Dict[str, Any]): Propertied that bound the search beyond the query string (e.g. birth date)
    Returns:
        wiki_article (str): The content of the wikipedia article
    """

    def _get_year_from_cat(categories: List[str], cat_type: str) -> int or None:
        valid_cats = [c for c in categories if cat_type in c]
        if len(valid_cats) > 0:
            matches = re.findall(r"\d{4}|\d{3}", valid_cats[0])
            if len(matches) > 0:
                valid_year = int(matches[0])
                return valid_year
        else:
            return None
        
    other_names = query_restrictions.get("other_names", [])
    birth_year = query_restrictions.get("birth_year", -1)
    death_year = query_restrictions.get("death_year", -1)

    # Return a list of terms that match our (usually and unintentionally) Fuzzy Term!
    page_names = wikipedia.search(query_str, results=3, suggestion=False)
    for name in other_names:
        if page_names:
            page_names += wikipedia.search(name, results=3, suggestion=False)
        else:
            page_names = wikipedia.search(name, results=3, suggestion=False)
    print(f"Options: {set(page_names)}")

    # Execute the best ranked queries and match with birth year
    page = None
    valid_candidates = []

    original_query_names = [query_str] + query_restrictions.get("other_names", []) 
    ranked_articles = rank_article_names(page_names, original_query_names, query_restrictions)

    for article in ranked_articles:
        page_name = article.wikipage_title
        # Now that we have a valid page name, we retrieve the Page Object
        try:
            print(f"\nRetrieving page for {page_name}")
            page = wikipedia.page(page_name, pageid=None, auto_suggest=False, redirect=True, preload=False)
        except:
            continue
        wiki_birth_year = _get_year_from_cat(page.categories, cat_type="births")
        wiki_death_year = _get_year_from_cat(page.categories, cat_type="deaths")
        
        if not wiki_birth_year: continue # We assume that the page is not a biography if it does not have a "XXX births" associated with it

        print(f"Wiki Life Data = ({wiki_birth_year} - {wiki_death_year})")
        
        if birth_year > 0:
            if birth_year == wiki_birth_year:
                if death_year > 0:
                    if death_year == wiki_death_year:
                        print(f"Page Chosen! Confidence Score = 3")
                        valid_candidates.append((page, 3)) # This is the best case in which everything matched perfectly!
                    else:
                        print(f"Page Chosen! Confidence Score = 2 (death)")
                        valid_candidates.append((page, 2)) # Well, the death did not match but the birth did so we add it
                else:
                    print(f"Page Chosen! Confidence Score = 2")
                    valid_candidates.append((page, 2)) # In this case we matched the birth_year and death is UNK by us so we trust the birth is enough to diambiaguate ...
            else:
                continue
        else: # This case is for UNK birth year in which case we only know the names. So we return immediately from the function with the highest ranked article!
            print(f"Page Chosen! Confidence Score = 1")
            # valid_candidates.append((page, 1)) # More appropriate would be to keep all the scores and then order the valid candidates by: (DatesConfidence, Levensthein, TokenOverlap)
            return page
    # Now return the best ...
    if len(valid_candidates) > 0:
        best_page, confidece_score = sorted(valid_candidates, key=lambda x: x[1])[-1]
        if confidece_score == 1: # We did not find anyone with much certainty so we return the highest ranked according to token_overlap and levenshtein
            return best_page
        else:
            return best_page
    else:
        return None


def save_wikipedia_page(page: wikipedia.WikipediaPage, output_path:str, include_metadata: bool = False):
    # Save JSON File
    if include_metadata:
        meta_path = f"{output_path}.meta.json"
        section_dict = {}
        for sec_title in page.sections:
            section_dict[sec_title] = page.section(sec_title)
        metadata = {
            "title": page.title,
            "url": page.url,
            "original_title": page.original_title,
            "text": page.content,
            "summary": page.summary,
            "sections": section_dict,
            "categories": page.categories,
            "links": page.links,
            "references": page.references,
            "images": page.images
        }
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
    # Save Plain Text
    with open(output_path, "w") as f:
        f.write(page.content)
