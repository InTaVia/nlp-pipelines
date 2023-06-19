import wikipedia # https://pypi.org/project/wikipedia/
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
import Levenshtein as lev
import re, json
from collections import OrderedDict

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


def get_raw_wikipedia_article(wiki_title: str) -> Dict[str, Any]:
    pass


def save_wikipedia_page(page: wikipedia.WikipediaPage, output_path:str, include_metadata: bool = False, section_dict: Dict = None):
    # Save JSON File
    if include_metadata:
        meta_path = f"{output_path}.meta.json"
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


def extract_sections(page_text: str) -> Dict[str, str]:
    section_matches = re.finditer(r"=+(\s.+\s)=+", page_text)
    section_ends = []
    section_starts = [0]
    section_titles = [(1, "Summary")]
    for title_match in section_matches:
        tmp = page_text[title_match.start():title_match.end()]
        sec_level = tmp.count("=")//2 - 1
        title = tmp.replace("=", "").strip()
        section_ends.append(title_match.start())
        section_starts.append(title_match.end() + 1)
        section_titles.append((sec_level, title))

    section_dict = {}
    ix = 1
    for (level, title), sec_st, sec_end in zip(section_titles, section_starts, section_ends):
        section_text = page_text[sec_st:sec_end]
        section_dict[title] = {"index": ix, "level": level, "content": section_text.strip()}
        ix += 1
    return section_dict


def extract_infobox(raw_text:str) -> Dict[str, str]:
    # Find the InfoBox in the RAW Wikipedia Page
    pattern = r"\{\{Infobox\s[\w\s]+\n(?:\|[\w\s]+\s+=\s*.*\n)*\}\}"
    # Extract the infobox using regex
    match = re.search(pattern, raw_text, re.DOTALL)
    # Check if a match is found
    if match:
        infobox_str = match.group()
    else:
        return None
    # Transfer String into a Python Dict (TODO: Create a clean dict with only useful fields?)
    infobox_dict = {}
    key_value_pattern = r"\|([\w\s]+)\s*=\s*(.*)"
    for line in infobox_str.split("\n"):
        # Check if the line matches the key-value pattern
        match = re.match(key_value_pattern, line)
        if match:
            key = match.group(1).strip()
            value = match.group(2).strip()
            infobox_dict[key] = value
    # Return Dict
    return infobox_dict


def _get_wiki_link_details(bracketed_string: str) -> Dict[str, str]:
    # Ignore Files and Category Tags
    if bracketed_string.startswith("[[File:"):
        return {}
    elif bracketed_string.startswith("[[Category:"):
        return {}
    # For normal [[Concept]]
    if "|" in bracketed_string[:-1]:
        tmp = bracketed_string.split("|")
        title = tmp[0][2:]
        surface_form = " ".join(tmp[1:])[:-3]
    else:
        title = bracketed_string[2:-3]
        surface_form = title
    
    title = title.replace('<br/>', ' ')
    if 's' == bracketed_string[-1]:
        surface_form = surface_form + 's'

    return {"title": title, "surfaceForm": surface_form, "wikiLink": f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"}


def get_wiki_linked_entities(raw_text: str) -> Dict[str, str]:
    wiki_links_dict = OrderedDict() # {'surfaceForm': 'wikipediaTitle'} --> 'wikipediaLink'? 'wikidataID'?
    for match in re.finditer(r"\[\[.+?\]\]", raw_text):
        text = raw_text[match.start():match.end()+1]
        wiki_info = _get_wiki_link_details(text)
        # print(f"{text}\n{wiki_info}\n-----")
        if len(wiki_info) > 0:
            wiki_links_dict[wiki_info['surfaceForm']] = wiki_info['wikiLink']
    return wiki_links_dict
