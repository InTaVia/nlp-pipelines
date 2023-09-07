

import requests
import sys, os
import pandas as pd
from typing import Dict, Any


def get_wikidata_id_from_wikipedia_url(wiki_url: str):
    url = 'https://query.wikidata.org/sparql'
    query = f"""
    SELECT ?wikidataID ?image
    WHERE {{
    <{wiki_url}> schema:about ?wikidataID .
    OPTIONAL {{?wikidataID wdt:P18 ?image}}
    }}
    """
    wikidata_id = None
    main_image = None
    # Call API
    try:
        r = requests.get(url, params={'format': 'json', 'query': query}, timeout=3)
        data = r.json() if r.status_code == 200 else None
    except:
        print("Failed to query Wikidata")
        data = None
    
    if data:
        # Feed data from Wikidata Response
        for item in data["results"]["bindings"]:
            wikidata_id = item["wikidataID"]["value"]
            main_image = item.get("image", {}).get("value", {})
            if main_image == {}: main_image = None
    return wikidata_id, main_image


def get_wikidata_basic_info(wikipedia_url: str) -> Dict[str, Any]:
    url = 'https://query.wikidata.org/sparql'
    query = f"""
    SELECT ?wikidataID ?image ?birthDate ?deathDate
    WHERE {{
    <{wikipedia_url}> schema:about ?wikidataID .
    OPTIONAL {{?wikidataID wdt:P18 ?image}}
    OPTIONAL {{?wikidataID wdt:P569 ?birthDate}} #To show property options in the sparql endpoint -> [fn]+control+space
    OPTIONAL {{?wikidataID wdt:P570 ?deathDate}}
    }}
    """
    wikidata_id = None
    main_image, birth_date, death_date = None, None, None
    # Call API
    try:
        r = requests.get(url, params={'format': 'json', 'query': query}, timeout=3)
        data = r.json() if r.status_code == 200 else None
    except:
        print("Failed to query Wikidata")
        data = None
    
    if data:
        # Feed data from Wikidata Response
        for item in data["results"]["bindings"]:
            wikidata_id = item["wikidataID"]["value"]
            main_image = item.get("image", {}).get("value", {})
            if main_image == {}: main_image = None
            if not birth_date:
                birth_date = item.get("birthDate", {}).get("value", {})
            if not death_date:
                death_date = item.get("deathDate", {}).get("value", {})
            if birth_date == {}: birth_date = None
            if death_date == {}: death_date = None
        return {
            "wikidata_id": wikidata_id,
            "main_image": main_image,
            "birth_date": birth_date,
            "death_date": death_date
        }
    else:
        return None

def get_wiki_persons_from_movement(movement_id: str):
    # This query works for literary, artistic, scientific or philosophical movement or scene associated with this person or work. 
    # For political ideologies use wdt:P1142 instead of wdt:P135.
    url = 'https://query.wikidata.org/sparql'

    query = f"""
        SELECT ?person ?personLabel ?edunormal ?edunormalLabel ?student_of ?student_ofLabel ?teacher_of ?teacher_ofLabel ?place_of_birth ?place_of_birthLabel ?place_of_death ?place_of_deathLabel ?date_of_birth ?date_of_death ?edustart ?eduend ?wikipediapage
        WHERE
        {{
        ?person wdt:P135 {movement_id}. # wdt:P135 == 'movement' AND movement_id is e.g. wd:Q34636 'Art Nouveau'
        ?person wdt:P31 wd:Q5.
        OPTIONAL {{?person wdt:P569 ?date_of_birth }} # direct triple
        OPTIONAL {{ ?person wdt:P20 ?place_of_death }}
        OPTIONAL {{?person wdt:P69 ?edunormal }}
        OPTIONAL {{?person wdt:P69 ?edu }}
        OPTIONAL {{ ?person p:P69/pq:P580 ?edustart }}
        OPTIONAL {{ ?person p:P69/pq:P582 ?eduend }}
        OPTIONAL {{ ?person wdt:P1066 ?student_of }}
        OPTIONAL {{ ?person wdt:P802 ?teacher_of }}
        OPTIONAL {{ ?person wdt:P569 ?date_of_birth }}
        OPTIONAL {{ ?person wdt:P570 ?date_of_death }}
        OPTIONAL {{
        ?wikipediapage schema:about ?person.
        ?wikipediapage schema:inLanguage "en" .
        ?wikipediapage schema:isPartOf <https://en.wikipedia.org/> }}
        SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }}
        }}
    """

    # Call API
    try:
        r = requests.get(url, params={'format': 'json', 'query': query}, timeout=3)
        data = r.json() if r.status_code == 200 else None
    except:
        print("Failed to query Wikidata")
        data = None

    tabular_data_all = []
    person_dict = {}
    if data:
        # Feed data from Wikidata Response
        for item in data["results"]["bindings"]:
            person_id = item["person"]["value"]
            try:
                date_birth = item['date_of_birth']['value'].split("T")[0]
            except:
                date_birth = None
            try:
                date_death = item['date_of_death']['value'].split("T")[0]
            except:
                date_death = None
            wiki_page = item.get("wikipediapage", {}).get("value")
            if wiki_page and "en.wikipedia" not in wiki_page: wiki_page = None
            row = {
                    "person": item["personLabel"]["value"],
                    "date_of_birth": date_birth,
                    "date_of_death": date_death,
                    "place_of_birth": item.get("place_of_birthLabel", {}).get("value"),
                    "place_of_death": item.get("place_of_deathLabel", {}).get("value"),
                    "educated_at": item.get("edunormalLabel", {}).get("value"),
                    "edu_start": item.get("edustart", {}).get("value"),
                    "edu_end": item.get("eduend", {}).get("value"),
                    "student_of": item.get("student_ofLabel", {}).get("value"),
                    "teacher_of": item.get("teacher_ofLabel", {}).get("value"),
                    "wiki_page": wiki_page
                }
            # Person_Dict only keeps the FIRST occurrence of a person ID since we only need their Wikipedia Id anyway
            if person_id not in person_dict and wiki_page:
                person_dict[person_id] = {
                    "person": item["personLabel"]["value"],
                    "date_of_birth": date_birth,
                    "date_of_death": date_death,
                    "wiki_page": wiki_page
                }

            tabular_data_all.append(row)
    
    return tabular_data_all, person_dict.values()


def get_all_instaces_of_category(category_id: str):
    url = 'https://query.wikidata.org/sparql'

    query = f"""
        SELECT ?cat ?catLabel (count (*) as ?count) {{
        ?cat wdt:P31 {category_id} .
        SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }}
        }} GROUP BY ?cat ?catLabel ORDER BY desc(?count)
    """

    # Call API
    try:
        r = requests.get(url, params={'format': 'json', 'query': query}, timeout=3)
        data = r.json() if r.status_code == 200 else None
    except:
        print("Failed to query Wikidata")
        data = None

    tabular_data_all = []
    if data:
        # Feed data from Wikidata Response
        for item in data["results"]["bindings"]:
            row = {
                    "category_name": item["catLabel"]["value"],
                    "category_id": item["cat"]["value"]   
                }
            tabular_data_all.append(row)
    
    return tabular_data_all


if __name__ == "__main__":
    category_id = "wd:Q968159" # Art movement
    art_movements = get_all_instaces_of_category(category_id)
    df = pd.DataFrame(art_movements)
    print(df.head())
    df.to_csv(f"english/resources/wikidata_art_movements.csv", index=False)
    
    movement_id="wd:Q34636"
    data_all, data_essential = get_wiki_persons_from_movement(movement_id)
    df = pd.DataFrame(data_essential)
    print(df.head())
    df.to_csv(f"english/resources/wikidata_{movement_id}_results.csv", index=False, sep="|")



