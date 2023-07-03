import json, os, re, time
import copy
from collections import Counter
from utils.utils_wiki import get_raw_wikipedia_article, get_wiki_linked_entities, get_relevant_items_from_infobox

inverse_relations_dict = {
    "born_in": "place_of_birth",
    "married_to": "married_to",
    "lived_in": "place_of_residence"
}


person_template = {
    "id": "", # duerer-pr-012
    "label": { "default": "" }, #surfaceForm
    "linkedIds": [],
    "media": [],
    "relations": [],
    "kind": "person",
    # "gender": { } # "id": "male", "label": { "default": "male" }
}

group_template = {
    "id": "", # duerer-gr-001
    "label": { "default": "" }, # surfaceForm
    "description": "",
    "source": { "citation": "Wikipedia"},
    "relations": [],
    "kind": "group",
    "type": {} # "id": "group-type-workspace", "label": { "default": "workspace" }
}

place_template = {
    "id": "", # "duerer-pl-001"
    "label": { "default": "" }, # surfaceForm
    "relations": [],
    "kind": "place",
    "type": { }, # { "id": "place-type-city", "label": { "default": "city" } }
    "geometry": { "type": "Point", "coordinates": None } # { "type": "Point", "coordinates": [8.34915, 49.69025] }
}

object_template = {
    "id": "", # duerer-ob-001
    "label": { "default": "" }, # surfaceForm
    "description": "",
    "source": { "citation": "Wikipedia" },
    "relations": [],
    "kind": "cultural-heritage-object",
    "type": {} # "id": "cultural-heritage-object-type-book", "label": { "default": "book" }
}


def convert_nlp_to_idm_json(nlp_path: str, idm_out_path: str):

    parent_idm = {
        "entities": [],
        "events": [],
        "media": [],
        "biographies": [],
        "vocabularies": [],
        "unmappedEntities": [],
        "collections": {}
    }
    
    nlp_dict = json.load(open(nlp_path))
    person_name = os.path.basename(nlp_path).split(".")[0]
    firstname = person_name.split("_")[0]
    lastname = person_name.split("_")[-1]
    
    # Open Raw File and Meta JSON to Complement Data
    wiki_raw = get_raw_wikipedia_article(wiki_title=person_name.replace("_", " ").title())
    # wiki_meta = json.load(open(f"english/data/wikipedia/{person_name}.txt.meta.json"))
    wiki_linked_dict = get_wiki_linked_entities(wiki_raw) # {'surfaceForm': 'wiki_link'}


    universal_dict = {} # {(locStart, locEnd): {prop: val, prop: val, ...}}
    entity_dict = {}
    # Add Entities
    for ent_obj in nlp_dict["data"]["entities"]:
        key = f"{ent_obj['locationStart']}_{ent_obj['locationEnd']}"
        if key in universal_dict:
            universal_dict[key]["ner"].append(ent_obj["category"])
        else:
            universal_dict[key] = {"nlp_id":ent_obj["ID"], "sent_id": ent_obj["sentenceID"], "locationStart": ent_obj["locationStart"], "locationEnd": ent_obj["locationEnd"],
                                    "surfaceForm": ent_obj["surfaceForm"], "ner": [ent_obj["category"]], "relations": [], "cluster_id": -1}
        entity_dict[ent_obj["ID"]] = ent_obj
    # Add Relations
    relation_dict = {}
    for relation in nlp_dict["data"].get("relations", []):
        subj_id = relation['subjectID']
        obj_id = relation['objectID']
        rel_subj = entity_dict.get(subj_id)
        if rel_subj:
            key = f"{rel_subj['locationStart']}_{rel_subj['locationEnd']}"
            universal_dict[key]["relations"].append(relation)
        rel_obj = entity_dict.get(obj_id)
        if rel_obj:
            key = f"{rel_obj['locationStart']}_{rel_obj['locationEnd']}"
            universal_dict[key]["relations"].append(relation) 
        relation_dict[relation["relationID"]] = relation
    # Add NLP NEL Links
    for link_ent in nlp_dict["data"].get("linked_entities", []):
        key = f"{link_ent['locationStart']}_{link_ent['locationEnd']}"
        if key in universal_dict:
            universal_dict[key]["wiki_link"] = link_ent["wikiURL"]
        else:
            universal_dict[key] = {"wiki_link": link_ent["wikiURL"]}
    # Add Coreference
    if "coreference" in nlp_dict["data"] and len(nlp_dict["data"]["coreference"]) > 0:
        for cl_id, cl_items in nlp_dict["data"].get("coreference", {}).items():
            for item in cl_items:
                key = f"{item['locationStart']}_{item['locationEnd']}"
                if key in universal_dict:
                    universal_dict[key]["cluster_id"] = int(cl_id)
                    universal_dict[key]["surfaceForm"] = item["surfaceForm"]
                else:
                    universal_dict[key] = {"cluster_id": int(cl_id), "surfaceForm": item["surfaceForm"]}
    
    # DEBUG:
    json.dump(universal_dict, open("cheche_universal.json", "w"), indent=2, ensure_ascii=False)

    ### Unify Entity Duplicates
    unified_universal_dict = {} # {f'ent_{cluster_id}': [universal_obj1, universal_obj2, ...]}
    ent_nlp2ent_univ = {}
    clustered_items = set()
    singleton_ids = 1
    if "coreference" in nlp_dict["data"]:
        singleton_ids = len(nlp_dict["data"]["coreference"]) + 1
        for cl_id, cl_items in nlp_dict["data"]["coreference"].items():
            for item in cl_items:
                key = f"{item['locationStart']}_{item['locationEnd']}"
                univ_item = universal_dict.get(key)
                if univ_item and "nlp_id" in univ_item:
                    ent_univ_key = f"ent_{univ_item['cluster_id']+1}"
                    if univ_item["cluster_id"] >= 0:
                        if ent_univ_key in unified_universal_dict:
                            if univ_item["nlp_id"] not in unified_universal_dict[ent_univ_key]["nlp_ids"]:
                                unified_universal_dict[ent_univ_key]["nlp_ids"].append(univ_item["nlp_id"])
                            if key not in unified_universal_dict[ent_univ_key]["spans"]:
                                unified_universal_dict[ent_univ_key]["spans"].append(key)
                            if univ_item.get("surfaceForm") not in unified_universal_dict[ent_univ_key]["surfaceForms"]:
                                unified_universal_dict[ent_univ_key]["surfaceForms"].append(univ_item.get("surfaceForm"))
                            for n in univ_item.get("ner", []):
                                if n not in unified_universal_dict[ent_univ_key]["ner"]:
                                    unified_universal_dict[ent_univ_key]["ner"].append(n)
                            for r in univ_item.get("relations", []):
                                if r not in unified_universal_dict[ent_univ_key]["relations"]:
                                    unified_universal_dict[ent_univ_key]["relations"].append(r)
                            if univ_item.get("wiki_link") and univ_item.get("wiki_link") not in unified_universal_dict[ent_univ_key]["wiki_links"]:
                                unified_universal_dict[ent_univ_key]["wiki_links"].append(univ_item.get("wiki_link"))
                        else:
                            unified_universal_dict[ent_univ_key] = {
                                "nlp_ids": [univ_item["nlp_id"]], # Maybe it will be cleaner to filter earlier ONLY FOR ITEMS WITH NLP_ID!!!
                                "spans": [key],
                                "surfaceForms": [univ_item["surfaceForm"]],
                                "ner": univ_item.get("ner", []),
                                "relations": univ_item.get("relations", []),
                                "wiki_links": [univ_item.get("wiki_link")]
                            }
                        ent_nlp2ent_univ[univ_item["nlp_id"]] = ent_univ_key
                    else:
                        unified_universal_dict[f"ent_{singleton_ids}"] = {
                                "nlp_ids": [univ_item["nlp_id"]],
                                "spans": [key],
                                "surfaceForms": [univ_item["surfaceForm"]],
                                "ner": univ_item.get("ner", []),
                                "relations": univ_item.get("relations", []),
                                "wiki_links": [univ_item.get("wiki_link")]
                            }
                        ent_nlp2ent_univ[univ_item["nlp_id"]] = f"ent_{singleton_ids}"
                        singleton_ids += 1
                    clustered_items.add(univ_item["nlp_id"])
                else:
                    pass # These are the cluster items that DO NOT have a recognized Entity by the NER's
    
    # If there's no coreference then Produce a Similar Formatted Dictionary where each cluster has one entity, so the next code still runs...
    # Even if there was correference, this loop adds all of the entities that did not have any mention in the CLUSTERS
    for span, univ_item in universal_dict.items():
        if "nlp_id" in univ_item and univ_item["nlp_id"] not in clustered_items:
            # if any([x in ["DATE", "PER", "PERSON", "LOC", "GPE", "FAC", "ORG", "WORK_OF_ART", "NORP"] for x in univ_item.get("ner", [])]):
            # print(univ_item)
            unified_universal_dict[f"ent_{singleton_ids}"] = {
                                    "nlp_ids": [univ_item["nlp_id"]],
                                    "spans": [span],
                                    "surfaceForms": [univ_item["surfaceForm"]],
                                    "ner": univ_item["ner"],
                                    "relations": univ_item.get("relations", []),
                                    "wiki_links": [univ_item.get("wiki_link")]
                                }
            # Add also Wikipedia Links present in Metadata
            wiki_link = wiki_linked_dict.get(univ_item["surfaceForm"])
            if wiki_link:
                unified_universal_dict[f"ent_{singleton_ids}"]["wiki_links"].append(wiki_link)
            ent_nlp2ent_univ[univ_item["nlp_id"]] = f"ent_{singleton_ids}"
            singleton_ids += 1

    json.dump(unified_universal_dict, open("cheche_unified_universal.json", "w"), indent=2, ensure_ascii=False)

    # 1) Populate Valid Entities (NLP + WikiMeta)
    idm_entity_dict = {}
    univ_id2idm_id = {}
    per_ix, pl_ix, gr_ix, obj_ix = 0,0,0,0
    event_ix = 1
    kown_coords_dict = {} # To avoid querying more than once for the same entity
    for ent_id, unified_ent_obj in unified_universal_dict.items():
        idm_ent = None
        # A) Choose the IDM Values best on the Most Common when unified
        surface_form = Counter(unified_ent_obj["surfaceForms"]).most_common(1)[0][0]
        ner_category = Counter(unified_ent_obj["ner"]).most_common(1)[0][0]
        # B) IDM ENTITIES
        if ner_category in ["PER", "PERSON"]:
            idm_ent = copy.deepcopy(person_template)
            per_ix += 1
            idm_ent["id"] = f"{lastname}-pr-{stringify_id(per_ix)}"
            idm_ent["kind"] = "person"
        elif ner_category in ["LOC", "GPE", "FAC"]:
            idm_ent = copy.deepcopy(place_template)
            pl_ix += 1
            idm_ent["id"] = f"{lastname}-pl-{stringify_id(pl_ix)}"
            idm_ent["kind"] = "place"
        elif ner_category in ["ORG"]:
            idm_ent = copy.deepcopy(group_template)
            gr_ix += 1
            idm_ent["id"] = f"{lastname}-gr-{stringify_id(gr_ix)}"
            idm_ent["kind"] = "group"
        elif ner_category in ["WORK_OF_ART"]:
            idm_ent = copy.deepcopy(object_template)
            obj_ix += 1
            idm_ent["id"] = f"{lastname}-ob-{stringify_id(obj_ix)}"
            idm_ent["kind"] = "cultural-heritage-object"
        # TODO: FIX? CURRENTLY WE ARE DROPPING "DATE" entities and therefore also relationships that point to those entities such as: "date_of_birth"
        # elif ner_category in ["DATE"]:
        
        # LAST) Add to the IDM Entities
        univ_id2idm_id[ent_id] = None
        if idm_ent:
            univ_id2idm_id[ent_id] = idm_ent["id"]
            idm_ent["label"] = {"default": surface_form}
            idm_entity_dict[ent_id] = idm_ent

    for ent_id, unified_ent_obj in unified_universal_dict.items():
        # If it is a valid entity then add the advanced Attributes
        if ent_id in idm_entity_dict:
            idm_ent = idm_entity_dict[ent_id]
            # C) Add Link Info form Wikipedia Meta (Coordinates)
            available_links = [x for x in unified_ent_obj["wiki_links"] if x is not None]
            if len(available_links) > 0:
                wiki_link = Counter(available_links).most_common(1)[0][0]
            else:
                wiki_link = None
            if wiki_link:
                print(wiki_link)
                if wiki_link not in kown_coords_dict:
                    items_dict = get_relevant_items_from_infobox(wiki_link)
                    coord = items_dict.get("coordinates")
                    kown_coords_dict[wiki_link] = coord
                else:
                    coord = kown_coords_dict[wiki_link]
                if coord:
                    idm_ent["geometry"] = {"type": "Point", "coordinates": coord}
            # Add Relations
            for rel_obj in unified_ent_obj.get("relations", []):
                ev_sub_id = idm_ent["id"].split("-")[1]
                full_event_id = f"{firstname}-{ev_sub_id}-ev-{stringify_id(event_ix)}"
                # idm_ent["relations"].append({"event": full_event_id, "role": f"role-{rel_obj['relationValue']}"})
                # Mapping pointing to the IDM IDS made in the previous LOOP!
                # TODO: We are currently loosing the events that have either subj_idm or obj_idm_id NULL!
                subj_univ_id = ent_nlp2ent_univ[rel_obj['subjectID']] 
                subj_idm_id = univ_id2idm_id[subj_univ_id]
                obj_univ_id = ent_nlp2ent_univ[rel_obj['objectID']]
                obj_idm_id = univ_id2idm_id[obj_univ_id]
                if full_event_id not in parent_idm["events"]:
                    if subj_idm_id and obj_idm_id:
                        idm_ent["relations"].append({"event": full_event_id, "role": f"role-{rel_obj['relationValue']}"})
                        parent_idm["events"].append({
                            "id": full_event_id,
                            "label": { "default": rel_obj["surfaceFormObj"] },
                            "kind": f"event-kind-{rel_obj['relationValue']}",
                            # "startDate": "",
                            "relations": [{ "entity": subj_idm_id, "role": f"role-{rel_obj['relationValue']}"}, # MAPPING!!! from rel_obj['subjectID'] --> UniversalID
                                        { "entity": obj_idm_id, "role": f"role-{inverse_relations_dict.get(rel_obj['relationValue'], 'unk')}"}]
                    })
                    event_ix += 1
                else:
                    if subj_idm_id and obj_idm_id:
                        idm_ent["relations"].append({"event": full_event_id, "role": f"role-{rel_obj['relationValue']}"})
                        parent_idm["events"][full_event_id]["relations"].append({ "entity": subj_idm_id, "role": f"role-{rel_obj['relationValue']}"})
                        parent_idm["events"][full_event_id]["relations"].append({ "entity": obj_idm_id, "role": f"role-{inverse_relations_dict.get(rel_obj['relationValue'], 'unk')}"})
                        event_ix += 1
            # Add updated ubject back to dict
            idm_entity_dict[ent_id] = idm_ent
    
    # 4) Transfer The MERGED Entity-Rel-Linked info into the parent object
    for _, ent_obj in idm_entity_dict.items():
        parent_idm["entities"].append(ent_obj)
    
    # 5) Save IDM JSON File
    with open(idm_out_path, "w") as fp:
        json.dump(parent_idm, fp, indent=2, ensure_ascii=False)

def stringify_id(number: int) -> str:
    if 0 < number < 10:
        return f"00{number}"
    elif 10 <= number < 100:
        return f"0{number}"
    else:
        return str(number)

if __name__ == "__main__":
    # convert_nlp_to_idm_json("english/data/json/albrecht_dürer.json", "english/data/idm/albrecht_dürer.idm.json")
    convert_nlp_to_idm_json("english/data/json/ida_laura_pfeiffer.json", "english/data/idm/ida_pfeiffer.idm.json")