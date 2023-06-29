import json, os, re, time
import copy
from collections import defaultdict
from utils.utils_wiki import get_raw_wikipedia_article, get_wiki_linked_entities, get_relevant_items_from_infobox

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
    lastname = person_name.split("_")[-1]
    
    # Open Raw File and Meta JSON to Complement Data
    wiki_raw = get_raw_wikipedia_article(wiki_title=person_name.replace("_", " ").title())
    wiki_meta = json.load(open(f"english/data/wikipedia/{person_name}.txt.meta.json"))
    wiki_linked_dict = get_wiki_linked_entities(wiki_raw) # {'surfaceForm': 'wiki_link'}

    ### Unify Entity Duplicates
    # Entity-Metions to ClusterIDs
    # mentions_cluster_dict, spans_cluster_dict  = {}, {}
    # if "coreference" in nlp_dict["data"]:
    #     for cl_id, cl_items in nlp_dict["data"]["coreference"].items():
    #         for item in cl_items:
    #             mentions_cluster_dict[item["surfaceForm"]] = cl_id
    #             spans_cluster_dict[(item["locationStart"], item["locationEnd"])] = cl_id

    # unified_entities = defaultdict(list)
    # for ent in nlp_dict["data"].get("entities", []):
    #     if "flair" in ent["method"]: 
    #         clean_entity_id = ent["ID"].strip("_flair")
    #         cluster_id = mentions_cluster_dict.get(ent["surfaceForm"], -1)
    #         unified_entities[cluster_id].append({"surfaceForm": ent["surfaceForm"], "category": ent["category"]})
    #     else:
    #         unified_entities[-1].append({"surfaceForm": ent["surfaceForm"], "category": ent["category"]})

    # uniqueID = 0
    # idm_entity_candidates = []
    # for cl_id, entities in unified_entities.items():
    #     longest_ent = sorted(entities, key= lambda x: len(x))[-1]
    #     print(cl_id, [e["surfaceForm"] for e in entities])
    #     idm_entity_candidates.append()

    unified_universal_dict = {} # {(locStart, locEnd): {prop: val, prop: val, ...}}
    # TODO: Values can be lists rather than idividual values, to acumulate labels form diff models and also diff relations, links etc.
    # The connections across are using uniqueIDs of ==> locstart_locEnd
    entity_dict = {}
    for ent_obj in nlp_dict["data"]["entities"]:
        unified_universal_dict[f"{ent_obj['locationStart']}_{ent_obj['locationEnd']}"] = {"surfaceForm": ent_obj["surfaceForm"],"ner": ent_obj["category"]}
        entity_dict[ent_obj["ID"]] = ent_obj
    for relation in nlp_dict["data"].get("relations", []):
        subj_id = f"{relation['subjectID']}_flair" # for now manually append '_flair'
        obj_id = f"{relation['objectID']}_flair"
        rel_subj = entity_dict.get(subj_id)
        if rel_subj:
            key = f"{rel_subj['locationStart']}_{rel_subj['locationEnd']}"
            if key in unified_universal_dict:
                unified_universal_dict[key]["relation"] = ("subjectOf", relation["relationValue"])
            else:
                unified_universal_dict[key] = {"relation": ("subjectOf", relation["relationValue"])}
        rel_obj = entity_dict.get(obj_id)
        if rel_obj:
            key = f"{rel_obj['locationStart']}_{rel_obj['locationEnd']}"
            if key in unified_universal_dict:
                unified_universal_dict[key]["relation"] = ("objectOf", relation["relationValue"])
            else:
                unified_universal_dict[key] = {"relation": ("objectOf", relation["relationValue"])}
    #Links
    for link_ent in nlp_dict["data"].get("linked_entities", []):
        key = f"{link_ent['locationStart']}_{link_ent['locationEnd']}"
        if key in unified_universal_dict:
            unified_universal_dict[key]["wiki_link"] = link_ent["wikiURL"]
        else:
            unified_universal_dict[key] = {"wiki_link": link_ent["wikiURL"]}
    #Coreference
    for cl_id, cl_items in nlp_dict["data"].get("coreference", {}).items():
        for item in cl_items:
            key = f"{item['locationStart']}_{item['locationEnd']}"
            if key in unified_universal_dict:
                unified_universal_dict[key]["cluster_id"] = cl_id
            else:
                unified_universal_dict[key] = {"cluster_id": cl_id}
    
    json.dump(unified_universal_dict, open("cheche_universal.json", "w"), indent=2, ensure_ascii=False)

    # Populate Valid Entities (NLP + WikiMeta)
    idm_entity_dict = {}
    unique_entities = {}
    per_ix, pl_ix, gr_ix, obj_ix = 1, 1, 1, 1
    for ent in nlp_dict["data"]["entities"]:
        idm_ent = None
        if ent["category"] in ["PER", "PERSON"]:
            idm_ent = copy.deepcopy(person_template)
            per_ix += 1
            idm_ent["id"] = f"{lastname}-pr-{stringify_id(per_ix)}"
            idm_ent["kind"] = "person"
        elif ent["category"] in ["LOC", "GPE"]:
            idm_ent = copy.deepcopy(place_template)
            pl_ix += 1
            idm_ent["id"] = f"{lastname}-pl-{stringify_id(pl_ix)}"
            idm_ent["kind"] = "place"
        elif ent["category"] in ["ORG"]:
            idm_ent = copy.deepcopy(group_template)
            gr_ix += 1
            idm_ent["id"] = f"{lastname}-gr-{stringify_id(gr_ix)}"
            idm_ent["kind"] = "group"
        elif ent["category"] in ["WORK_OF_ART"]:
            idm_ent = copy.deepcopy(object_template)
            obj_ix += 1
            idm_ent["id"] = f"{lastname}-ob-{stringify_id(obj_ix)}"
            idm_ent["kind"] = "cultural-heritage-object"
        
        # Add Link Info form Wikipedia Meta
        if ent["surfaceForm"] in wiki_linked_dict:
            wiki_link = wiki_linked_dict[ent["surfaceForm"]]
            items_dict = get_relevant_items_from_infobox(wiki_link)
            coord = items_dict.get("coordinates")
            if coord:
                idm_ent["geometry"] = {"type": "Point", "coordinates": coord}

        # Add to the List of Entities (De-Duplicate entities with the SAME surfaceForm)
        if idm_ent and ent["surfaceForm"] not in unique_entities:
            idm_ent["label"] = {"default": ent["surfaceForm"]}
            idm_entity_dict[ent["ID"]] = idm_ent
            unique_entities[ent["surfaceForm"]] = ent["ID"]
            
    # 2) Populate Valid Relations and Link to Entities ?
    event_ix = 1
    for rel_obj in nlp_dict["data"].get("relations", []):
        subj_id = f"{rel_obj['subjectID']}_flair" # for now manually append '_flair'
        obj_id = f"{rel_obj['objectID']}_flair"
        idm_subj_entity = idm_entity_dict.get(subj_id)
        idm_obj_entity = idm_entity_dict.get(obj_id)
        if idm_subj_entity and idm_obj_entity:
            ev_sub_id = idm_obj_entity['id'].split("-")[1]
            full_event_id = f"duerer-{ev_sub_id}-ev-{stringify_id(event_ix)}"
            idm_subj_entity["relations"].append({"event": full_event_id, 
                                                 "role": f"role-{rel_obj['relationValue']}"})
            idm_entity_dict[subj_id] = idm_subj_entity
            if full_event_id not in parent_idm["events"]:
                parent_idm["events"].append({
                        "id": full_event_id,
                        "label": { "default": rel_obj["surfaceFormObj"] },
                        "kind": f"event-kind-{rel_obj['relationValue']}",
                        "startDate": "",
                        "relations": [{ "entity": idm_subj_entity['id'], "role": f"role-{rel_obj['relationValue']}"}]
                })
            else:
                parent_idm["events"][full_event_id]["relations"].append({ "entity": idm_subj_entity['id'], "role": f"role-{rel_obj['relationValue']}"})
            event_ix += 1

    # 3) Populate Valid Links to Wikipedia URLs
    for link_ent in nlp_dict["data"].get("linked_entities", []):
        if link_ent["entityID"] in idm_entity_dict:
            if idm_entity_dict[link_ent["entityID"]]["kind"] == "person":
                linked_id = {
                    "id": link_ent["wikiTitle"],
                    "provider": {
                        "label": {"default": "Wikipedia"},
                        "baseUrl": link_ent["wikiURL"]
                    }
                }
                idm_entity_dict[link_ent["entityID"]]["linkedIds"].append(linked_id)
    
    # 4) Transfer The merged Entity-Rel-Linked info into the parent object
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
    convert_nlp_to_idm_json("english/data/json/albrecht_dürer.json", "english/data/idm/albrecht_dürer.idm.json")