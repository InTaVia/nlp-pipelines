import json, os
import copy

person_template = {
    "id": "", # duerer-pr-012
    "label": { "default": "" }, #surfaceForm
    "linkedIds": [],
    "media": [],
    "relations": [],
    "kind": "person",
    "gender": { } # "id": "male", "label": { "default": "male" }
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
    # 1) Populate Valid Entities
    entity_dict = {}
    unique_entities = set()
    per_ix, pl_ix, gr_ix, obj_ix = 0, 0, 0, 0
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
        
        if idm_ent and ent["surfaceForm"] not in unique_entities:
            idm_ent["label"] = ent["surfaceForm"]
            entity_dict[ent["ID"]] = idm_ent
            unique_entities.add(ent["surfaceForm"])
            
    # 2) Populate Valid Relations and Link to Entities ?
    # 3) Populate Valid Links to Wikipedia URLs
    for link_ent in nlp_dict["data"]["linked_entities"]:
        if link_ent["entityID"] in entity_dict:
            if entity_dict[link_ent["entityID"]]["kind"] == "person":
                linked_id = {
                    "id": link_ent["wikiTitle"],
                    "provider": {
                        "label": "Wikipedia",
                        "baseUrl": link_ent["wikiURL"]
                    }
                }
                entity_dict[link_ent["entityID"]]["linkedIds"].append(linked_id)
    # 4) Transfer The merged Entity-Rel-Linked info into the parent object
    for _, ent_obj in entity_dict.items():
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
    convert_nlp_to_idm_json("english/data/json/albrecht_dürer.flair.json", "english/data/idm/albrecht_dürer.idm.json")