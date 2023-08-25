import glob, os, json
from nlp_to_idm_json import convert_nlp_to_idm_json, stringify_id
from nlp_to_idm_json import person_template, place_template, group_template, object_template


def generate_idms_from_group(movement_name: str):
    movement_out = f"data/idm/{movement_name}"

    if not os.path.exists(movement_out): os.mkdir(movement_out)

    for filepath in glob.glob(f"data/wikipedia/{movement_name}/*.nlp.json"):
        person_name = filepath.split("/")[-1][:-9]
        convert_nlp_to_idm_json(filepath, f"{movement_out}/{person_name}.idm.json", wiki_root_path=f"data/wikipedia/{movement_name}/")


def unify_idm_jsons(movement_name: str):
    group_parent_idm = {
        "entities": [],
        "events": [],
        "media": [],
        "biographies": [],
        "vocabularies": {},
        "unmappedEntities": [],
        "collections": {}
    }
    all_entities = {}
    entity_mapper = {}
    all_pr_id, all_pl_id, all_gr_id, all_ob_id = 0, 0, 0, 0
    for filepath in glob.glob(f"data/idm/{movement_name}/*.idm.json"):
        obj = json.load(open(filepath))
        # Handle Unification of ENTITIES
        print(len(obj["entities"]))
        for ent in obj["entities"]:
            bio_id = ent["id"].split('-')[0]
            type_id = ent["id"].split('-')[1]
            unique_id = "-".join(ent["id"].split('-')[2:])
            if "0" in unique_id or "1" in unique_id or "2" in unique_id:
                if type_id == "pr":
                    all_pr_id += 1
                    mov_ent_id = f"{movement_name}-{type_id}-{stringify_id(all_pr_id)}"
                elif type_id == "pl":
                    all_pl_id += 1
                    mov_ent_id = f"{movement_name}-{type_id}-{stringify_id(all_pl_id)}"
                elif type_id == "gr":
                    all_gr_id += 1
                    mov_ent_id = f"{movement_name}-{type_id}-{stringify_id(all_gr_id)}"
                elif type_id == "ob":
                    all_ob_id += 1
                    mov_ent_id = f"{movement_name}-{type_id}-{stringify_id(all_ob_id)}"
            else:
                mov_ent_id = f"{movement_name}-{type_id}-{unique_id}"
            # Change Fields to be in the Unified Movement IDM
            entity_mapper[ent["id"]] = mov_ent_id
            ent["id"] = mov_ent_id
            all_entities[mov_ent_id] = ent
            # TODO: Handle Unification of EVENTS
            
    print(len(all_entities))
    for ent in entity_mapper.items():
        print(ent)
    
    # for ent in all_entities.items():
    #     # TODO: For "relations" inside entities in the enxt iteration the entity_ids should be changed using the entity mapper!!!
    #     pass

if __name__ == "__main__":
    #generate_idms_from_group("Art_Nouveau")
    unify_idm_jsons("Art_Nouveau")
    pass