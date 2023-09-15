import glob, os, json
from nlp_to_idm_json import convert_nlp_to_idm_json, stringify_id, create_idm_event
from nlp_to_idm_json import person_template, place_template, group_template, object_template


def generate_idms_from_group(movement_name: str):
    movement_out = f"data/idm/{movement_name}"
    if not os.path.exists(movement_out): os.mkdir(movement_out)
    for filepath in glob.glob(f"data/wikipedia/{movement_name}/*.nlp.json"):
        person_name = filepath.split("/")[-1][:-9]
        convert_nlp_to_idm_json(filepath, f"{movement_out}/{person_name}.idm.json", wiki_root_path=f"data/wikipedia/{movement_name}/")


def unify_idm_jsons(movement_name: str):
    all_entities = {}
    all_biographies = []
    entity_mapper = {} # Map file-specific IDs to Group General IDs
    all_pr_id, all_pl_id, all_gr_id, all_ob_id = 0, 0, 0, 0
    event_id, all_events, event_mapper = 0, {}, {}
    all_md_id, media_mapper, all_media_objs = 0, {}, []
    all_event_kinds, all_roles = {}, {}
    aek_id, ar_id = 0, 0
    main_entities = [] # This list contains the central entity of each "Original ego network"
    for filepath in glob.glob(f"data/idm/{movement_name}/*.idm.json"):
        obj = json.load(open(filepath))
        # Handle Unification of ENTITIES
        print(len(obj["entities"]))
        for ix, ent in enumerate(obj["entities"]):
            if len(ent["id"]) == 0: continue
            if ix == 0: main_entities.append(ent["id"])
            bio_id = ent["id"].split('-')[0]
            type_id = ent["id"].split('-')[1]
            unique_id = "-".join(ent["id"].split('-')[2:])
            if "0" in unique_id or "1" in unique_id or "2" in unique_id or "3" in unique_id:
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
            # Change media-ids to a general counter
            if "media" in ent and len(ent["media"]) > 0:
                all_md_id += 1
                mov_ent_media_id = f"{movement_name}-m-{stringify_id(all_md_id)}"
                media_mapper[ent["media"][0]] = mov_ent_media_id
                ent["media"] = [mov_ent_media_id]
            # Add to the General Entity Dict
            all_entities[mov_ent_id] = ent
        # Unification of EVENTS with a single event index for all files
        for event in obj["events"]:
            event_id += 1
            new_id = f"{movement_name}-ev-{stringify_id(event_id)}"
            event_mapper[event["id"]] = new_id
            event["id"] = new_id
            all_events[new_id] = event
        for ek in obj["vocabularies"]["event-kind"]:
            aek_id += 1
            all_event_kinds[aek_id] = ek
        for r in obj["vocabularies"]["role"]:
            ar_id += 1
            all_roles[ar_id] = r
        # Map the Media Objects to the new Ids
        for media_obj in obj["media"]:
            all_media_objs.append(media_obj)
        # Add all the biographies available
        for bio in obj["biographies"]:
            all_biographies.append(bio)

    json.dump(entity_mapper, open("cheche_entity_mapper.json", "w"), indent=2, ensure_ascii=False)  
    # Map the File-based ID's to the Movement IDs in the relations of all entities    
    for _, ent in all_entities.items():
        for rel in ent["relations"]:
            rel["event"] = event_mapper[rel["event"]]
    # Map the File-based Relations inside the Events into the global relations
    for _, event in all_events.items():
        for rel in event["relations"]:
            if rel["entity"]:
                rel["entity"] = entity_mapper[rel["entity"]]

    json.dump(media_mapper, open("cheche_media_mapper.json", "w"), indent=2, ensure_ascii=False)     

    for media_obj in all_media_objs:
        if media_obj["id"] in media_mapper:
            media_obj["id"] = media_mapper[media_obj["id"]]

    # # An extra event to Connect main subjects ?? Or just a Relation?? This gives Internal Server Error
    # for subj_ent_id, obj_ent_id in zip(main_entities, main_entities[1:]):
    #     subj_idm_id = entity_mapper[subj_ent_id]
    #     obj_idm_id = entity_mapper[obj_ent_id]
    #     ev_sub_id = f"{movement_name}-ev-{stringify_id(event_id)}"
    #     # This event is "passive" (the object "was created"), that's why the other entity we need to get is the Subj Entity (the creator)
    #     # We are assuming the first entity of the text == Subject of the Biography
    #     subj_idm_ent = all_entities[subj_idm_id]
    #     obj_idm_ent = all_entities[obj_idm_id]
    #     event_info = {"full_event_id": ev_sub_id, 
    #                     "event_label": "same_movement_as", 
    #                     "event_kind": "same_movement_as", 
    #                     "subj_role": "same_movement_as", 
    #                     "obj_role": "same_movement_as"}
    #     event_relations = [{ "entity": obj_idm_ent, "role": f"role-{event_info['obj_role']}"},
    #                     {"entity": subj_idm_id, "role": f"role-{event_info['subj_role']}" }
    #                 ]
    #     subj_idm_ent, obj_idm_ent, event_obj, all_event_kinds, all_roles = create_idm_event(event_info, subj_idm_id, subj_idm_ent, obj_idm_ent, event_relations, all_event_kinds, all_roles)
    #     event_id += 1
    #     all_events[ev_sub_id] = event_obj
    #     # Add updated object back to dict
    #     all_entities[subj_idm_id] = subj_idm_ent
    #     all_entities[obj_idm_id] = obj_idm_ent
        

    # Save the Grouped Entities, Relations and Events into a single IDM JSON
    group_parent_idm = {
        "entities": sorted(all_entities.values(), key= lambda x: - len(x["relations"])),
        "events": sorted(all_events.values(), key= lambda x: x["id"]),
        "media": all_media_objs,
        "biographies": all_biographies,
        "vocabularies": {"event-kind": list(all_event_kinds.values()), "role": list(all_roles.values())},
        "unmappedEntities": [],
        "collections": {}
    }

    json.dump(group_parent_idm, open("cheche_group_idm.json", "w"), indent=2, ensure_ascii=False)


if __name__ == "__main__":
    # generate_idms_from_group("Art_Nouveau")
    unify_idm_jsons("Art_Nouveau")