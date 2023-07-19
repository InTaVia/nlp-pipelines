from typing import Dict, Tuple, Any
import json, os, re, time
import copy
from collections import defaultdict, Counter
from utils.utils_wiki import get_raw_wikipedia_article, get_wiki_linked_entities, get_relevant_items_from_infobox

inverse_relations_dict = {
    "based_in": "location_of",
    "born_in": "place_of_birth",
    "child_of": "parent_of",
    "lived_in": "place_of_residence",
    "married_to": "married_to",
    "parent_of": "child_of",
    "sibling_of": "sibling_of"
}

person_template = {
    "id": "", # duerer-pr-012
    "label": { "default": "" }, #surfaceForm
    "linkedIds": [],
    "media": [],
    "relations": [],
    # "kind": "person",
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
    # "type": { }, # { "id": "place-type-city", "label": { "default": "city" } }
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
        "vocabularies": {#"media-kind": [],                      # { "id": "media-kind-image", "label": { "default": "image" } }
                         #"cultural-heritage-object-type": [],   # { "id": "cultural-heritage-object-type-book", "label": { "default": "book" }
                         #"group-type": [],                      # { "id": "group-type-workspace", "label": { "default": "workspace" } },
                         #"historical-event-type": [],           # { "id": "historical-event-type-coronation", "label": { "default": "coronation" }}
                         #"occupation": [],                      # { "id": "occupation-bildhauer", "label": { "default": "Bildhauer" } }
                         #"place-type": [],                      # { "id": "place-type-city", "label": { "default": "city" } }
                         #"event-kind": [],                      # { "id": "event-kind-creation", "label": { "default": "creation" } },
                         #"role": []                             # { "id": "role-object_created", "label": { "default": "object_created" } },
                        },
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
    surfaceForm_dict = defaultdict(list) # Across ALL NLP LAYERS collect all the ID's that match to a surface form
    # Add Entities
    entity_dict = {}
    for ent_obj in nlp_dict["data"]["entities"]:
        key = f"{ent_obj['locationStart']}_{ent_obj['locationEnd']}"
        if key in universal_dict:
            universal_dict[key]["ner"].append(ent_obj["category"])
        else:
            universal_dict[key] = {"nlp_id":ent_obj["ID"], "sent_id": ent_obj["sentenceID"], "locationStart": ent_obj["locationStart"], "locationEnd": ent_obj["locationEnd"],
                                    "surfaceForm": ent_obj["surfaceForm"], "ner": [ent_obj["category"]], "relations": [], "cluster_id": -1}
        entity_dict[ent_obj["ID"]] = ent_obj
        surfaceForm_dict[key].append(ent_obj["ID"])
    # Add Relations
    relation_dict = {}
    for relation in nlp_dict["data"].get("relations", []):
        subj_id = relation['subjectID']
        obj_id = relation['objectID']
        rel_subj = entity_dict.get(subj_id)
        rel_obj = entity_dict.get(obj_id)
        if rel_subj and rel_obj:
            key = f"{rel_subj['locationStart']}_{rel_subj['locationEnd']}"
            universal_dict[key]["relations"].append(relation)
            key = f"{rel_obj['locationStart']}_{rel_obj['locationEnd']}"
            universal_dict[key]["relations"].append(relation) 
            relation_dict[relation["relationID"]] = relation
    # # Add Semantic roles
    # for proposition in nlp_dict["data"].get("semantic_roles", []):
    #     key = f"{proposition['locationStart']}_{proposition['locationEnd']}"
    #     predicate = proposition.get("predicateSense", proposition["surfaceForm"])
    #     triples = {predicate: []}
    #     for arg in proposition["arguments"]:
    #         triples[predicate].append((arg["surfaceForm"], arg["category"]))
    #     if key in universal_dict:
    #         universal_dict[key]["srl"].append(triples)
    #     else:
    #         universal_dict[key] = {"srl": [triples], "sent_id": proposition["sentenceID"]}
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

    ### Sentence-Based Event Unification
    sentence_based_event_ids = defaultdict(set)
    sentence_based_events = defaultdict(set)
    for span, nlp_info in universal_dict.items():
        sent_id = nlp_info.get("sent_id", "-1")
        if sent_id in sentence_based_event_ids:
            sentence_based_event_ids[sent_id].add(span)
        else:
            sentence_based_event_ids[sent_id] = set([span])

        if sent_id in sentence_based_events:
            if "ner" in nlp_info:
                ner = nlp_info["ner"][0]
                sentence_based_events[sent_id]["ner"].add((nlp_info["surfaceForm"], ner))
            if "relations" in nlp_info:
                rels = []
                for rel in nlp_info["relations"]:
                    sentence_based_events[sent_id]["relations"].add((rel["surfaceFormSubj"], rel["relationValue"], rel["surfaceFormObj"]))
            # if "srl" in nlp_info:
            #    tmp["srl"] += nlp_info["srl"] # TODO: "SMART TRIPLE SELECTION" (From Go's code)
        else:
            tmp = {}
            if "ner" in nlp_info:
                ner = nlp_info["ner"][0]
                tmp["ner"] = set([(nlp_info["surfaceForm"], ner)])
            else:
                tmp["ner"] = set()
            if "relations" in nlp_info:
                rels = set()
                for rel in nlp_info["relations"]:
                    rels.add((rel["surfaceFormSubj"], rel["relationValue"], rel["surfaceFormObj"]))
                tmp["relations"] = rels
            # if "srl" in nlp_info:
            #     tmp["srl"] = nlp_info["srl"] # TODO: "SMART TRIPLE SELECTION" (From Go's code)
            sentence_based_events[sent_id] = tmp


    # DEBUG:
    for key, dct in sentence_based_events.items():
        for k, v in dct.items():
            sentence_based_events[key][k] = list(v)
    json.dump(sentence_based_events, open("cheche_sentence_based.json", "w"), indent=2, ensure_ascii=False)


    ### Unify Entity Duplicates
    if "coreference" in nlp_dict["data"]:
        # {f'ent_{cluster_id}': [universal_obj1, universal_obj2, ...]}
        unified_universal_dict, clustered_items, ent_nlp2ent_univ, singleton_ids, surfaceForm2ent_univ = create_unified_universal_dict(nlp_dict, universal_dict)
    else:
        unified_universal_dict = {}
        clustered_items = set()
        ent_nlp2ent_univ = {}
        singleton_ids = 1
        surfaceForm2ent_univ = {}
    
    # DEBUG:
    json.dump(surfaceForm2ent_univ, open("cheche_surface.json", "w"), ensure_ascii=False, indent=2)

    # Even if there was NO coreference, this loop adds all of the entities that did not have any mention in the CLUSTERS
    for span, univ_item in universal_dict.items():
        if "nlp_id" in univ_item and univ_item["nlp_id"] not in clustered_items:
            # To disambiguate for 100% String Matching cases e.g. all unclustered ents whose surfaceForm is exactly "Vienna"
            if univ_item["surfaceForm"] in surfaceForm2ent_univ:
                ent_univ_key = surfaceForm2ent_univ[univ_item["surfaceForm"]]
                unified_universal_dict[ent_univ_key]["nlp_ids"].append(univ_item["nlp_id"])
                unified_universal_dict[ent_univ_key]["spans"].append(span)
                unified_universal_dict[ent_univ_key]["surfaceForms"].append(univ_item["surfaceForm"])
                if univ_item.get("ner"):
                    unified_universal_dict[ent_univ_key]["ner"] += univ_item["ner"]
                if univ_item.get("relations"):
                    unified_universal_dict[ent_univ_key]["relations"] += univ_item["relations"]
                if univ_item.get("wiki_link"):
                    unified_universal_dict[ent_univ_key]["wiki_links"].append(univ_item["wiki_link"])
                wiki_link = wiki_linked_dict.get(univ_item["surfaceForm"])
                if wiki_link:
                    unified_universal_dict[ent_univ_key]["wiki_links"].append(wiki_link)
                ent_nlp2ent_univ[univ_item["nlp_id"]] = ent_univ_key
            # Nothing to diambiguate, new univ_entity_id ...
            else:
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

    # DEBUG:
    json.dump(unified_universal_dict, open("cheche_unified_universal.json", "w"), indent=2, ensure_ascii=False)

    # 1) Populate Valid Entities (NLP + WikiMeta)
    idm_entity_dict = {}
    univ_id2idm_id, idm_id2univ_id = {}, {}
    per_ix, pl_ix, gr_ix, obj_ix = 0,0,0,0
    event_ix = 1
    kown_coords_dict = {} # To avoid querying more than once for the same entity
    event_vocab, role_vocab = {}, {}
    for ent_id, unified_ent_obj in unified_universal_dict.items():
        idm_ent = None
        # A) Choose the IDM Values best on the Most Common when unified
        surface_form = sorted(unified_ent_obj["surfaceForms"], key= lambda x: len(x))[-1] # Assign the longest found form (otherwise lastName is chosen)
        ner_category = Counter(unified_ent_obj["ner"]).most_common(1)[0][0] # Assign the most NER predicted label
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
            idm_id2univ_id[idm_ent["id"]] = ent_id
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
            # Convert NLP CHO Entities --> IDM Creation Events and Relations [role-object_created, role-was_creator]
            if "WORK_OF_ART" in unified_ent_obj["ner"]:
                subj_idm_id = f"{lastname}-pr-001"
                # This event is "passive" (the object "was created"), that's why the other entity we need to get is the Subj Entity (the creator)
                # We are assuming the first entity of the text == Subject of the Biography
                subj_idm_ent = idm_entity_dict[idm_id2univ_id[subj_idm_id]]
                event_info = {"full_event_id": f"{lastname}-{ev_sub_id}-ev-{stringify_id(event_ix)}", 
                              "event_label": unified_ent_obj["surfaceForms"][0], 
                              "event_kind": "creation", 
                              "subj_role": "was_creator", 
                              "obj_role": "object_created"}
                subj_idm_ent, idm_ent, event_obj, event_vocab, role_vocab = create_idm_event(event_info, subj_idm_id, subj_idm_ent, idm_ent, event_vocab, role_vocab)
                # Add updated object back to dict
                idm_entity_dict[idm_id2univ_id[subj_idm_id]] = subj_idm_ent
                # Add Event to IDM Main Object
                parent_idm["events"].append(event_obj)
                event_ix += 1

            # Convert NLP Relations --> IDM Relations
            for rel_obj in unified_ent_obj.get("relations", []):
                ev_sub_id = idm_ent["id"].split("-")[1]
                subj_role =  rel_obj['relationValue']
                obj_role = inverse_relations_dict.get(rel_obj['relationValue'], 'unk')
                event_info = {"full_event_id": f"{lastname}-{ev_sub_id}-ev-{stringify_id(event_ix)}", 
                              "event_label": rel_obj["surfaceFormObj"], 
                              "event_kind": f"event-kind-{subj_role}", 
                              "subj_role": subj_role, 
                              "obj_role": obj_role}
                subj_univ_id = ent_nlp2ent_univ[rel_obj['subjectID']] 
                subj_idm_id = univ_id2idm_id[subj_univ_id]
                obj_univ_id = ent_nlp2ent_univ[rel_obj['objectID']]
                obj_idm_ent = idm_entity_dict.get(obj_univ_id) # They don't exist when it is a labeled entity outside the scope of interest (e.g. NORP, DATE, etc...)
                if obj_idm_ent:
                    idm_ent, obj_idm_ent, event_obj, event_vocab, role_vocab = create_idm_event(event_info, subj_idm_id, idm_ent, obj_idm_ent, event_vocab, role_vocab)
                    # Add updated object back to dict
                    idm_entity_dict[obj_univ_id] = obj_idm_ent
                    # Add Event to IDM Main Object
                    parent_idm["events"].append(event_obj)
                    event_ix += 1

            # Add updated current object back to dict
            idm_entity_dict[ent_id] = idm_ent
    
    # 4) Transfer The MERGED Entity-Rel-Linked info into the parent object
    for _, ent_obj in idm_entity_dict.items():
        parent_idm["entities"].append(ent_obj)
    
    # Fill In Vocabularies
    parent_idm["vocabularies"]["event-kind"] = list(event_vocab.values())
    parent_idm["vocabularies"]["role"] = list(role_vocab.values())

    # 6) Save IDM JSON File
    with open(idm_out_path, "w") as fp:
        json.dump(parent_idm, fp, indent=2, ensure_ascii=False)


def stringify_id(number: int) -> str:
    if 0 < number < 10:
        return f"00{number}"
    elif 10 <= number < 100:
        return f"0{number}"
    else:
        return str(number)


def create_unified_universal_dict(nlp_dict: Dict[str, Any], universal_dict: Dict[str, Any]) -> Tuple[Any]:
    unified_universal_dict = {} 
    ent_nlp2ent_univ = {}
    clustered_items = set()
    surfaceForm2ent_univ = {}
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
                            "nlp_ids": [univ_item["nlp_id"]],
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
                surfaceForm2ent_univ[univ_item["surfaceForm"]] = ent_univ_key
            else:
                # print(univ_item)
                pass # These are the cluster items that DO NOT have a recognized Entity by the NER's. But if they appear in SRL we can grab the event!

    return unified_universal_dict, clustered_items, ent_nlp2ent_univ, singleton_ids, surfaceForm2ent_univ


def create_idm_event(event_info: Dict[str, str], subj_idm_id: str, subj_idm_ent: Dict, obj_idm_ent: Dict, event_vocab: Dict, role_vocab: Dict):
    full_event_id = event_info["full_event_id"]
    event_label = event_info["event_label"]
    event_kind = event_info["event_kind"] # "creation"
    subj_role =  event_info["subj_role"] # "was_creator"
    obj_role = event_info["obj_role"] # "object_created"
    subj_idm_ent["relations"].append({"event": full_event_id, "role": f"role-{subj_role}"})
    obj_idm_ent["relations"].append({"event": full_event_id, "role": f"role-{obj_role}"})

    event_obj = {
                "id": full_event_id,
                "label": { "default": event_label },
                "kind": f"event-kind-{event_kind}",
                # "startDate": "",
                "relations": [{ "entity": obj_idm_ent["id"], "role": f"role-{obj_role}" },
                              { "entity": subj_idm_id, "role": f"role-{subj_role}" },
                                # { "entity": "duerer-pl-012", "role": "role-took_place_at" },
                                # { "entity": "duerer-gr-019", "role": "role-current_location" }
                            ]
            }
    
    event_vocab[f"event-kind-{event_kind}"] = {"id": f"event-kind-{event_kind}", "label": {"default": f"{obj_role}"}}
    role_vocab[f"role-{subj_role}"] = { "id": f"role-{subj_role}", "label": { "default": subj_role}}
    role_vocab[f"role-{obj_role}"] = { "id": f"role-{obj_role}", "label": { "default": obj_role}}

    return subj_idm_ent, obj_idm_ent, event_obj, event_vocab, role_vocab

    

if __name__ == "__main__":
    convert_nlp_to_idm_json("english/data/json/albrecht_dürer.json", "english/data/idm/albrecht_dürer.idm.json")
    # convert_nlp_to_idm_json("english/data/json/ida_laura_pfeiffer.json", "english/data/idm/ida_pfeiffer.idm.json")