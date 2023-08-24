from typing import Dict, Tuple, Any, List
import json, os, re, time
import copy
from collections import defaultdict, Counter
from utils.utils_wiki import get_raw_wikipedia_article, get_wiki_linked_entities, get_relevant_items_from_infobox, get_wikipedia_url_encoded
from urllib.parse import unquote
import datetime
import dateutil.parser as parser

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
    main_person_url = get_wikipedia_url_encoded(person_name.replace("_", " ").title())
    main_person_id = unquote(main_person_url.split("/")[-1])

    all_sentences = {str(sent["sentenceID"]): sent["text"] for sent in nlp_dict["data"]["morpho_syntax"]["flair_0.12.2"]}
    
    # Open Raw File and Meta JSON to Complement Data
    wiki_raw = get_raw_wikipedia_article(wiki_title=person_name.replace("_", " ").title())
    wiki_meta = json.load(open(f"data/wikipedia/{person_name}.txt.meta.json"))
    wiki_linked_dict = get_wiki_linked_entities(wiki_raw) # {'surfaceForm': 'wiki_link'}
    for meta_link in wiki_meta["links"]:
        wiki_linked_dict[meta_link] = get_wikipedia_url_encoded(meta_link)
    wiki_linked_dict[person_name.replace("_", " ").title()] = main_person_url
    json.dump(wiki_linked_dict, open("cheche_wiki_linked.json", "w"), indent=2, ensure_ascii=False)

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
    # Add Semantic roles
    for proposition in nlp_dict["data"].get("semantic_roles", []):
        key = f"{proposition['locationStart']}_{proposition['locationEnd']}"
        predicate = proposition["surfaceForm"] #proposition.get("predicateSense", proposition["surfaceForm"])
        triples = {predicate: []}
        for arg in proposition["arguments"]:
            triples[predicate].append((arg["surfaceForm"], arg["category"]))
        universal_dict[key] = {"srl": triples, "sent_id": proposition["sentenceID"]}
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
    json.dump(universal_dict, open("data/idm/nlpidm_universal.json", "w"), indent=2, ensure_ascii=False)

    ### Sentence-Based Event Unification
    sentence_based_event_ids = defaultdict(set)
    sentence_based_events = {}
    for span, nlp_info in universal_dict.items():
        sent_id = str(nlp_info.get("sent_id", "-1"))
        if sent_id in sentence_based_event_ids:
            sentence_based_event_ids[sent_id].add(span)
        else:
            sentence_based_event_ids[sent_id] = set([span])

        if sent_id in sentence_based_events:
            if "ner" in nlp_info:
                ner = nlp_info["ner"][0]
                if "ner" in sentence_based_events[sent_id]:
                    sentence_based_events[sent_id]["ner"].add((nlp_info["surfaceForm"], ner))
                else:
                    sentence_based_events[sent_id]["ner"] = set()
            if "relations" in nlp_info:
                if "relations" not in sentence_based_events[sent_id]:
                    sentence_based_events[sent_id]["relations"] = []
                for rel in nlp_info["relations"]:
                    sentence_based_events[sent_id]["relations"].add((rel["surfaceFormSubj"], rel["relationValue"], rel["surfaceFormObj"]))
            if "srl" in nlp_info:
                srl_triples = get_smart_srl_triples(nlp_info["srl"], lastname)
                if "srl" in sentence_based_events[sent_id]:
                    sentence_based_events[sent_id]["srl"] += srl_triples
                elif len(srl_triples) > 0:
                    sentence_based_events[sent_id]["srl"] = srl_triples
        else:
            tmp = {}
            if "ner" in nlp_info:
                ner = nlp_info["ner"][0]
                tmp["ner"] = set([(nlp_info["surfaceForm"], ner)])
            if "relations" in nlp_info:
                rels = set()
                for rel in nlp_info["relations"]:
                    rels.add((rel["surfaceFormSubj"], rel["relationValue"], rel["surfaceFormObj"]))
                tmp["relations"] = rels
            if "srl" in nlp_info:
                srl_triples = get_smart_srl_triples(nlp_info["srl"], lastname)
                if len(srl_triples) > 0:
                    tmp["srl"] = srl_triples
            sentence_based_events[sent_id] = tmp

    # DEBUG:
    # This "dumb iteration" is to transform the sets into lists because sets are not serializable!
    for sent_id, nlp_layer in sorted(sentence_based_events.items(), key= lambda x: x[0]):
        for nlp_name, nlp_values in nlp_layer.items():
            sentence_based_events[sent_id][nlp_name] = list(nlp_values)
    json.dump(sentence_based_events, open("data/idm/nlpidm_sentence_based_dict.json", "w"), indent=2, ensure_ascii=False)

    # Sentence-Based Events and SRL-Based Events
    sentences_with_events, triples_with_events = [], []
    for sent_id, nlp_layer in sentence_based_events.items():
        dates_found = []
        for ner in nlp_layer.get("ner", []):
            if "DATE" in ner[1] and re.search(r"\d{4}", ner[0]):
                date = ner[0]
                sentences_with_events.append(f"[{sent_id}] {date} --> {all_sentences[sent_id]}")
                dates_found.append(date)
        for date in dates_found:
            for srl in nlp_layer.get("srl", []):
                prop = " ".join(srl)
                if date not in prop and re.search(r"\d{4}", prop): continue # Skip these cases as they have contradictory dates!
                # prop = f"{date} --> {prop}"
                if date in prop: # "Agressive Filter" to keep only the triples that also have ARGM-LOC mtchint the DATE Entity
                    prop = (date, srl)
                    triples_with_events.append(prop)

    with open("data/idm/nlpidm_sentence_based_events.txt", "w") as f:
        for sent in sentences_with_events:
            f.write(f"{sent}\n")

    with open("data/idm/nlpidm_srl_based_events.txt", "w") as f:
        for tr in triples_with_events:
            f.write(f"{tr}\n")


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
    json.dump(surfaceForm2ent_univ, open("data/idm/nlpidm_surface.json", "w"), ensure_ascii=False, indent=2)

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
    json.dump(unified_universal_dict, open("data/idm/nlpidm_unified_universal.json", "w"), indent=2, ensure_ascii=False)

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
        wiki_link = wiki_linked_dict.get(surface_form)
        # B) IDM ENTITIES
        if ner_category in ["PER", "PERSON"]:
            idm_ent = copy.deepcopy(person_template)
            idm_ent["kind"] = "person"
            if wiki_link:
                idm_ent["id"] = f"{main_person_id}-pr-{unquote(wiki_link.split('/')[-1])}"
            else:
                per_ix += 1
                idm_ent["id"] = f"{main_person_id}-pr-{stringify_id(per_ix)}"
        elif ner_category in ["LOC", "GPE", "FAC"]:
            idm_ent = copy.deepcopy(place_template)
            idm_ent["kind"] = "place"
            if wiki_link:
                idm_ent["id"] = f"{main_person_id}-pl-{unquote(wiki_link.split('/')[-1])}"
            else:
                pl_ix += 1
                idm_ent["id"] = f"{main_person_id}-pl-{stringify_id(pl_ix)}"
        elif ner_category in ["ORG"]:
            idm_ent = copy.deepcopy(group_template)
            idm_ent["kind"] = "group"
            if wiki_link:
                idm_ent["id"] = f"{main_person_id}-gr-{unquote(wiki_link.split('/')[-1])}"
            else:
                gr_ix += 1
                idm_ent["id"] = f"{main_person_id}-gr-{stringify_id(gr_ix)}"
        elif ner_category in ["WORK_OF_ART"]:
            idm_ent = copy.deepcopy(object_template)
            idm_ent["kind"] = "cultural-heritage-object"
            if wiki_link:
                idm_ent["id"] = f"{main_person_id}-ob-{unquote(wiki_link.split('/')[-1])}"
            else:
                obj_ix += 1
                idm_ent["id"] = f"{main_person_id}-ob-{stringify_id(obj_ix)}"
            
        
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
                subj_idm_id = f"{main_person_id}-pr-{main_person_id}"
                ev_sub_id = idm_ent["id"]
                # This event is "passive" (the object "was created"), that's why the other entity we need to get is the Subj Entity (the creator)
                # We are assuming the first entity of the text == Subject of the Biography
                subj_idm_ent = idm_entity_dict[idm_id2univ_id[subj_idm_id]]
                event_info = {"full_event_id": f"{main_person_id}-{ev_sub_id}-ev-{stringify_id(event_ix)}", 
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
                ev_sub_id = idm_ent["id"] # .split("-")[1]
                subj_role =  rel_obj['relationValue']
                obj_role = inverse_relations_dict.get(rel_obj['relationValue'], 'unk')
                event_info = {"full_event_id": f"{main_person_id}-{ev_sub_id}-ev-{stringify_id(event_ix)}", 
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
    
    # ##Convert DATE <--> SRL Links Into (Main_Subject, predicate, object) Events. ALWAYS attached ot Main Subject
    subj_idm_id = f"{main_person_id}-pr-{main_person_id}"
    for tr in triples_with_events:
        subj_idm_ent = idm_entity_dict[idm_id2univ_id[subj_idm_id]]
        date, event_triple = tr[0], tr[1]
        ev_sub_id = subj_idm_ent["id"]
        start_date, end_date = normalize_date(date)
        if start_date:
            event_info = {"full_event_id": f"{main_person_id}-{ev_sub_id}-ev-{stringify_id(event_ix)}", 
                            "event_label": " ".join(event_triple), 
                            "event_kind": event_triple[1], 
                            "subj_role": event_triple[1], 
                            "obj_role": event_triple[1],
                            "startDate": start_date
                            }
        # else:
        #     print("FAILED DATE!", date)
        if end_date: event_info["endDate"] = end_date
        subj_idm_ent, idm_ent, event_obj, event_vocab, role_vocab = create_idm_event(event_info, subj_idm_id, subj_idm_ent, subj_idm_ent, event_vocab, role_vocab)
        # Add updated object back to dict
        idm_entity_dict[idm_id2univ_id[subj_idm_id]] = subj_idm_ent
        # Add Event to IDM Main Object
        parent_idm["events"].append(event_obj)
        event_ix += 1

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

def normalize_date(datelike_string: str) -> Tuple[str,str]:
    start_date, end_date = None, None
    if (len(datelike_string) == 4 or len(datelike_string) == 3) and (re.match(r"^\d{4}$", datelike_string) or re.match(r"^\d{3}$", datelike_string)):
        start_date = datetime.datetime(int(datelike_string),1,1).isoformat().split("T")[0]
    elif re.match(r"^\d{4}-\d{4}$", datelike_string) or re.match(r"^\d{3}-\d{4}$", datelike_string) or re.match(r"^\d{3}-\d{3}$", datelike_string):
        st, en = datelike_string.split("-")
        start_date = datetime.datetime(int(st),1,1).isoformat().split("T")[0]
        end_date = datetime.datetime(int(en),12,31).isoformat().split("T")[0]
    #elif 
    else:
        try:
            start_date = parser.parse(datelike_string).split("T")[0]
        except:
            pass
    return start_date, end_date


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
                "relations": [{ "entity": obj_idm_ent["id"], "role": f"role-{obj_role}" },
                              { "entity": subj_idm_id, "role": f"role-{subj_role}" }
                            ]
            }
    
    if event_info.get("startDate"):
        event_obj["startDate"] = event_info.get("startDate")
    if event_info.get("endDate"):
        event_obj["endDate"] = event_info.get("endDate")

    event_vocab[f"event-kind-{event_kind}"] = {"id": f"event-kind-{event_kind}", "label": {"default": f"{obj_role}"}}
    role_vocab[f"role-{subj_role}"] = { "id": f"role-{subj_role}", "label": { "default": subj_role}}
    role_vocab[f"role-{obj_role}"] = { "id": f"role-{obj_role}", "label": { "default": obj_role}}

    return subj_idm_ent, obj_idm_ent, event_obj, event_vocab, role_vocab



def get_smart_srl_triples(srl_list: Dict, last_name: str) -> List[Tuple]:
    all_triples = []
    for predicate_word, args in srl_list.items():
        args_dict = {arg[1]: arg[0] for arg in args} # {arg_label: arg_surfaceForm}
        print(f"\n===== {predicate_word} ---> {list(args_dict.keys())} =====")
        # SRL Triple Construction
        # Agent = A0 + Patient = [A1 | A2 | A3]
        main_agent = None
        if "ARG0" in args_dict:
            main_agent = args_dict["ARG0"]
            main_patient = []
            if "ARG1" in args_dict:
                main_patient.append(args_dict["ARG1"])
            if "ARG2" in args_dict:
                main_patient.append(args_dict["ARG2"])
            if "ARG3" in args_dict:
                main_patient.append(args_dict["ARG3"])
            if "ARG4" in args_dict:
                main_patient.append(args_dict["ARG4"])
        elif "ARG1" in args_dict:
            main_agent = args_dict["ARG1"]
            main_patient = []
            if "ARG2" in args_dict:
                main_patient.append(args_dict["ARG2"])
            if "ARG3" in args_dict:
                main_patient.append(args_dict["ARG3"])
            if "ARG4" in args_dict:
                main_patient.append(args_dict["ARG4"])
        elif "ARG2" in args_dict:
            main_agent = args_dict["ARG2"]
            main_patient = []
            if "ARG3" in args_dict:
                main_patient.append(args_dict["ARG3"])
            if "ARG4" in args_dict:
                main_patient.append(args_dict["ARG4"])

        if main_agent:
            # Simplify Agent (Basic parenthesis Rule)
            if "(" in main_agent and ")" in main_agent:
                start_par = main_agent.index("(")
                main_agent = main_agent[:start_par]
            # Put Main Patient Together
            main_patient = " ".join(main_patient)
            if "(" in main_patient and ")" in main_patient:
                start_par = main_patient.index("(")
                main_patient = main_patient[:start_par]
            # Negate predicate if necessary
            
            if "ARGM-NEG" in args_dict:
                current_triple = [main_agent, f"{args_dict['ARGM-NEG']} {predicate_word}", ""]
            else:
                current_triple = [main_agent, predicate_word, ""]
            # Complement
            complement = main_patient
            if "ARGM-TMP" in args_dict and len(args_dict["ARGM-TMP"]) > 0:
                complement += " " + args_dict["ARGM-TMP"]
            if "ARGM-CAU" in args_dict and len(args_dict["ARGM-CAU"]) > 0:
                complement += " " + args_dict["ARGM-CAU"]
            if "ARGM-ADV" in args_dict and len(args_dict["ARGM-ADV"]) > 0:
                complement += " " + args_dict["ARGM-ADV"]
            if "ARGM-GOL" in args_dict and len(args_dict["ARGM-GOL"]) > 0:
                complement += " " + args_dict["ARGM-GOL"]
            if "ARGM-LOC" in args_dict and len(args_dict["ARGM-LOC"]) > 0:
                complement += " " + args_dict["ARGM-LOC"]
            current_triple[2] = complement
            all_triples.append(current_triple)
    
    valid_triples = []
    for trip in all_triples:
        agent_str = trip[0].lower()
        if agent_str == "he" or agent_str == "she" or last_name.lower() in agent_str or  "his" in agent_str or "her" in agent_str and len(trip[2]) > 0:
            valid_triples.append(trip)
    print(valid_triples)
    return valid_triples


if __name__ == "__main__":
    convert_nlp_to_idm_json("data/json/albrecht_dürer.json", "data/idm/albrecht_dürer.idm.json")
    # convert_nlp_to_idm_json("english/data/json/ida_laura_pfeiffer.json", "english/data/idm/ida_pfeiffer.idm.json")