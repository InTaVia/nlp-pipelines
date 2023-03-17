"""
    This Script works separately from the basic pipeline environment and assumes 2 things:
        1) The basic pipelne already parsed the texts and constructed the JSON files 
        2) Dutch Coref is already installed and running. This to download and follow the instructions in: https://github.com/andreasvc/dutchcoref
    The script will add a "coref" layer to the existing JSON
"""
import csv, json
from typing import List, Dict, Any
from collections import defaultdict
import glob, os
import subprocess

## TODO: THIS CODE NEEDS ADAPTING (CURRENTLY IS JUST COPY PASTE FROM BIONET)

def main():
    # COREF_BASEPATH = "/home/jdazaareva2/data/Repos/my-vu-experiments/BiographyNet/coref_tmp_dir/"
    COREF_BASEPATH = "dutch/data/coref_info"
    JSON_BASEPATH = "dutch/data/json/*"
    for in_file in glob.glob(JSON_BASEPATH):
        process_and_generate_coref_files(in_file, COREF_BASEPATH)
        ### This step adds a field to the original json line and saves it in a new file
        out_file = f"{in_file.split('.')[0]}.with_coref.jsonl"
        add_coref_info(in_file, "dutch/data/json", out_file, COREF_BASEPATH, "dutchcoref")
	

def process_and_generate_coref_files(biography_json_path: str, coref_basepath: str):
	os.mkdir(coref_basepath) # 'coref_tmp_dir'
	skept = 0
	with open(biography_json_path) as f:
		for i, line in enumerate(f.readlines()):
			row = json.loads(line)
			current_path = f"coref_tmp_dir/{row['id_composed']}"
			os.makedirs(current_path, exist_ok=True)
			# Get SRL from Pipeline
			text_filename = f"{current_path}/example.txt"
			with open(text_filename, "w") as f:
				f.write(row['text_clean'])
			filepath = f"{current_path}/example.tok"
			output_coref = f"{current_path}/{row['id_composed']}"
			# Run Dutch Coref based on the generated files
			subprocess.run(["sh","run_coref_dutch.sh", text_filename, filepath, current_path, output_coref])
			print(i)
			# if i == 1:
			# 	break
	print(f"Skept {skept} in total")


def add_coref_info(biography_json: List[str], path_prefix: str, output_json: str, coref_root: str, coref_model_name: str):
	with open(biography_json) as f:
		with open(output_json, "w") as fout:
			for i, line in enumerate(f.readlines()):
				row = json.loads(line)
				coref_dir = f"{coref_root}/{row['id_composed']}"
				row['coreference'] = jsonify_coref_output(coref_dir, row['id_composed'], coref_model_name)
				# Write to new file
				fout.write(json.dumps(row)+"\n")
				json.dump(row, open(f"{path_prefix}/{row['id_composed']}.with_coref.json"), indent=2)


def jsonify_coref_output(coref_main_dir: str, bio_id: str, coref_model_name: str) -> List[Dict[str, Any]]:
	if bio_id == "99999999_99": return []
	mentions = csv.DictReader(open(f"{coref_main_dir}/{bio_id}.mentions.tsv"), delimiter="\t")
	conll_coref = read_conll_coref(f"{coref_main_dir}/{bio_id}.conll")
	bionet_json = json.load(open(f"{coref_main_dir}/{bio_id}.json"))
	print(f"Processing {bionet_json['id_composed']}")
	conll_coref_text = [x['text'] for x in conll_coref]
	conll2json_tokens = align_token_sequences(conll_coref_text, bionet_json['text_tokens'])
	clusters = get_coref_clusters(conll_coref, mentions, bionet_json['text_token_objects'], conll2json_tokens)
	# Filter and Clean Clusters to get the most relevant Persons and Locations references
	clean_clusters = []
	for id, mentions in clusters.items():
		person_mentions, location_mentions = [], []
		for men in mentions:
			if men['is_human'] or men['entity_type'] == 'PER':
				person_mentions.append(men)
			elif men['entity_type'] == 'LOC':
				location_mentions.append(men)
		all_mentions = person_mentions+location_mentions
		if len(all_mentions) > 0:
			# print(f"------------ {id} ------------")
			for men in all_mentions:
				# print(men)
				clean_clusters.append({
					"cluster_id": id,
					"mention_id": len(clean_clusters),
					"mention_type": men["mention_type"],
					"category": men["entity_type"],
					"surfaceForm": men["text"],
					"locationStart": men["char_start"],
					"locationEnd": men["char_end"],
					"tokenStart": men["token_start"],
					"tokenEnd": men["token_end"],
					"method": coref_model_name
				})
	return clean_clusters


def read_conll_coref(filepath: str) -> List[Dict[str, Any]]:
	tokens = []
	sent_ix, doc_tok_ix = 0, 0
	with open(filepath) as f:
		for line in f:
			row = line.strip().split()
			if line.startswith("#"):
				continue
			elif len(row) == 0:
				sent_ix += 1
			else:
				tok_text = row[3]
				tok_id = int(row[2])
				# tok_cluster = row[-1]
				tokens.append({'sent_id': sent_ix, 'token_sent_id':tok_id, 'token_doc_id': doc_tok_ix,'text': tok_text})
				doc_tok_ix += 1
		return tokens


def ordered_unique(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

def align_token_sequences(reference_tokens: List[str], other_tokens: List[str]) -> Dict[int, int]:
	# Create sequences including their respective indices
	reference_tokens = [(i, x) for i,x in enumerate(reference_tokens)]
	other_tokens = [(i, x) for i,x in enumerate(other_tokens)]
	window_size = 2
	# Keep track of the tokens that are already aligned and a list to possibly add weaker connections
	matched, missed = [], []
	for i, ref_tok in enumerate(reference_tokens):
		for oth_tok in other_tokens[i:i+window_size]:
			if ref_tok[1] == oth_tok[1]:
				matched.append((ref_tok, oth_tok))
				break
			else:
				matched.append((ref_tok, None))
				missed.append(oth_tok)
	matched = ordered_unique(matched)
	missed = ordered_unique(missed)
	# Add at the end of the sequence all of the trailing reference tokens (to recover the full reference token list)
	if len(reference_tokens) > len(other_tokens):
		for elem in reference_tokens[len(other_tokens):]:
			matched.append((elem, other_tokens[-1]))
	# Here we Generate a parallel sequences included weaker (missed) connections to fill-in the gaps between the token matches
	normalized = []
	missed_ix, latest_other_ix = 0, 0
	for i, (ref_tok, pair_tok) in enumerate(matched):
		if not pair_tok:
			for m in missed[:5]:
				m_ix, m_txt = m
				ref_txt = ref_tok[1]
				if (ref_txt.startswith(m_txt) or m_txt.startswith(ref_txt)) and m_ix >= latest_other_ix:
					normalized.append((ref_tok, m))
					missed.pop(0)
		else:
			normalized.append((ref_tok, pair_tok))
			latest_other_ix = pair_tok[0]
	# We use the normalized sequence to return the dictionary of indices. Both Indices are ALWAYS 0-based or 1-based
	ref2other = {}
	for cotok, jtok in ordered_unique(normalized):
		# print(f"{cotok} ---> {jtok}")
		ref2other[cotok[0]] = jtok[0]
	return ref2other
    

def get_coref_clusters(conll: List[Dict[str, Any]], mentions: csv.DictReader, json_token_objects: List[Dict[str, Any]], conll2json: Dict[int, int]) -> List[Dict[str, Any]]:
	clusters = defaultdict(list)
	json_tokens_text = [x['text'] for x in json_token_objects]
	for men in mentions:
		cluster_id = int(men['cluster'])
		token_start = int(men['start']) - 1
		token_end = int(men['end'])
		conll_mention = " ".join([x['text'] for x in conll[token_start:token_end]])
		json_token_start = conll2json.get(token_start)
		json_token_end = conll2json.get(token_end)
		if json_token_start and json_token_end:
			pass
		elif json_token_start and not json_token_end:
			json_token_end = token_end
		elif json_token_end and not json_token_start:
			json_token_start = token_start
		else:
			# print("FAIL!", token_start, token_end, men)
			continue
		json_mention = " ".join(json_tokens_text[json_token_start:json_token_end])
		char_start = json_token_objects[json_token_start]['start_char']
		char_end = json_token_objects[json_token_end-1]['end_char']
		assert conll_mention == men['text']
		is_human = True if men['human'] == '1' else False
		# print(men['text'], "<---r--->", json_mention, f"{token_start}=={json_token_start}", f"{token_end}=={json_token_end}")
		clusters[cluster_id].append({
			'text': json_mention, 
			'token_start': json_token_start, 
			'token_end': json_token_end, 
			'char_start': char_start,
			'char_end': char_end,
			'mention_type': men['type'],
			'entity_type': men['neclass'], 
			'is_human': is_human,
			'gender': men['gender'],
			'number': men['number']
			})
	return clusters


if __name__ == '__main__':
	main()