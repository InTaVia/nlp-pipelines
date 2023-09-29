#!/usr/bin/env bash
source ~/anaconda3/etc/profile.d/conda.sh

# Choose a name of Interest
name="Ida Pfeiffer"

# Get Wikipedia Article
conda activate intavia_flair
python en_get_wikipedia_articles.py --name "$name"

# Run Flair Pipeline
echo "Running Flair Pipeline..."
python en_text_to_json_flair.py "data/wikipedia"

# Run AllenNLP Pipeline
conda activate intavia_allen
echo "Running AllenNLP Pipeline..."
python en_text_to_json_allen.py --from_flair_json --path "data/wikipedia"

# Generate Structured Data File
echo "Generating IDM JSON..."
python nlp_to_idm_json.py --name "$name"