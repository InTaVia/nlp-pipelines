# nlp-pipelines
Scripts for different languages to go form text to structured data

### In A Nutshell

The **english** section of this repository contains two separate cooperative pipelines. You can run them independently (in that case you just need to install one of the environments), or, if you install both environments and you want to combine outputs in a single JSON, you must run first the Flair pipeline and then the AllenNLP pipeline.

Each pipeline predicts different NLP layers:

1. Flair Pipeline
    * Tokenization
    * Sentence Splitting
    * Named Entity Recognition
    * Semantic Frame Disambiguation
    * Relation Exraction
    * Entity Linking
    * Includes SpaCy 3.5


2. AllenNLP Pipeline
    * Tokenization
    * Sentence Splitting
    * Named Entity Recognition
    * Semantic Role Labeling
    * Correference Resolution
    * Includes SpaCy 3.2


### Quick Start

1. Install everything necessary for the Flair Pipeline. To do this, run the following commands in the project root directory:
```
conda create -n intavia_flair python=3.10
conda activate intavia_flair
cd english
pip install -r requirements_flair.txt
```

2. Install Heideltime. This is a bit complicated so refer to the file `english/install_heideltime.md`. In case you cannot install heideltime, the pipelines can still work (without predicting time expressions of course), you will just need to comment the imports of that library.

3. If you also want to use the AllenNLP pipeline, install everything necessary for it in a **separate environment**. Run the following commands in the project root directory:
```
conda create -n intavia_allen python=3.7.16
conda activate intavia_allen
cd english
pip install -r requirements_allennlp.txt
```

You might also want to install python-heideltime in this environment. If you already installed it for the flair environment all you need to do is go to the `python-heideltime` directory and run (with the intavia_allen environment activated):

```
python3 -m pip install .
```


### How to Run

1. **Obtain Wikipedia Articles**

You can see how to download Wikipedia Files (and run the code) in the notebook `english/make_wikipedia_lists.ipynb`. All you need is to create a file with the list of names (and, I known, birth_dates and death_dates) and then run the functions provided in the notebook. See the examples of lists included under the folder `english/resources/`.


2. **Run NLP Pipeline, you can choose:**

    a. **Flair Pipeline ONLY**

    ```
        conda activate intavia_flair
        python english/en_text_to_json_flair.py "english/data/wikipedia/your_custom_files/"
    ```

    b. **AllenNLP Pipeline ONLY**

    ```
        conda activate intavia_allen
        python english/en_text_to_json_allen.py --from_text --path "english/data/wikipedia/your_custom_files/"
    ```

    c. **BOTH Pipelines**

    ```
        conda activate intavia_flair
        python english/en_text_to_json_flair.py "english/data/wikipedia/your_custom_files/"
        conda activate intavia_allen
        python english/en_text_to_json_allen.py --from_flair_json --path "english/data/wikipedia/your_custom_files/"
    ```