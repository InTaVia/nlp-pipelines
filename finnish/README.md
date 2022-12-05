# The Finnish NLP tools for annotating biographies

## Tools

The following tools have been used to annotate the Finnish texts.

### Nelli

### Name-finder

### Reksi

[Reksi service](https://github.com/SemanticComputing/reksi) is a tool and an API endpoint for named entity recognition and linking using a tool that consists of numerous regular expressions used to identify different information from text. The tool can also be used to link entities to corresponding ontologies and vocabularies provided that the user has predefined them. The service accepts text input, identifies named entities, and returns a resultset in JSON format. The resultset contains annotated text and a list of named entities, their types, locations, and optionally links to existing ontologies.

Currently the tool is able to identify named entities from Finnish texts. The types of entities it can identify currently are references to dates (currently hardcoded between 1100 and 2090), social security numbers, numerous registry numbers, URLs, email addresses, phone numbers, measure units, money and currencies, IP addresses, statutes, directives, and their sections and clauses. More entity classes can be added via configuration file where each type of entity has a class name in square brackets and it is followed by adding a variable pattern that is set by user definable regular expression.

The application executes each regular expression and collects entities they find from each sentence into the resultset. Before the resultset is transformed into JSON, the entities are disambiguated in favor of the longest match. This means that if there are several entities with overlapping locations, the longest matching entity is chosen by default. (The linking can enable better disambiguation but that will be considered later.)

### Gender guessing service

The [Gender identification service](https://github.com/SemanticComputing/gender-guessing-service) guesses genders based on statistical knowledge on names. It utilizes [Person name ontology (HENKO)](https://version.aalto.fi/gitlab/seco/suomen-henkilonimisto) to calculate to a given name a possibility it is a male or female name depending on the usage of the given name on women and men that is described in the person name ontology. In the end, the application gives calculus results for the user.

### Henko

The [Person name ontology (HENKO)](https://version.aalto.fi/gitlab/seco/suomen-henkilonimisto) is a vast collection of Finnish person names. It consists of given and family names, their usage statistics, matronyms, patronyms, nobiliary particles, etc. from the 3rd century to present time. It contains approximately 54 000 person name records from the The Finnish Digital Agency, BiographySampo, Norssi High School Alumni, and AcademySampo. The data can be be brwoser using the [ONKI Light service](https://light.onki.fi/henko/).

### Wrappers

The rest of the tools have been published and produce data for Finnish language texts. They have been wrapped to use the common dataformat schema.

#### Finer

#### FinBER NER model

#### Turku neural parser / Turku dependency parser

#### LAS (Lexical analysis service)
