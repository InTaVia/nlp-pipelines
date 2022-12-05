# The Finnish NLP tools for annotating biographies

## Tools

The following tools have been used to annotate the Finnish texts.

### Nelli

### Reksi

### Name-finder

### Gender guessing service

The [Gender identification service](https://github.com/SemanticComputing/gender-guessing-service) guesses genders based on statistical knowledge on names. It utilizes [Person name ontology (HENKO)](https://version.aalto.fi/gitlab/seco/suomen-henkilonimisto) to calculate to a given name a possibility it is a male or female name depending on the usage of the given name on women and men that is described in the person name ontology. In the end, the application gives calculus results for the user.

### Henko

The [Person name ontology (HENKO)](https://version.aalto.fi/gitlab/seco/suomen-henkilonimisto) is a vast collection of Finnish person names. It consists of given and family names, their usage statistics, matronyms, patronyms, nobiliary particles, etc. from the 3rd century to present time. It contains approximately 54 000 person name records from the The Finnish Digital Agency, BiographySampo, Norssi High School Alumni, and AcademySampo. The data can be be brwoser using the [ONKI Light service](light.onki.fi/henko/).

### Wrappers

The rest of the tools have been published and produce data for Finnish language texts. They have been wrapped to use the common dataformat schema.

#### Finer

#### FinBER NER model

#### Turku neural parser / Turku dependency parser

#### LAS (Lexical analysis service)
