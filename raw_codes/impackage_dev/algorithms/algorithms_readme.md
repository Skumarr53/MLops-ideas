# Algorithms: Overview
This project is an NLP (Natural Language Processing) setup in Databricks, designed to perform various text analysis tasks. The project includes several modules for different functionalities such as instructor model creation, psycholinguistic methods, statistical analysis, and topic modeling.

- [Algorithms: Overview](#algorithms-overview)
  - [Folder Structure](#folder-structure)
  - [Modules](#modules)
    - [InstructorModel](#instructormodel)
    - [PsycholinguisticMethods](#psycholinguisticmethods)
    - [Statistics](#statistics)
    - [TopicXModel](#topicxmodel)


## Folder Structure
The project is organized into the following main modules:

- **InstructorModel**: Handles model initialization and prediction.
- **PsycholinguisticMethods**: Provides methods for psycholinguistic analysis such as LM analysis, FOG analysis, and polarity scoring.
- **Statistics**: Offers statistical functionalities including dollar amount extraction, numerical value extraction, and sentiment score calculation.
- **TopicXModel**: Manages topic modeling and relevance scoring.

## Modules

### InstructorModel
- **File**: `impackage_dev/algorithms/instructor_model.py`  
This module is responsible for creating and initializing the InstructorModel and making predictions.

    **Key Methods:**
    - `__init__(self, model_address, context_length, instruction, device='cuda')`: Initializes the InstructorModel with the given parameters.
    - `predict(self, sents)`: Feeds inputs to the model and returns the predicted embeddings.

### PsycholinguisticMethods
- **File**: `impackage_dev/algorithms/psycholinguistic_methods.py`  
This module provides various psycholinguistic methods for text analysis.

    **Key Methods:**
    - `LM_analysis_per_section(self, text_list)`: Analyzes text using the Lougrhan and McDonald dictionary-based method.
    - `fog_analysis_per_section(self, text_list)`: Generates the fog index for the input text.
    - `polarity_score_per_section(self, text_list)`: Generates the polarity score for the input text to identify sentiment.

### Statistics
- **File**: `impackage_dev/algorithms/statistics.py`  
This module offers statistical functionalities for text analysis.

    **Key Methods:**
    - `get_dollar_amounts(self, text_list)`: Reads the dollar amount from the input text.
    - `get_numbers(self, text_list)`: Extracts numerical values from the given string.
    - `combine_sent(self, x, y)`: Combines sentiment scores.

### TopicXModel
- **File**: `impackage_dev/algorithms/topicx_model.py`  
This module is responsible for topic modeling and relevance scoring.

**Key Methods:**
- `get_match_set(self, matches)`: Generates the match set for unigrams and bigrams.
- `match_count(self, text, match_sets, phrases=True)`: Generates the count dictionary with matched counts of unigrams and bigrams.
- `generate_match_count(self, currdf, word_set_dict)`: Generates the match count using the topics data.
- `generate_topic_statistics(self, currdf, word_set_dict)`: Generates new columns with topic total count stats.
- `generate_sentence_relevance_score(self, currdf, word_set_dict)`: Generates relevance scores for sentences.