- [ ] Update the existing word_tokenize function to exclude words specified by a parameter. below functions tokenize text
   - *centralized_nlp_package\preprocessing\text_preprocessing.py\tokenize_and_lemmatize_text* (mainly focusing on customizing token.pos_ not in ['NUM', 'SYM'] and token.ent_type_ not in ['PERSON', 'GPE', 'LOC', 'ORG'])
   - *centralized_nlp_package\text_processing\text_utils.py\word_tokenizer* (This is for psycholinguistic code)
- [x] Filter a DataFrame by eliminating rows where the specified column includes any of the given keywords. 
       *centralized_nlp_package/utils/helpers.py/df_remove_rows_with_keywords.*
- [x] Add a function to apply pandas row transofmration group of functions, 
- [ ] add a parameter as list as to what columns need to be computedfor topic report generation