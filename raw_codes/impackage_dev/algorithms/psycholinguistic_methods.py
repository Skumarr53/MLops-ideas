# Databricks notebook source
# MAGIC %run ./../preprocessor/dictionary_model_preprocessor

# COMMAND ----------

# MAGIC %run ./statistics

# COMMAND ----------

# MAGIC %run ../dataclasses/dataclasses

# COMMAND ----------

import numpy as np
from dataclasses import dataclass



# COMMAND ----------

"""
NAME : PsycholinguisticMethods

DESCRIPTION:
This module serves the Psycholinguistic Method functionalities:
                LM_analysis
                FOG_analysis
                Polarity Score
"""
class PsycholinguisticMethods(object):
  
    """Psycholinguistic Methods CLASS"""
    
    def __init__(self):
      """Initialize the attributes of the Psycholinguistic Methods object"""
      self.blob_obj = BlobStorageUtility()
      self.filecfg = BlobFilenameConfig()
      self.config = config
      self.litigious_words_List=self.get_words_list(self.filecfg.litigious_flnm)
      self.complex_words_list =self.get_words_list(self.filecfg.complex_flnm)
      self.uncertain_words_list = self.get_words_list(self.filecfg.uncertianity_flnm)
      self.syllables = self.get_syllable_count(self.filecfg.syllable_flnm)
      self.positive_words_List = self.get_words_list(self.filecfg.vocab_pos_flnm)
      self.negative_words_list = self.get_words_list(self.filecfg.vocab_neg_flnm)
      self.preprocess_obj=DictionaryModelPreprocessor()
      self.statistics_obj=Statistics()
    
    def get_blob_stroage_path(self, filname):
      return self.config.model_artifacts_path + self.config.preprocessing_words_path + filename
    
    def get_words_list(self,filename):
      file_path = self.get_blob_stroage_path(filename)
      return self.blob_obj.load_list_from_txt(file_path)
    
    def get_syllable_count(self, file_name):
      """Reads the syllable words from the blob storage"""
      file_path = self.get_blob_stroage_path(file_name)
      return self.blob_obj.read_syllable_count(file_path)
  
    def LM_analysis_per_section(self, text_list):
      """Analysing text using the Lougrhan and MCDonald dictionary based method.
      And returns the litigious, uncertainity, complex word count and their scores.

      Parameters:
      argument1 (list): text list

      Returns:
      tuple:(litigious words score(double),complex words score(double),uncertain words score(double),word count(int),litigious words count(int),complex words count(int),uncertain words count(int))

      """
      text, input_words, word_count = self.preprocess_obj.preprocess_text(text_list)
      
      if (text) and (word_count > 1):
        litigious_words_count = 0
        complex_words_count = 0
        uncertain_words_count = 0
        for input_word in input_words:
          if input_word in self.litigious_words_List:
              litigious_words_count +=1

          if input_word in self.complex_words_list:
              complex_words_count +=1

          if input_word in self.uncertain_words_list:
              uncertain_words_count +=1
          litigious_words_score = litigious_words_count / (word_count)
          complex_words_score = complex_words_count / (word_count)
          uncertain_words_score = uncertain_words_count / (word_count)
        return (litigious_words_score, complex_words_score, uncertain_words_score, word_count, litigious_words_count, complex_words_count, uncertain_words_count)

      else:
        return (np.nan , np.nan , np.nan , np.nan,  np.nan,  np.nan,  np.nan)
      
    def LM_analysis_per_sentence(self, text_list):
      """Analysing list of sentences using the Lougrhan and MCDonald dictionary based method.
      And returns the litigious, uncertainity, complex word count and their scores.

      Parameters:
      argument1 (list): text list

      Returns:
      tuple:(litigious words score(double),complex words score(double),uncertain words score(double),word count(int),litigious words count(int),complex words count(int),uncertain words count(int),litigious words count list(list),complex words count list(list),uncertain words count list(list))

      """
      text, input_words, word_count = self.preprocess_obj.preprocess_text(text_list)
      text_list, input_words_list, word_count_list = self.preprocess_obj.preprocess_text_list(text_list)
      litigious_words_count_list=[]
      complex_words_count_list=[]
      uncertain_words_count_list=[]

      if (len(text_list)) and (len(word_count_list) > 1):
        for index in range(len(input_words_list)):
          litigious_words_count_per_sentence = 0
          complex_words_count_per_sentence = 0
          uncertain_words_count_per_sentence = 0
          for input_word in input_words_list[index]:
            if input_word in self.litigious_words_List:
                litigious_words_count_per_sentence+=1

            if input_word in self.complex_words_list:
                complex_words_count_per_sentence+=1

            if input_word in self.uncertain_words_list:
                uncertain_words_count_per_sentence+=1
          litigious_words_count_list.append(litigious_words_count_per_sentence)
          complex_words_count_list.append(complex_words_count_per_sentence)
          uncertain_words_count_list.append(uncertain_words_count_per_sentence)
        return ( word_count_list,litigious_words_count_list,complex_words_count_list,uncertain_words_count_list)

      else:
        return (None , None , None , None)  
        
    def fog_analysis_per_section(self, text_list):
      """Generates the fog index for the input text.
      Fog index used to evaluate how easily some text can be read by its intended audience.

      Parameters:
      argument1 (list): text list

      Returns:
      double, int:returns fog index, complex word count, average word per sentence, total word count

      """
      if isinstance(text_list, list):
        raw_text = ' '.join(text_list)
      else:
        raw_text = text_list

      total_word_count = len(raw_text.split(' '))
      average_word_per_sentence = np.mean([len(sentence.strip().split(" ")) for sentence in raw_text.split('. ')])
      text, input_words, word_count = self.preprocess_obj.preprocess_text(text_list)

      if (text) and (word_count > 1):
        complex_word_count = 0
        for input_word in input_words:
          # check if word exists in our dictionary
          if(input_word in self.syllables.keys()):

            if (input_word.endswith('es')):

              # Check if root word is a many-syllable word - in this case excluding just 's' may yield root
              if (input_word[:-1] in self.syllables.keys()):

                if(self.syllables[input_word[:-1]] > 2):
                  complex_word_count +=1
                  continue
                else:
                  continue

              # Check if root word is a many-syllable word - in this case excluding 'es' may yield root       
              elif (input_word[:-2] in self.syllables.keys()):

                if(self.syllables[input_word[:-2]] > 2):
                  complex_word_count +=1
                  continue
                else:
                  continue  

            if(input_word.endswith('ing')):

              # Excluding 'ing' may yield root 
              if (input_word[:-3] in self.syllables.keys()):

                if(self.syllables[input_word[:-3]] > 2):
                  complex_word_count +=1
                  continue
                else:
                  continue

            if(input_word.endswith('ed')):

              if (input_word[:-1] in self.syllables.keys()):

                if(self.syllables[input_word[:-1]] > 2):
                  complex_word_count +=1
                  continue
                else:
                  continue

              elif (input_word[:-2] in self.syllables.keys()):

                if(self.syllables[input_word[:-2]] > 2):
                  complex_word_count +=1
                  continue
                else:
                  continue

            # In case no recognized suffix is added            
            if(self.syllables[input_word] > 2):
              complex_word_count +=1

        fog_index = 0.4 * (average_word_per_sentence + 100 * (complex_word_count/total_word_count))
        return (fog_index, complex_word_count, average_word_per_sentence, total_word_count)

      else:
        return (np.nan, np.nan, np.nan, np.nan)

    def fog_analysis_per_sentence(self, text_list):
      """Generates the fog index for the input text which is a list of sentences.
      Fog index used to evaluate how easily some text can be read by its intended audience.

      Parameters:
      argument1 (list): text list

      Returns:
      tuple:(fog_index(double), complex_word_count(int), average_word_per_sentence(int), total_word_count(int))
      """
      total_word_count = np.sum([len(sentence.split(" ")) for sentence in text_list])
      total_word_count_list = [len(sentence.split(" ")) for sentence in text_list]
      average_word_per_sentence = np.mean([len(sentence.split(" ")) for sentence in text_list])
      text, input_words, word_count = self.preprocess_obj.preprocess_text(text_list)
      text_list, input_words_list, word_count_list = self.preprocess_obj.preprocess_text_list(text_list)

      if (len(text_list)) and (len(word_count_list) > 1):
        complex_word_count_list=[]
        fog_index_list=[]
        for index in range(len(input_words_list)):
          complex_words_count_per_sentence = 0
          word_count_per_sentence=0
          for input_word in input_words_list[index]:
            # check if word exists in our dictionary
            if(input_word in self.syllables.keys()):

              if (input_word.endswith('es')):

                # Check if root word is a many-syllable word - in this case excluding just 's' may yield root
                if (input_word[:-1] in self.syllables.keys()):

                  if(self.syllables[input_word[:-1]] > 2):
                    complex_words_count_per_sentence+=1
                    continue
                  else:
                    continue

                # Check if root word is a many-syllable word - in this case excluding 'es' may yield root       
                elif (input_word[:-2] in self.syllables.keys()):

                  if(self.syllables[input_word[:-2]] > 2):
                    complex_words_count_per_sentence+=1
                    continue
                  else:
                    continue  

              if(input_word.endswith('ing')):

                # Excluding 'ing' may yield root 
                if (input_word[:-3] in self.syllables.keys()):

                  if(self.syllables[input_word[:-3]] > 2):
                    complex_words_count_per_sentence+=1
                    continue
                  else:
                    continue

              if(input_word.endswith('ed')):
                if (input_word[:-1] in self.syllables.keys()):
                  if(self.syllables[input_word[:-1]] > 2):
                    complex_words_count_per_sentence+=1
                    continue
                  else:
                    continue

                elif (input_word[:-2] in self.syllables.keys()):

                  if(self.syllables[input_word[:-2]] > 2):
                    complex_words_count_per_sentence+=1
                    continue
                  else:
                    continue

              # In case no recognized suffix is added            
              if(self.syllables[input_word] > 2):
                complex_words_count_per_sentence+=1
          word_count_per_sentence=total_word_count_list[index]
          complex_word_count_list.append(complex_words_count_per_sentence)
          fog_index_list.append(0.4*(word_count_per_sentence+100*(complex_words_count_per_sentence/word_count_per_sentence)))
        return (fog_index_list, complex_word_count_list,total_word_count_list )

      else:
        return (None, None, None)

    def polarity_score_per_section(self, text_list):
      """Generates the polarity score for the input text to identify the sentiment
      Parameters:
      argument1 (list): text list

      Returns:
      tuple:(polarity score(double), word count(int), sum negative(int), positive words count(int), legacy score(float))
      """
      text, input_words, word_count = self.preprocess_obj.preprocess_text(text_list)

      if (text) and (word_count > 1):
        positive_words_count = 0
        negative_words_count = 0 
        positive_words_list = []
        negative_words_list = []
        for i in range(0, word_count):
          if input_words[i] in self.negative_words_list:
            negative_words_count -=1
            negative_words_list.append(input_words[i])

          if input_words[i] in self.positive_words_List:

            if i >= 3:

              if self.preprocess_obj.word_negated(input_words[i - 1]) or self.preprocess_obj.word_negated(input_words[i - 2]) or self.preprocess_obj.word_negated(input_words[i - 3]):
                negative_words_count -=1
                negative_words_list.append(input_words[i] + ' (with negation)')
              else:
                positive_words_count += 1
                positive_words_list.append(input_words[i])

            elif i == 2:

              if self.preprocess_obj.word_negated(input_words[i - 1]) or self.preprocess_obj.word_negated(input_words[i - 2]):
                negative_words_count -=1
                negative_words_list.append(input_words[i] + ' (with negation)')
              else:
                positive_words_count += 1
                positive_words_list.append(input_words[i])

            elif i == 1:

              if self.preprocess_obj.word_negated(input_words[i - 1]):
                negative_words_count -=1
                negative_words_list.append(input_words[i] + ' (with negation)')
              else:
                positive_words_count += 1
                positive_words_list.append(input_words[i])

            elif i == 0:
              positive_words_count += 1
              positive_words_list.append(input_words[i])

        sum_negative = negative_words_count 
        sum_negative = sum_negative * -1

        polarity_score = (positive_words_count - sum_negative) / (word_count)
        legacy_score = self.statistics_obj.combine_sent(positive_words_count, sum_negative)
        return (polarity_score, word_count, sum_negative, positive_words_count, legacy_score)

      else:
        return (np.nan, np.nan, np.nan , np.nan, np.nan)
    
    def polarity_score_per_sentence(self, text_list):
      """Generates the polarity score for the input list of sentences to identify the sentiment
      Parameters:
      argument1 (list): text list

      Returns:
      tuple:(polarity score(double), word count(int), sum negative(int), positive words count(int), legacy score(float),positive words count list(list),negative words count list(list))
      """
      text, input_words, word_count = self.preprocess_obj.preprocess_text(text_list)
      text_list, input_words_list, word_count_list = self.preprocess_obj.preprocess_text_list(text_list)
      if (len(text_list)) and (len(word_count_list) > 1):
        positive_words_count_list =[]
        negative_words_count_list = [] 
        for index in range(len(input_words_list)):
          input_word_per_sentence=input_words_list[index]
          positive_words_count_per_sentence=0
          negative_words_count_per_sentence=0
          for word_index in range(len(input_word_per_sentence)):
            if input_word_per_sentence[word_index] in self.negative_words_list:
              negative_words_count_per_sentence-=1

            if input_word_per_sentence[word_index] in self.positive_words_List:

              if word_index >= 3:

                if self.preprocess_obj.word_negated(input_word_per_sentence[word_index - 1]) or self.preprocess_obj.word_negated(input_word_per_sentence[word_index - 2]) or self.preprocess_obj.word_negated(input_word_per_sentence[word_index - 3]):
                  negative_words_count_per_sentence-=1
                else:
                  positive_words_count_per_sentence+=1

              elif word_index == 2:

                if self.preprocess_obj.word_negated(input_word_per_sentence[word_index - 1]) or self.preprocess_obj.word_negated(input_word_per_sentence[word_index - 2]):
                  negative_words_count_per_sentence-=1
                else:
                  positive_words_count_per_sentence+=1

              elif word_index == 1:

                if self.preprocess_obj.word_negated(input_word_per_sentence[word_index - 1]):
                  negative_words_count_per_sentence-=1
                else:
                  positive_words_count_per_sentence+=1

              elif word_index == 0:
                positive_words_count_per_sentence+=1
          positive_words_count_list.append(positive_words_count_per_sentence)
          negative_words_count_list.append(negative_words_count_per_sentence*-1)

        return (word_count_list, positive_words_count_list,negative_words_count_list)

      else:
        return (None, None, None)
      
    def tone_count_with_negation_check(self, text_list):
      """
      Count positive and negative words with negation check. Account for simple negation only for positive words.
      Simple negation is taken to be observations of one of negate words occurring within three words
      preceding a positive words.
      Parameters:
      argument1 (list): text list

      Returns:
      array, int:returns polarity_scores, word_count, negative_word_count_list, positive_word_count_list, legacy_scores

      """ 

      polarity_scores = []
      legacy_scores = []
      word_count = []
      negative_word_count_list = []
      positive_word_count_list = []

      sentiment_metrics = self.polarity_score_per_section(text_list)

      polarity_scores.append(sentiment_metrics[0])
      word_count.append(sentiment_metrics[1])
      negative_word_count_list.append(sentiment_metrics[2])
      positive_word_count_list.append(sentiment_metrics[3])
      legacy_scores.append(sentiment_metrics[4])

      return (polarity_scores, word_count, negative_word_count_list, positive_word_count_list, legacy_scores)
      
    def tone_count_with_negation_check_per_sentence(self, text_list):
      """
      Count positive and negative words with negation check. Account for simple negation only for positive words.
      Simple negation is taken to be observations of one of negate words occurring within three words
      preceding a positive words.
      Parameters:
      argument1 (list): text list

      Returns:
      tuple:(polarity scores(float), word count(int), negative word count list(list), positive word count list(list), legacy scores(double)))

      """ 

      word_count = []
      positive_word_count_list_per_sentence=[]
      negative_word_count_list_per_sentence=[]
      sentiment_metrics = self.polarity_score_per_sentence(text_list)
      word_count.append(sentiment_metrics[0])
      positive_word_count_list_per_sentence.append(sentiment_metrics[1])
      negative_word_count_list_per_sentence.append(sentiment_metrics[2])

      return (word_count, positive_word_count_list_per_sentence,negative_word_count_list_per_sentence)
