# Databricks notebook source
!pip install InstructorEmbedding==1.0.1
!pip install sentence_transformers==2.2.2
!pip install --upgrade huggingface_hub==0.20.3

# COMMAND ----------

from InstructorEmbedding import INSTRUCTOR
import torch
torch.cuda.empty_cache()

# COMMAND ----------

"""
NAME : InstructorModel

DESCRIPTION:
This module serves the below functionalities:
                CREATES InstructorModel and intializes its parameters
                PREDICTION
              
"""

class InstructorModel:
  def __init__(self,model_address,context_length,instruction, device = 'cuda'):
    """Initialize the attributes of the instructor model object"""
    self.model_address=model_address
    self.model=INSTRUCTOR(self.model_address)
    self.model.max_seq_length = context_length
    self.model.tokenizer.model_max_length = context_length
    self.instruction =  instruction
    self.model.to('cuda')
    self.model.eval()
    self.device = device
     
    
  def predict(self, sents):
    """
    This block feeds inputs to model and returns the predicted embedding
    
    Parameters:
    argument1 (list): list of sentences
  
    Returns:
    embeddings
    
    """
    inputs = [[self.instruction, sent] for sent in sents]
    with torch.no_grad():
      embeddings = self.model.encode(inputs, device=self.device)
    return embeddings
      
  
    
  
