# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook Description
# MAGIC This notebook is used to analyze our competitor's earnings call transcript pdf. We implement sentiment analysis on different sections and topic modeling on analyst's questions. The outputs include:
# MAGIC 1. CEO & CFO sentiment plot for current 4 quarters earnings calls (html.) or MD sentiment plot for current 4 quarters earnings calls (html.)
# MAGIC 2. QA sentiment plot for current 4 quarters earnings calls (html.)
# MAGIC 3. analyst questions' topic modeling bar plot (html.)
# MAGIC 4. sentiment score summary and detailed QA sentiment and topic summary (docx.) + with one sentence sentiment summary (manually or by Copilot)
# MAGIC
# MAGIC Cluster Selection: Please use 13.3 runtime GPU for this notebook. It will take ~ 2 mins to load the model.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Load packages

# COMMAND ----------

pip install requests==2.28.1

# COMMAND ----------

pip install python-docx==1.1.2

# COMMAND ----------

pip install PyMuPDF==1.24.9

# COMMAND ----------

pip install transformers==4.40.1

# COMMAND ----------

pip install safetensors==0.4.3

# COMMAND ----------

!pip install gensim==4.2.0
!pip install spacy==3.4.4
!pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.4.0/en_core_web_sm-3.4.0.tar.gz
!pip install dask distributed --upgrade
!pip install dask==2022.12.1

# COMMAND ----------

pip install accelerate==0.20.3

# COMMAND ----------

import seaborn as sns
import spacy
from spacy.lang.en import English
from transformers import TextClassificationPipeline
import pandas as pd
import numpy as np
import tqdm
from tqdm import tqdm
tqdm.pandas()
import re
from collections import Counter
import gc
import json

import ast
from IPython.display import clear_output
from dateutil.relativedelta import relativedelta
from datetime import datetime
from datetime import timedelta
import pickle 
from docx import Document

import requests
import fitz
import warnings
import plotly.graph_objects as go
import plotly.express as px
import textwrap
import io 
import sys
import contextlib
from shutil import copyfile
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
warnings.filterwarnings("ignore")

# Load model directly
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel, TextClassificationPipeline, ZeroShotClassificationPipeline
import torch

device = 0 if torch.cuda.is_available() else -1

# COMMAND ----------

spacy_tokenizer = spacy.load('en_core_web_sm')


# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Load the helper functions and NLI model

# COMMAND ----------

# Helper functions
def NLI_sent_total(each_row, labels_sent):
  max_key_dict = []
  if each_row != "":
    for i, row in enumerate(split_sentences(each_row)):
      if 'AlphaSense' not in row and 'All Rights Reserved' not in row and "All other trademarks mentioned belong to their respective owners" not in row:
        text1_md, text2_md = create_text_pair([row], inference_template, labels_sent)
        inference_result1_md = pl_inference1([f"{text1_md[i]}</s></s>{text2_md[i]}" for i in range(len(text1_md)) ], padding=True, top_k=None, batch_size = 16, truncation = True, max_length = 512)
        result = inference_summary1(text1_md, text2_md, inference_result1_md)
        sentiment_dict = {'positive': '', 'negative': '', 'neutral': ''}
        for dic in result:
          if 'positive' in dic['labels']:
            sentiment_dict['positive'] = dic['scores']['entailment']
          elif 'negative' in dic['labels']:
            sentiment_dict['negative'] = dic['scores']['entailment']
          else:
            sentiment_dict['neutral'] = dic['scores']['entailment']
        max_key = None
        max_value = float('-inf')
        for key, value in sentiment_dict.items():
          if value > max_value:
            max_value = value
            max_key = key
        max_key_dict.append(max_key)
    senti_counts = dict(Counter(max_key_dict))
    positive_sent =  senti_counts.get('positive', 0)
    negative_sent =  senti_counts.get('negative', 0)
    total_count = senti_counts.get('positive', 0) + senti_counts.get('negative', 0) + senti_counts.get('neutral', 0)
    sentiment = round((positive_sent - negative_sent) / total_count,2)
  else:
    sentiment = "No matched sentences in this question"
    total_count = 0
  return sentiment, total_count

def create_text_pair(transcript, inference_template, labels):
  template = inference_template + "{label}."
  text1, text2 = [], []
  for t in transcript:
      for l in labels:
          text1.append(t)
          text2.append(template.format(label=l))
  return text1, text2

def inference_summary1(text1, text2, inference_result):
  result_dict = []
  for i, sentence in enumerate(text1):
    sent_dict = {}
    sent_dict['sequence'] = sentence
    sent_dict['labels'] = text2[i]
    sent_dict['scores'] = {}
    for s in inference_result[i]:
      if s['label'] == 'entailment':
        sent_dict['scores']['entailment'] = s['score']
      elif s['label'] == 'not_entailment':
        sent_dict['scores']['not_entailment'] = s['score']
    result_dict.append(sent_dict)
  return result_dict

def wrap_text_with_page(txt, width):
  parts = re.split(r'(question \d+)', txt)
  wrapped_parts = []
  for i in range(1, len(parts), 2):
    page_header = parts[i]
    page_content = parts[i+1] if i + 1 < len(parts) else ''
    wrapped_content = textwrap.wrap(page_content, width = width)
    wrapped_parts.append(" <br> " + page_header + "<br>" + "<br>".join(wrapped_content))
  return "<br>".join(wrapped_parts)

def get_last_four_quarters():
    current_date = datetime.now() 
    current_year = current_date.year
    current_month = current_date.month

    # Determine the current quarter
    current_quarter = (current_month - 1) // 3 + 1

    quarters = []
    for i in range(1,5):
        quarter = current_quarter - i
        year = current_year
        if quarter <= 0:
            quarter += 4
            year -= 1
        quarters.append(f"{quarter}Q{str(year)[-2:]}")

    return quarters[::-1]  # Reverse to get the correct order

def extract_text_from_pdf(pdf_path): 
  # Open the PDF file 
  document = fitz.open(pdf_path) 
  text = '' 
  # Extract text from all pages 
  for page in document: 
    text += page.get_text() + '\n' 
  return text

def extract_participants(text):
  participants_re = re.compile(r'Company Participants(.*?)(?:Other Participants|MANAGEMENT DISCUSSION SECTION)', re.DOTALL)
  execs = {'CEO': [], 'CFO': []}
  participants_match = participants_re.search(text)
  if not participants_match:
    raise ValueError("Company Participants section not found")
  participants_section = participants_match.group(1).strip()
  participant_lines = participants_section.split('\n')
  participants = []
  participant_indices = []
  for i, line in enumerate(participant_lines):
    line = line.strip()
    if line:
      if len(re.split(r'\xa0-\xa0| - ', line)) >= 2:
        name = re.split(r'\xa0-\xa0| - ', line)[0].strip()
        participants.append(name)
        participant_indices.append(i)

  # reconstructs lines based on where the names are located to avoid errors caused my random/misplaced \n
  for i, index in enumerate(participant_indices):
    begin = index
    end = len(participant_lines) if i==len(participant_indices)-1 else participant_indices[i+1]
    reconstructed = ' '.join(participant_lines[begin:end])
    if 'Chief Financial Officer' in reconstructed:
      execs['CFO'].append(participants[i])
    if 'Chief Executive Officer' in reconstructed:
      execs['CEO'].append(participants[i])
  participants.append("Unverified Participant")
  return participants, execs

def split_management_discussion(text, participants, ceo, cfo):
  management_section_re = re.compile(r'MANAGEMENT DISCUSSION SECTION(.*?)QUESTION AND ANSWER SECTION', re.DOTALL)
  participant_re = re.compile(r'({})\s*\s*(.*)'.format('|'.join(re.escape(p) for p in participants)), re.IGNORECASE)
  exclude_re = re.compile(r'^(Analyst:|Operator:|Page\s*\d+|AlphaSense|©)')
  management_section_match = management_section_re.search(text)
  if not management_section_match:
    raise ValueError("Management Discussion Section not found")
  management_section = management_section_match.group(1).strip()
  speeches = {'CEO': '', 'CFO': '', 'other executives': ''}
  current_speaker = ''
  lines = management_section.split('\n')
  for line in lines:
    if exclude_re.match(line):
      continue
    match = participant_re.match(line)
    if match:
      speaker = match.group(1)
      text = match.group(2).strip() + ' '
      if any([e in speaker for e in ceo]):
        current_speaker = 'CEO'
      elif any([e in speaker for e in cfo]):
        current_speaker = 'CFO'
      else:
        current_speaker = 'other executives'
      speeches[current_speaker] += text
    else:
      if current_speaker:
        speeches[current_speaker] += line.strip() + ' '
  return speeches

def split_sentences(text):
  nlp = spacy.load("en_core_web_sm")
  doc = nlp(text)
  sentences = [sent.text for sent in doc.sents]
  return sentences

def parse_questions_and_answers_v1(text): 
  # Regular expressions to identify questions, answers, and exclude irrelevant lines 
  question_re = re.compile(r'Question\s*[-:]?\s*(.*)') #, re.IGNORECASE
  answer_re = re.compile(r'Answer\s*[-:]?\s*(.*)') #, re.IGNORECASE
  exclude_re = re.compile(r'^(Analyst:|Page\s*\d+|AlphaSense|©)')
  operator_re = re.compile(r'^Operator$')
  lines = text.split('\n') 
  # print(lines)
  qa_list = [] 
  current_qa = {'question': '', 'answer': ''} 
  is_question = False 
  is_answer = False
  skip_lines = False
  for i,line in enumerate(lines):
    

    if operator_re.match(line):
      #operator line detected
      skip_lines = True 

    # Skip irrelevant lines 
    question_match = question_re.match(line) 
    answer_match = answer_re.match(line)

    if question_match or answer_match:
      skip_lines = False
    
    if exclude_re.match(line) or skip_lines: 
      continue 
    

    if question_match: 
      # print(question_match)
      if current_qa['question'] or current_qa['answer']: 
        # print(current_qa)
        qa_list.append(current_qa) 
        current_qa = {'question': '', 'answer': ''} 
      is_question = True 
      is_answer = False 
      current_qa['question'] += question_match.group(1).strip() + ' ' 
    elif answer_match: 
    
      is_question = False 
      is_answer = True 
      current_qa['answer'] += answer_match.group(1).strip() + ' '
    else: 
      if is_question: 
        current_qa['question'] += line.strip() + ' ' 
      elif is_answer: 
        current_qa['answer'] += line.strip() + ' ' 
  if current_qa['question'] or current_qa['answer']: 
    qa_list.append(current_qa) 
  return qa_list 

def MD_plot(df_md, df_qa, company, quarter):
  fig = make_subplots(rows=1, cols=2, subplot_titles=['                                                                               Sentiment Score: Maximum Positive Sentiment=1, Maximum Negative Sentiment=-1, Sentiment Range=[1,-1]', ""], shared_yaxes=True ) 
  plot_df_md = df_md
  for speaker in plot_df_md.index:
    if speaker == 'CEO':
      color = px.colors.qualitative.Set3[8] #'dodgerblue'
    elif speaker == 'CFO':
      color = px.colors.qualitative.Set3[0]
    else:
      color = px.colors.qualitative.Set3[5] #'orange'
    fig.add_trace(go.Bar(x=plot_df_md.columns, y=plot_df_md.loc[speaker], name = speaker, marker=dict(color = color),hoverlabel = dict(namelength = -1)),row = 1, col = 1)

  plot_df_qa = df_qa
  for speaker in plot_df_qa.index:
    if speaker == 'Questions':
      color = px.colors.qualitative.Set3[1] #'dodgerblue'
    else:
      color = px.colors.qualitative.Set3[2]
    fig.add_trace(go.Bar(x=plot_df_qa.columns, y=plot_df_qa.loc[speaker], name = speaker, marker=dict(color = color),hoverlabel = dict(namelength = -1)),row = 1, col = 2)  

  fig.update_layout(title_text = company_name + " Earnings Call " + report_quarter + " Analysis", showlegend=True)
  fig.update_layout(height = 700, width = 1300)
  fig.update_layout(barmode='group', bargap=0.50,bargroupgap=0.0)
  fig.update_yaxes(title_text = 'MD', row=1, col=1)
  fig.update_yaxes(title_text = 'QA', row=1, col=2)

  fig.update_layout(
      font_color="tan",
      title_font_color="tan",
      legend_title_font_color="tan"
  )
  fig.update_layout(plot_bgcolor='rgb(50,50,50,50)', paper_bgcolor= 'black')
  fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='darkslategray')
  fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='dimgray')
  fig.update_layout(yaxis_range=[-1.2,1.2])
  fig.show()

  return fig

def sentiment_plot(df_section, section, company, quarter):
  fig = make_subplots(rows=1, cols=1, subplot_titles=['Sentiment Score: Maximum Positive Sentiment=1, Maximum Negative Sentiment=-1, Sentiment Range=[1,-1]'], shared_yaxes=True ) 
  if section == 'MD':

    plot_df_md = df_section
    for speaker in plot_df_md.index:
      if speaker == 'CEO':
        color = px.colors.qualitative.Set3[8] #'dodgerblue'
      elif speaker == 'CFO':
        color = px.colors.qualitative.Set3[0]
      else: 
        color = px.colors.qualitative.Set3[5] #'orange'
      fig.add_trace(go.Bar(x=plot_df_md.columns, y=plot_df_md.loc[speaker], name = speaker, marker=dict(color = color),hoverlabel = dict(namelength = -1)),row = 1, col = 1)
    fig.update_yaxes(title_text = 'MD', row=1, col=1)
    fig.update_layout(title_text = company_name + " Earnings Call " + report_quarter + " Analysis - Management Discussion", showlegend=True)

  elif section == 'QA':

    plot_df_qa = df_section
    for speaker in plot_df_qa.index:
      if speaker == 'Questions':
        color = px.colors.qualitative.Set3[1] #'dodgerblue'
      else:
        color = px.colors.qualitative.Set3[2]
      fig.add_trace(go.Bar(x=plot_df_qa.columns, y=plot_df_qa.loc[speaker], name = speaker, marker=dict(color = color),hoverlabel = dict(namelength = -1)),row = 1, col = 1)  
    fig.update_yaxes(title_text = 'QA', row=1, col=1)
    fig.update_layout(title_text = company_name + " Earnings Call " + report_quarter + " Analysis - Q&A", showlegend=True)

  
  fig.update_layout(height = 700, width = 1200)
  fig.update_layout(barmode='group', bargap=0.50,bargroupgap=0.0)

  fig.update_layout(
      font_color="tan",
      title_font_color="tan",
      legend_title_font_color="tan"
  )
  fig.update_layout(plot_bgcolor='rgb(50,50,50,50)', paper_bgcolor= 'black')
  fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='darkslategray')
  fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='dimgray')
  fig.update_layout(yaxis_range=[-1.2,1.2])
  fig.show()

  return fig

def analyze_current_quarter(values, quarters):
  numeric_values = [(index, value) for index, value in enumerate(values) if isinstance(value, (int, float))]
  non_numeric_indices = [index for index, value in enumerate(values) if not isinstance(value, (int, float))] 
  if len(numeric_values) < 2:
    return None, non_numeric_indices, None, None
  last_index, last_value = numeric_values[-1]
  second_last_index, second_last_value = numeric_values[-2]
  last_quarter = quarters[last_index]
  second_last_quarter = quarters[second_last_index]
  sorted_values = sorted([value for _, value in numeric_values], reverse=True)
  rank = sorted_values.index(last_value) + 1
  change = last_value - second_last_value
  change_type = "increased" if change > 0 else "decreased" if change < 0 else "remained the same"
  return rank, non_numeric_indices, change, change_type, last_quarter, second_last_quarter

# COMMAND ----------

# NLI sentiment labels
labels_sent = ['This question has the positive sentiment', 'This question has the negative sentiment', 'This question has the neutral sentiment']

# COMMAND ----------

# Loadd NLI model
classification_model_name = "/dbfs/mnt/access_work/UC25/Libraries/HuggingFace/deberta-v3-large-zeroshot-v2"

tokenizer_1 = AutoTokenizer.from_pretrained(classification_model_name)
model_1 = AutoModelForSequenceClassification.from_pretrained(classification_model_name)

inference_template = ""
pl_inference1 = pipeline(task="text-classification", model = model_1, tokenizer = tokenizer_1, device = device)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Rename and upload the pdf
# MAGIC 1. Rename the new earnings call shared by IR with following format: **{company acronym}_{report quarter}Q{report year}_Transcript.pdf** (such as: AB_4Q24_Transcript.pdf)
# MAGIC 2. Upload it to the blob
# MAGIC
# MAGIC

# COMMAND ----------

historical_quarter = get_last_four_quarters()
company_name = dbutils.widgets.get("Company Name")
company_acronym = dbutils.widgets.get("Company Acronym")
report_quarter = historical_quarter[-1]
company_path = "/dbfs/mnt/access_work/UC25/Topic Modeling/Investor Relations_Test/" + company_acronym + "_Santhosh/"

pdf_path_p1 = company_path + company_acronym + "_" + report_quarter + "_Transcript.pdf" 
pdf_path_p4 = company_path + company_acronym + "_" + historical_quarter[-4] + "_Transcript.pdf"
pdf_path_p3 = company_path + company_acronym + "_" + historical_quarter[-3] + "_Transcript.pdf"
pdf_path_p2 = company_path + company_acronym + "_" + historical_quarter[-2] + "_Transcript.pdf"

# COMMAND ----------

pdf_path_p1

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Extract text from PDF

# COMMAND ----------

# Extract text from the PDF 
text_p4 = extract_text_from_pdf(pdf_path_p4) 
text_p3 = extract_text_from_pdf(pdf_path_p3) 
text_p2 = extract_text_from_pdf(pdf_path_p2) 
text_p1 = extract_text_from_pdf(pdf_path_p1) 


# COMMAND ----------

# Extract participants from the text 
participants_p4, execs_p4 = extract_participants(text_p4) 
participants_p3, execs_p3 = extract_participants(text_p3) 
participants_p2, execs_p2 = extract_participants(text_p2) 
participants_p1, execs_p1 = extract_participants(text_p1) 

# COMMAND ----------

participants_p1

# COMMAND ----------

execs_p1

# COMMAND ----------

# Split the Management Discussion Section by speakers 
speeches_p4 = split_management_discussion(text_p4, participants_p4, execs_p4['CEO'], execs_p4['CFO']) 
speeches_p3 = split_management_discussion(text_p3, participants_p3, execs_p3['CEO'], execs_p3['CFO']) 
speeches_p2 = split_management_discussion(text_p2, participants_p2, execs_p2['CEO'], execs_p2['CFO']) 
speeches_p1 = split_management_discussion(text_p1, participants_p1, execs_p1['CEO'], execs_p1['CFO']) 


# COMMAND ----------

questions_and_answers_p1 = parse_questions_and_answers_v1(text_p1) 
questions_and_answers_p2 = parse_questions_and_answers_v1(text_p2) 
questions_and_answers_p3 = parse_questions_and_answers_v1(text_p3) 
questions_and_answers_p4 = parse_questions_and_answers_v1(text_p4) 

# COMMAND ----------

analysts_question1

# COMMAND ----------

questions_and_answers_p1[1]

# COMMAND ----------

re.sub(pattern, '', questions_and_answers_p1[1]['question'])

# COMMAND ----------

pattern = r'\(?\d{2}:\d{2}:\d{2}\)?'
analysts_question1 = " ".join([":".join(re.sub(pattern, '', dic['question']).split(':')[1:]).strip() for dic in questions_and_answers_p1 if len(dic['question'].split()) > 20])
analysts_question2 = " ".join([":".join(re.sub(pattern, '', dic['question']).split(':')[1:]).strip() for dic in questions_and_answers_p2 if len(dic['question'].split()) > 20])
analysts_question3 = " ".join([":".join(re.sub(pattern, '', dic['question']).split(':')[1:]).strip() for dic in questions_and_answers_p3 if len(dic['question'].split()) > 20])
analysts_question4 = " ".join([":".join(re.sub(pattern, '', dic['question']).split(':')[1:]).strip() for dic in questions_and_answers_p4 if len(dic['question'].split()) > 20])


# COMMAND ----------

answer1 = " ".join([":".join(re.sub(pattern, '', dic['answer']).split(':')[1:]).strip() for dic in questions_and_answers_p1 if len(dic['question'].split()) > 20])
answer2 = " ".join([":".join(re.sub(pattern, '', dic['answer']).split(':')[1:]).strip() for dic in questions_and_answers_p2 if len(dic['question'].split()) > 20])
answer3 = " ".join([":".join(re.sub(pattern, '', dic['answer']).split(':')[1:]).strip() for dic in questions_and_answers_p3 if len(dic['question'].split()) > 20])
answer4 = " ".join([":".join(re.sub(pattern, '', dic['answer']).split(':')[1:]).strip() for dic in questions_and_answers_p4 if len(dic['question'].split()) > 20])

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. MD & QA Sentiment Plot

# COMMAND ----------

md_NLI_sent_df = pd.DataFrame({'SPEAKER':['CEO', 'CFO', 'Whole Management Discussion', 'Other Executives'], 
                               historical_quarter[0]:[NLI_sent_total(speeches_p4['CEO'],labels_sent)[0], NLI_sent_total(speeches_p4['CFO'],labels_sent)[0], NLI_sent_total(speeches_p4['CEO'] + ' ' + speeches_p4['CFO'] + ' ' + speeches_p4['other executives'],labels_sent)[0], NLI_sent_total(speeches_p4['other executives'],labels_sent)[0]], 
                               historical_quarter[1]:[NLI_sent_total(speeches_p3['CEO'],labels_sent)[0], NLI_sent_total(speeches_p3['CFO'],labels_sent)[0], NLI_sent_total(speeches_p3['CEO'] + ' ' + speeches_p3['CFO'] + ' ' + speeches_p3['other executives'],labels_sent)[0], NLI_sent_total(speeches_p3['other executives'],labels_sent)[0]], 
                               historical_quarter[2]:[NLI_sent_total(speeches_p2['CEO'],labels_sent)[0], NLI_sent_total(speeches_p2['CFO'],labels_sent)[0], NLI_sent_total(speeches_p2['CEO'] + ' ' + speeches_p2['CFO'] + ' ' + speeches_p2['other executives'],labels_sent)[0], NLI_sent_total(speeches_p2['other executives'],labels_sent)[0]], 
                               historical_quarter[3]:[NLI_sent_total(speeches_p1['CEO'],labels_sent)[0], NLI_sent_total(speeches_p1['CFO'],labels_sent)[0], NLI_sent_total(speeches_p1['CEO'] + ' ' + speeches_p1['CFO'] + ' ' + speeches_p1['other executives'],labels_sent)[0], NLI_sent_total(speeches_p1['other executives'],labels_sent)[0]]})

# COMMAND ----------

md_len_df = pd.DataFrame({'SPEAKER':['CEO', 'CFO', 'Other Executives'], 
                               historical_quarter[0]: [len(split_sentences(speeches_p4['CEO'])), len(split_sentences(speeches_p4['CFO'])), len(split_sentences(speeches_p4['other executives']))], 
                               historical_quarter[1]: [len(split_sentences(speeches_p3['CEO'])), len(split_sentences(speeches_p3['CFO'])), len(split_sentences(speeches_p3['other executives']))], 
                               historical_quarter[2]: [len(split_sentences(speeches_p2['CEO'])), len(split_sentences(speeches_p2['CFO'])), len(split_sentences(speeches_p2['other executives']))], 
                               historical_quarter[3]: [len(split_sentences(speeches_p1['CEO'])), len(split_sentences(speeches_p1['CFO'])), len(split_sentences(speeches_p1['other executives']))]})

# COMMAND ----------

qa_NLI_sent_df = pd.DataFrame({'SPEAKER':["Questions", "Answers"], 
                               historical_quarter[0]:[NLI_sent_total(analysts_question4,labels_sent)[0], NLI_sent_total(answer4,labels_sent)[0]], 
                               historical_quarter[1]:[NLI_sent_total(analysts_question3,labels_sent)[0], NLI_sent_total(answer3,labels_sent)[0]], 
                               historical_quarter[2]:[NLI_sent_total(analysts_question2,labels_sent)[0], NLI_sent_total(answer2,labels_sent)[0]], 
                               historical_quarter[3]:[NLI_sent_total(analysts_question1,labels_sent)[0], NLI_sent_total(answer1,labels_sent)[0]]})

# COMMAND ----------

qa_len_df = pd.DataFrame({'SPEAKER':["Questions", "Answers"], 
                               historical_quarter[0]: [len(split_sentences(analysts_question4)), len(split_sentences(answer4))], 
                               historical_quarter[1]: [len(split_sentences(analysts_question3)), len(split_sentences(answer3))], 
                               historical_quarter[2]: [len(split_sentences(analysts_question2)), len(split_sentences(answer2))], 
                               historical_quarter[3]: [len(split_sentences(analysts_question1)), len(split_sentences(answer1))]})

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Verify the length of the speaker-level section to ensure the PDF has correctly parsed the CEO and CFO

# COMMAND ----------

md_len_df

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Verify the length of the speaker-level section to ensure the PDF has correctly parsed the Questions and Answers

# COMMAND ----------

qa_len_df

# COMMAND ----------

md_NLI_sent_df.set_index('SPEAKER', inplace=True)
qa_NLI_sent_df.set_index('SPEAKER', inplace=True)

# COMMAND ----------

md_NLI_sent_df_red = md_NLI_sent_df.iloc[0:2]
md_NLI_sent_df_red

# COMMAND ----------

md_NLI_sent_df_whole = md_NLI_sent_df.iloc[[2]]
md_NLI_sent_df_whole

# COMMAND ----------

qa_NLI_sent_df_red = qa_NLI_sent_df
qa_NLI_sent_df_red

# COMMAND ----------

md_len_df.set_index('SPEAKER', inplace=True)

# COMMAND ----------

## Plot the sentiment score for MD section 
if (md_len_df.iloc[0:2] == 0).any().any():
  # If the CEO and CFO sections are not identified, then plot sentiment score for whole MD
  fig = sentiment_plot(md_NLI_sent_df_whole, 'MD', company_name, report_quarter)
  fig.write_html("/dbfs/mnt/access_work/UC25/Topic Modeling/Investor Relations_Test/" + company_name + '_' + report_quarter + "_MD_historical_sentiment_v1.html")
else:
  # If the CEO and CFO sections are correctly identified, then plot sentiment score for CEO and CFO
  fig = sentiment_plot(md_NLI_sent_df_red, 'MD', company_name, report_quarter)
  fig.write_html("/dbfs/mnt/access_work/UC25/Topic Modeling/Investor Relations_Test/" + company_name + '_' + report_quarter + "_CEO_CFO_historical_sentiment_v1.html")


# COMMAND ----------

# Sentiment plot for QA
fig = sentiment_plot(qa_NLI_sent_df_red, 'QA', company_name, report_quarter)

# COMMAND ----------

fig.write_html("/dbfs/mnt/access_work/UC25/Topic Modeling/Investor Relations_Test/" + company_name + '_' + report_quarter + "_QA_historical_sentiment_v1.html")

# COMMAND ----------

ceo_sentiments = md_NLI_sent_df_red.loc['CEO', :]
cfo_sentiments = md_NLI_sent_df_red.loc['CFO', :]
md_sentiments = md_NLI_sent_df_whole.loc['Whole Management Discussion', :]
q_sentiments = qa_NLI_sent_df_red.loc['Questions', :]
a_sentiments = qa_NLI_sent_df_red.loc['Answers', :]

# COMMAND ----------

print(f'Sentiment Scores Summary for {company_name} {historical_quarter[-1]}')
if (md_len_df.iloc[0:2] == 0).any().any():
  speaker_sentiments = [md_sentiments, q_sentiments, a_sentiments]
  speaker = ["whole management discussion section", "analyst's questions", "answers"]
else:
  speaker_sentiments = [ceo_sentiments, cfo_sentiments, md_sentiments, q_sentiments, a_sentiments]
  speaker = ["CEO", "CFO", "whole management discussion section", "analyst's questions", "answers"]
print('------')
for i in range(len(speaker)):
  rank, non_numeric_indices, change, change_type, current_quarter, last_quarter = analyze_current_quarter(speaker_sentiments[i], historical_quarter)
  print(f"Sentiment score for {current_quarter} {speaker[i]} is ranked top {rank} among last 4 quarters.")
  print('------')

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. QA Section Topic Plot

# COMMAND ----------

# MAGIC %md
# MAGIC #### Check whether all important questions are parsed correctly

# COMMAND ----------

questions_and_answers_p1 = parse_questions_and_answers_v1(text_p1) 

# COMMAND ----------

## Sanity check
for i, qa in enumerate(questions_and_answers_p1, start=1):
  print(f"Question {i}: {qa['question'].strip()}")
  print(f"Answer {i}: {qa['answer'].strip()}")

  print()

# COMMAND ----------

# MAGIC %md
# MAGIC **NOTE:** Since the pdf has some speaker identification error, we need to manually count the number of questions in Q&A. Many of the short questions have no helpful information, so we also want to exclude them. **We are using the num_token parameter to drop some bad quality questions.** This parameter is varied according to different earnings calls. Please manually glipse the pdf to have a basic idea of what is the suitable number for this company.

# COMMAND ----------

# MAGIC %md
# MAGIC #### num_token
# MAGIC
# MAGIC Suggested value: 15-25

# COMMAND ----------

num_token = int(dbutils.widgets.get("num_token"))
print(num_token)

# COMMAND ----------

pattern = r'\(?\d{2}:\d{2}:\d{2}\)?'
analysts_question = [[":".join(re.sub(pattern, '', dic['question']).split(':')[1:]).strip()] for dic in questions_and_answers_p1 if len(dic['question'].split()) > num_token]

# COMMAND ----------

# Compare this number with the manually counted number of questions in the QA section. If they are different, change the num_token variable above.
len(analysts_question)

# COMMAND ----------

# Print the questions to see the details
analysts_question

# COMMAND ----------

# MAGIC %md
# MAGIC #### new_IR_topic

# COMMAND ----------

# MAGIC %md
# MAGIC **Note:** Load the default company specific topic list from Blob. If IR wants to edit the topic list, you can also change the list directly. Please remember to store the new list back to Blob.

# COMMAND ----------

with open("/dbfs/mnt/access_work/UC25/Topic Modeling/Investor Relations_Test/IR_company_topics_dictionary_24Q3.json", "r") as f:
    IR_topic_dict = json.load(f)

# COMMAND ----------

new_IR_topics = IR_topic_dict[company_acronym]

# COMMAND ----------

new_IR_topics

# COMMAND ----------

IR_template = "This question is about "
new_IR_topics = [IR_template + tp for tp in new_IR_topics]
labels = new_IR_topics

# COMMAND ----------

# MAGIC %md
# MAGIC #### para_threshold  
# MAGIC This is the minimum entailment score threshold to say **one question** is related with the specific topic 
# MAGIC Suggested value: 0.3 - 0.2

# COMMAND ----------

para_threshold = float(dbutils.widgets.get("para_threshold"))
print(para_threshold)

# COMMAND ----------

# MAGIC %md
# MAGIC #### sent_threshold  
# MAGIC This is the minimum entailment score threshold to say **one sentence** is related with the specific topic   
# MAGIC Suggested value: 0.3 - 0.2

# COMMAND ----------

sent_threshold = float(dbutils.widgets.get("sent_threshold"))
print(sent_threshold)

# COMMAND ----------

# MAGIC %md
# MAGIC **Note:**   
# MAGIC Run next cell and check whether each sentence topic makes sense to you.   
# MAGIC **The matched topics should meet two criteria:** 1. the entailment score of the whole question and the topic is higher than para_threshold 2. the entailment score of 1+ sentence in the question and the topic is higher than sent_threshold.   
# MAGIC If you think something is missing, you can lower the para_threshold and sent_threshold, or add new topics in the new_IR_topics list.

# COMMAND ----------

company_dict = {}
new_q_flag = {i+1: 'Question ' + str(i+1) +': ' + q[0] for i,q in enumerate(analysts_question)}

for question, raw_text in enumerate(analysts_question):
  print('=====================================================================')
  print('Question: ', question + 1)

  text1_md, text2_md = create_text_pair(raw_text, inference_template, labels)
  inference_result1_md = pl_inference1([f"{text1_md[i]}</s></s>{text2_md[i]}" for i in range(len(text1_md)) ], padding=True, top_k=None, batch_size = 16, truncation = True, max_length = 512)
  result = inference_summary1(text1_md, text2_md, inference_result1_md)
  print(raw_text)
  for i, r in enumerate(result):
    if r['scores']['entailment'] > para_threshold:
      ## Paragraph-level:
      topic = r['labels'][23:-1]
      
      ## Sentence-level:
      selected_text = split_sentences(r['sequence'])
      text3, text4 = create_text_pair(selected_text, inference_template, [r['labels'][:-1]])
      inference_result = pl_inference1([f"{text3[i]}</s></s>{text4[i]}" for i in range(len(text3)) ], padding=True, top_k=None, batch_size = 16, truncation = True, max_length = 512)
      result_v2 = inference_summary1(text3, text4, inference_result)
      selected_sentences = []
      for j, r2 in enumerate(result_v2):
        if r2['scores']['entailment'] > sent_threshold:
          selected_sentences.append(r2['sequence'])
      if selected_sentences != []:
        print('-------------------------------------------------------------------------')
        print('Topic: ', topic)
        print("sentences result: ", "".join(selected_sentences))  
        if topic not in company_dict.keys():
          company_dict[topic] = dict([])
          company_dict[topic]['para_question'] = []
          company_dict[topic]['para_raw'] = []
          company_dict[topic]['sentences_question'] = []
          company_dict[topic]['sentences_raw'] = []        
        company_dict[topic]['para_question'].append('question '+ str(question+1) + " " + r['sequence'])
        company_dict[topic]['para_raw'].append(r['sequence'])
        company_dict[topic]['sentences_question'].append('question '+ str(question + 1) + " " + " ".join(selected_sentences))
        company_dict[topic]['sentences_raw'].append(" ".join(selected_sentences))
        new_q_flag[question + 1] = 1

        

# COMMAND ----------

# Questions without any matched topic
[q for i,q in new_q_flag.items() if q != 1]

# COMMAND ----------



# COMMAND ----------

data_lst = []
for topic, content in company_dict.items():
  row = {'topic': topic, 'para(with page)': content['para_question'], 'para (raw)': " ".join(content['para_raw']), 'sentences(with page)': content['sentences_question'], 'sentences (raw)': " ".join(content['sentences_raw'])}
  data_lst.append(row)
company_summary_df = pd.DataFrame(data_lst)
company_summary_df['para count_extractQ'] = company_summary_df['sentences(with page)'].apply(len)
company_summary_df['para count_fullQ'] = company_summary_df['para(with page)'].apply(len)
company_summary_df['para(with page) br'] = company_summary_df['para(with page)'].apply(lambda x: " ".join(w for w in x)).apply(lambda txt: wrap_text_with_page(txt, 100))
company_summary_df['sentences(with page) br'] = company_summary_df['sentences(with page)'].apply(lambda x: "".join(w for w in x)).apply(lambda txt: wrap_text_with_page(txt, 100))
company_summary_df['NLI_sentiment'] = company_summary_df['sentences (raw)'].apply(lambda x: NLI_sent_total(x, labels_sent)[0] )
company_summary_df['NLI_sentence_count'] = company_summary_df['sentences (raw)'].apply(lambda x: NLI_sent_total(x, labels_sent)[1] )
company_summary_df = company_summary_df.sort_values(by=['para count_extractQ', 'NLI_sentence_count', 'NLI_sentiment'], ascending= [True, True, True])

# COMMAND ----------

company_summary_df = company_summary_df[company_summary_df['NLI_sentence_count'] > 0]

# COMMAND ----------

company_summary_df

# COMMAND ----------

###### OPTION CODE: if you want to exclude some topics

# manual_excluded_tp = ['Fed rate cut', 'operating income', 'private and alternative asset classes']
# company_summary_df = company_summary_df[~company_summary_df['topic'].isin(manual_excluded_tp)]

# COMMAND ----------

company_summary_df1 = company_summary_df.sort_values(by=['para count_fullQ', 'NLI_sentence_count', 'NLI_sentiment'])

# COMMAND ----------

fig = go.Figure()
fig.add_trace(go.Bar(
    y=company_summary_df1['topic'],
    x=company_summary_df1['para count_fullQ'],
    name='',
    text= '',
    textposition="inside",
    textfont=dict(color="black"),
    marker=dict(color = company_summary_df1['NLI_sentiment'],
                     colorscale= px.colors.diverging.RdYlGn,
                    #  color_continuous_scale=[(1, "green"), (0, "yellow"), (-1, "red")],
                    cmax = 1, cmin = -1,
                    colorbar = dict(title = "Color Scale: Sentiment Score", lenmode = 'fraction', len = 0.3,x=1,y=0.9),
                line = dict(
                  color = "tan",
                  width = 4,
                )
),
    customdata = np.stack((company_summary_df1['NLI_sentiment'], company_summary_df1['para(with page) br'], company_summary_df1['topic'], company_summary_df1['NLI_sentence_count']), axis=-1),
    hovertemplate = "<b>%{customdata[2]}</b><br>" + " </br> Sentiment: %{customdata[0]} </br> Questions Count: %{x} </br> Matched Sentences Count:  %{customdata[3]} </br> </br> Text: </br>%{customdata[1]}", 
    orientation='h'
))

# Here we modify the tickangle of the xaxis, resulting in rotated labels.
fig.update_layout(width=1200, height = 1200)
fig.update_layout(template='plotly_white')
fig.update_layout(title_text = "Topics mentioned in " + company_name + " " + report_quarter + " Earnings Call Analyst's Questions (Full Questions)")
fig.update_xaxes(title_text = 'Questions Count')
fig.update_yaxes(title_text = "Topic")
fig.update_coloraxes(showscale=True)
fig.update_layout(showlegend=False)

fig.update_layout(
    font_color="tan",
    title_font_color="tan",
    legend_title_font_color="tan"
)
fig.update_layout(plot_bgcolor='rgb(50,50,50,50)', paper_bgcolor= 'black')
fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='darkslategray')
fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='dimgray')

fig.show()

# COMMAND ----------

fig.write_html("/dbfs/mnt/access_work/UC25/Topic Modeling/Investor Relations_Test/" + company_name + '_' + report_quarter + "_topics_bar_plot_byQ_FullQ_v1.html")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Write sentiment summary and Q&A topic summary to docx.

# COMMAND ----------

pattern = r'\(?\d{2}:\d{2}:\d{2}\)?'
qa_all = [[":".join(re.sub(pattern, '', dic['question']).split(':')[1:]).strip(), dic['answer']] for dic in questions_and_answers_p1 if len(dic['question'].split()) > num_token]

# COMMAND ----------

print(f'Sentiment Scores Summary for {company_name} {historical_quarter[-1]}')
if (md_len_df.iloc[0:2] == 0).any().any():
  speaker_sentiments = [md_sentiments, q_sentiments, a_sentiments]
  speaker = ["whole management discussion section", "analyst's questions", "answers"]
else:
  speaker_sentiments = [ceo_sentiments, cfo_sentiments, md_sentiments, q_sentiments, a_sentiments]
  speaker = ["CEO", "CFO", "whole management discussion section", "analyst's questions", "answers"]
print('------')
for i in range(len(speaker)):
  rank, non_numeric_indices, change, change_type, current_quarter, last_quarter = analyze_current_quarter(speaker_sentiments[i], historical_quarter)
  print(f"Sentiment score for {current_quarter} {speaker[i]} is ranked top {rank} among last 4 quarters.")
  print('------')


print("\n")

print(f'Topics Summary for {company_name} {historical_quarter[-1]} Q&A section')
for question, raw_text in enumerate(qa_all):
  print('=====================================================================')
  print('Question: ', question + 1)

  text1_md, text2_md = create_text_pair([raw_text[0]], inference_template, labels)
  inference_result1_md = pl_inference1([f"{text1_md[i]}</s></s>{text2_md[i]}" for i in range(len(text1_md)) ], padding=True, top_k=None, batch_size = 16, truncation = True, max_length = 512)
  result = inference_summary1(text1_md, text2_md, inference_result1_md)
  print(raw_text[0])

  print('-------------------------------------------------------------------------')
  print('Topics: ')
  for i, r in enumerate(result):
    if r['scores']['entailment'] > para_threshold:
      ## Paragraph-level:
      topic = r['labels'][23:-1]

      ## Sentence-level:
      selected_text = split_sentences(r['sequence'])
      text3, text4 = create_text_pair(selected_text, inference_template, [r['labels'][:-1]])
      inference_result = pl_inference1([f"{text3[i]}</s></s>{text4[i]}" for i in range(len(text3)) ], padding=True, top_k=None, batch_size = 16, truncation = True, max_length = 512)
      result_v2 = inference_summary1(text3, text4, inference_result)
      selected_sentences = []
      for j, r2 in enumerate(result_v2):
        if r2['scores']['entailment'] > sent_threshold:
          selected_sentences.append(r2['sequence'])
      if selected_sentences != []:
        
        print(topic)

  print('-------------------------------------------------------------------------')

  print('Sentiment (1 is positive, 0 is neutral, and -1 is negative): ')
  print('- full question: ', NLI_sent_total(raw_text[0],labels_sent)[0])
  print('- full answer: ', NLI_sent_total(raw_text[1],labels_sent)[0])
  print('-------------------------------------------------------------------------')  
  print('Answer: ', question + 1)
  print(raw_text[1])

# COMMAND ----------



# COMMAND ----------

output_buffer = io.StringIO()

# COMMAND ----------

with contextlib.redirect_stdout(output_buffer):
  print(f'Sentiment Scores Summary for {company_name} {historical_quarter[-1]}')
  if (md_len_df.iloc[0:2] == 0).any().any():
    speaker_sentiments = [md_sentiments, q_sentiments, a_sentiments]
    speaker = ["whole management discussion section", "analyst's questions", "answers"]
  else:
    speaker_sentiments = [ceo_sentiments, cfo_sentiments, md_sentiments, q_sentiments, a_sentiments]
    speaker = ["CEO", "CFO", "whole management discussion section", "analyst's questions", "answers"]
  print('------')
  for i in range(len(speaker)):
    rank, non_numeric_indices, change, change_type, current_quarter, last_quarter = analyze_current_quarter(speaker_sentiments[i], historical_quarter)
    print(f"Sentiment score for {current_quarter} {speaker[i]} is ranked top {rank} among last 4 quarters.")
    print('------')
  
  print("\n")

  print(f'Topics Summary for {company_name} {historical_quarter[-1]} Q&A section')
  for question, raw_text in enumerate(qa_all):
    print('=====================================================================')
    print('Question: ', question + 1)

    text1_md, text2_md = create_text_pair([raw_text[0]], inference_template, labels)
    inference_result1_md = pl_inference1([f"{text1_md[i]}</s></s>{text2_md[i]}" for i in range(len(text1_md)) ], padding=True, top_k=None, batch_size = 16, truncation = True, max_length = 512)
    result = inference_summary1(text1_md, text2_md, inference_result1_md)
    print(raw_text[0])

    print('-------------------------------------------------------------------------')
    print('Topics: ')
    for i, r in enumerate(result):
      if r['scores']['entailment'] > para_threshold:
        ## Paragraph-level:
        topic = r['labels'][23:-1]

        ## Sentence-level:
        selected_text = split_sentences(r['sequence'])
        text3, text4 = create_text_pair(selected_text, inference_template, [r['labels'][:-1]])
        inference_result = pl_inference1([f"{text3[i]}</s></s>{text4[i]}" for i in range(len(text3)) ], padding=True, top_k=None, batch_size = 16, truncation = True, max_length = 512)
        result_v2 = inference_summary1(text3, text4, inference_result)
        selected_sentences = []
        for j, r2 in enumerate(result_v2):
          if r2['scores']['entailment'] > sent_threshold:
            selected_sentences.append(r2['sequence'])
        if selected_sentences != []:
          
          print(topic)

    print('-------------------------------------------------------------------------')

    print('Sentiment (1 is positive, 0 is neutral, and -1 is negative): ')
    print('- full question: ', NLI_sent_total(raw_text[0],labels_sent)[0])
    print('- full answer: ', NLI_sent_total(raw_text[1],labels_sent)[0])
    print('-------------------------------------------------------------------------')  
    print('Answer: ', question + 1)
    print(raw_text[1])

# COMMAND ----------

sys.stdout = sys.__stdout__
output_buffer.getvalue()

# COMMAND ----------

doc = Document()
doc.add_paragraph(output_buffer.getvalue())

# COMMAND ----------

destination_path = '/dbfs/mnt/access_work/UC25/Topic Modeling/Investor Relations_Test/' + company_name + '_' + report_quarter + '_sentiment&topics_summary_v1.docx'


# COMMAND ----------

destination_path

# COMMAND ----------

temp_file = '/tmp/test1_'+ company_name +'.docx'

doc.save(temp_file)

copyfile(temp_file, destination_path)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Download Word.docx, Write One Sentence Summary about Quarterly Sentiment Score Change and Add it in Word Document
# MAGIC **1. Select and copy the highlighted quarterly sentiment score change from Word document.**
# MAGIC ![](/files/Yujing/resources/IR_sentiment_text_screenshot.PNG)
# MAGIC **2. Write one sentence summary based on the quarterly sentiment score change. Or you can use Copilot to generate the one sentence summary**. You can write the prompt as: "Please use one sentence to summarize the sentiment ranks for [Company Name]'s earnings call transcript: xxxx"
# MAGIC ![](/files/Yujing/resources/IR_sentiment_copilot_result.PNG)
# MAGIC **3. Paste the answer in the Word document.**
# MAGIC ![](/files/Yujing/resources/IR_sentiment_text_withsummary_screenshot.PNG)

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ## Optional code: IR company-level topic list

# COMMAND ----------

# IR_topic_dict = {"ALIT": ["flows", "earnings", "EPS", "retirement", "benefit ratio", "voluntary", "utilization", "repricing", "organic growth", "surrender", "fee rate", "fee revenue", "group life", "Life business", "group disability", "spread", "revenue yield", "stop loss"],
#                  "AMP": ["net flows (net inflows)", "outflows", "fee rate", "client engagement", "revenue yield", "operating margin", "sidecar", "partnership", "private assets", "alternative asset", "international distribution", "operating environment", "interest rate", "alternative income", "commercial real estate", "cash flow", "acquisition", "organic investment"],
#                  "BEN": ["flows", "fee rate", "client engagement", "revenue yield", "operating margin", "partnership", "VII", "private assets", "alternative asset", "international distribution", "operating environment", "interest rate", "alternative income", "commercial real estate", "cash flow", "acquisition", "organic investment"],
#                  "BLK": ["net flows (net inflows)", "outflows", "fee rate", "client engagement", "revenue yield", "operating margin", "sidecar", "partnership", "private assets", "alternative asset", "international distribution", "operating environment", "interest rate", "alternative income", "commercial real estate", "cash flow", "acquisition", "organic investment"],
#                  "CRBG": ["flows", "EPS", "Group Retirement", "Protection Solutions", "mortality", "organic growth", "surrender", "fee rate", "fee revenue", "group life", "Life Insurance", "spread"],
#                  "EQH": ["flows", "EPS", "Group Retirement", "Protection Solutions", "mortality", "organic growth", "surrender", "fee rate", "fee revenue", "group life", "Life Insurance", "spread"],
#                  "IVZ": ["net flows (net inflows)", "outflows", "fee rate", "client engagement", "revenue yield", "operating margin", "sidecar", "partnership", "private assets", "alternative asset", "international distribution", "operating environment", "interest rate", "alternative income", "commercial real estate", "cash flow", "acquisition", "organic investment"],
#                  "LNC": ["flows", "earnings", "EPS", "retirement", "benefit ratio", "voluntary", "utilization", "repricing", "organic growth", "surrender", "fee rate", "fee revenue", "group life", "Life business", "group disability", "spread", "revenue yield"],
#                  "MET": ["flows", "earnings", "EPS", "retirement", "benefit ratio", "voluntary", "utilization", "repricing", "organic growth", "surrender", "fee rate", "fee revenue", "group life", "Life business","Group Benefits", "group disability", "spread", "revenue yield"],
#                  "PFG":["flows", "earnings", "EPS", "retirement", "specialty benefits", "PGI", "organic growth", "surrenders", "fee rate", "fee revenue", "group life","life insurance", "group disability", "spread", "revenue yield"],
#                  "PRU": ["flows", "earnings", "EPS", "retirement", "benefit ratio", "voluntary", "utilization", "repricing", "organic growth", "surrender", "fee rate", "fee revenue", "group life", "Life business", "group disability", "spread", "revenue yield", "stop loss"],
#                  "SLF": ["flows", "EPS", "Group Retirement", "Protection Solutions", "mortality", "organic growth", "surrender", "fee rate", "fee revenue", "group life", "Life Insurance", "spread", "stop loss", "dental", "repricing"],
#                  "TROW": ["flows", "fee rate", "client engagement", "revenue yield", "operating margin", "partnership", "VII", "private assets", "alternative asset", "international distribution", "operating environment", "interest rate", "alternative income", "commercial real estate", "cash flow", "acquisition", "organic investment"]}

# COMMAND ----------

# import json

# with open("/dbfs/mnt/access_work/UC25/Topic Modeling/Investor Relations/IR_company_topics_dictionary_24Q3.json", "w") as f:
#     json.dump(IR_topic_dict, f)