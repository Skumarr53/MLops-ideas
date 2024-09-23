# Databricks notebook source
# MAGIC %run ./../preprocessor/text_preprocessor

# COMMAND ----------

import pytz

# COMMAND ----------

class XMLParser:
  def __init__(self, min_date, max_date, new_data_dims, curr_df, hist_df):
      """
      Initializes attributes for parsing 

      Parameters:
      argument1 (date): min_date
      argument2 (date): max_date
      argument3 (int):shape of the dataframe
      argument4 (DataFrame):current dataframe
      argument5 (DataFrame):history dataframe
      """
      self.xlmns = config.xlmns
      eastern_tzinfo = pytz.timezone(config.timezone)
      self.txt_preprocessor=TextPreprocessor()
      load_date_time = self.utc_to_local(datetime.now(),eastern_tzinfo)
      load_date_str = datetime.strptime(load_date_time.strftime("%Y-%m-%d %H:%M:%S"), "%Y-%m-%d %H:%M:%S") 
      self.last_parsed_datetime = load_date_str
      self.parsed_df = pd.DataFrame()
      self.min_last_ts_date = min_date
      self.max_last_ts_date = max_date
      if curr_df.empty:
        #last_upload_dt should be intialized with the date from which we need to parse the historical data.
        self.last_upload_dt = pd.to_datetime('2024-01-01 00:00:00.000')
      else:
        self.last_upload_dt = curr_df.UPLOAD_DATETIME.max()
        
      self.last_data_dims = new_data_dims
      self.discussion_labels=config.discussion_labels
      if not curr_df.empty:
        self.hist_df = curr_df
        curr_list = curr_df.to_numpy().tolist()
        parsed_CT_list = []
        for row in curr_list:
          parsed_CT_list.append(self.parse_CT_xml(row))
      else:
        parsed_CT_list = {}
        self.hist_df = hist_df

      self.parsed_df = pd.DataFrame.from_records(parsed_CT_list)

      
  def utc_to_local(self, utc_dt, tz_var):
      return utc_dt.replace(tzinfo=timezone.utc).astimezone(tz=tz_var)
  
  def parse_CT_xml(self, xml_list_row):
      """Parse XML strings stored in Snowflake
      Parameters:
      argument1 : call transcript xml row data 
       
      Returns:
      dictionary object with call transcipt discussion detials.

      """ 
      SF_date = []
      SF_ID = []
      entity_ID = []
      error = []
      SF_date.append(xml_list_row[0])
      SF_ID.append(xml_list_row[1])
      entity_ID.append(xml_list_row[2])
      xml_str = xml_list_row[3]
      version_ID = xml_list_row[4]
      upload_dt = xml_list_row[5]
      event_datetime = xml_list_row[8]

      # get root element
      try:
        root = ET.fromstring(xml_str)
        error.append('no')
      except Exception as e:
        print("Oops!", e.__class__, "occurred for transcript with ID = " + str(xml_list_row[1]) + ".")
        print("Next entry.")
        print
        error.append('yes')
        call_transcript_dict = {"ENTITY_ID": entity_ID[0], "SF_DATE": SF_date[0], "SF_ID": SF_ID[0], "ERROR" : error[0], 
                     "TRANSCRIPT_STATUS": '', "COMPANY_NAME": '', "VERSION_ID" : '', "UPLOAD_DT_UTC" : '',
                     "COMPANY_ID": '', "CALL_NAME": '', "CALL_ID": '', "DATE": '', 
                     "CEO_ID": '', "CEO_DISCUSSION": '', "EXEC_DISCUSSION": '', "ANALYST_QS": '', "CEO_ANS": '', 
                     "EXEC_ANS": '', "MGNT_DISCUSSION": '', "QA_SECTION": '', "EARNINGS_CALL": '', 
                     'EVENT_DATETIME_UTC': ''}
        return call_transcript_dict

      # We maintain lists of participant IDs to track whether someone speaking is an analyst or corporate representative
      corp_representative_ids = []
      analyst_ids = []

      # Basic metadata 
      company_name = []
      company_id = []
      transcript_id = root.get('id')
      status = root.get('product')

      call_transcript_title = root.find('./' + self.xlmns + 'meta/' + self.xlmns + 'title').text

      #CHECK TO SEE IF CALL ID IN XML IS MISSING AND, IF IT IS, REPLACE WITH SNOWFLAKE ID IF AVAILABLE OR EMPTY STRING IF NOT
      if root.get('id'):
        call_transcript_id = ''.join(root.get('id'))
      else:
        call_transcript_id = ''.join(xml_list_row[1])

      # Only parse corrected earnings calls for current pipeline
      if 'correct' not in status.lower():
        return self.parsed_df

      # Call date
      date = root.find('./' + self.xlmns + 'meta/' + self.xlmns + 'date').text
      year = int(date[:4])
      month = int(date[5:7])
      day = int(date[8:10])

      # These lists will be populated with paragraphs of speech corresponding to our sections
      CEO_speech = []
      executive_speech = []
      CEO_answers = []
      executive_answers = []
      analysts_questions = []

      # To track CEO speech
      CEO_ID = []
      CEO_unique_ID = []
      CEO_matches = 0

      # Additional metadata
      number_of_analysts = 0
      merger = 0
      earnings_call = 1

      # This loop collects participant IDs and other metadata
      for child in root.findall('./' + self.xlmns + 'meta/' + self.xlmns + 'participants/'):
        if child.get('type') == 'corprep':
          corp_representative_ids.append(child.get('id'))

          if self.txt_preprocessor.checkCEO(child.get('title'))==1:
            CEO_matches += 1
            company_name.append(child.get('affiliation'))
            company_id.append(child.get('affiliation_entity'))
            CEO_ID.append(child.get('id'))
            CEO_unique_ID.append(child.get('entity'))

        if child.get('type') == 'analyst':
          analyst_ids.append(child.get('id'))

      number_of_analysts = len(analyst_ids)

      if company_id==[]:
        company_id = [root.find('./' + self.xlmns + 'meta/' + self.xlmns + 'companies/' + self.xlmns + 'company').text]
      try:
        # This loop goes through two bodies of text - Management discussion and Q&A. It collects text paragraph-wise 
        # and stores them in the relevant variable. 
        for child in root.findall('./' + self.xlmns + 'body/' + self.xlmns + 'section'):
          if child.get('name') == 'MANAGEMENT DISCUSSION SECTION':
            for participant in child.findall('./'):
              p_id = participant.get('id')
              if p_id in corp_representative_ids:
                if p_id in CEO_ID:
                  for para in participant.findall('./' + self.xlmns + 'plist/'):
                    if(para.text!= None):
                      para.text = " ".join(para.text.split())
                      CEO_speech.append(para.text)
                else:
                  for para in participant.findall('./' + self.xlmns + 'plist/'):
                    if(para.text!= None):
                      para.text = " ".join(para.text.split())
                      executive_speech.append(para.text)

          if child.get('name') == 'Q&A':
            for participant in child.findall('./'):
              p_id = participant.get('id')
              if p_id in analyst_ids:
                for para in participant.findall('./' + self.xlmns + 'plist/'):
                  if(para.text!= None):
                    para.text = " ".join(para.text.split())
                    analysts_questions.append(para.text)

              if p_id in corp_representative_ids:
                if p_id in CEO_ID: 
                  for para in participant.findall('./' + self.xlmns + 'plist/'):
                    if(para.text!= None):
                      para.text = " ".join(para.text.split())
                      CEO_answers.append(para.text)   
                else:
                  for para in participant.findall('./' + self.xlmns + 'plist/'):
                    if(para.text!= None):
                      para.text = " ".join(para.text.split())
                      executive_answers.append(para.text)
      except Exception as ex:
        raise ex
      merger = 1 if len(set(company_name))>1 else 0 
      earnings_call_title = self.txt_preprocessor.check_earnings(call_transcript_title)
      earnings_call = 1
      management_discusion = ' '.join(CEO_speech) + ' '.join(executive_speech)
      management_answers = ' '.join(CEO_answers) + ' '.join(executive_answers)
      QandA_total = [' '.join(analysts_questions) + management_answers]
      management_answers = [management_answers]
      management_discusion = [management_discusion]
      company_name = ' '.join(company_name)

      call_transcript_dict = {"ENTITY_ID" : entity_ID[0], "SF_DATE": SF_date[0], "SF_ID": SF_ID[0], "ERROR" : error[0],
                   "TRANSCRIPT_STATUS" : status, "VERSION_ID" : version_ID, "UPLOAD_DT_UTC" : upload_dt,
                   "COMPANY_NAME": company_name, "COMPANY_ID": company_id[0], "CALL_NAME": call_transcript_title, 
                   "CALL_ID": call_transcript_id, "DATE": date, "CEO_DISCUSSION": CEO_speech, "EXEC_DISCUSSION": executive_speech, 
                   "ANALYST_QS": analysts_questions, "CEO_ANS": CEO_answers, "EXEC_ANS": executive_answers, "MGNT_DISCUSSION" : management_discusion,
                   "QA_SECTION": QandA_total, "EARNINGS_CALL": earnings_call, 'EVENT_DATETIME_UTC': event_datetime}


      #call_df = pd.DataFrame.from_dict(call_dict)
      #self.parsed_df = pd.concat([self.parsed_df, call_df], ignore_index = True)
      #self.parsed_df = self.parsed_df.append(call_dict, ignore_index = True)

      #return self.parsed_df.copy()
      return call_transcript_dict
