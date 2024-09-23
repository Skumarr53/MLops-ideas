# Databricks notebook source
# MAGIC %run ./../preprocessor/text_preprocessor

# COMMAND ----------

# MAGIC %run ../dataclasses/dataclasses

# COMMAND ----------

import pytz

# COMMAND ----------

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple, Union
import pandas as pd
import pytz
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

'''
specify run_mode: 'daily' | 'historical' 
'''

class XMLParser:
    def __init__(self, min_date: datetime, max_date: datetime, new_data_dims: Tuple[int, int], curr_df: pd.DataFrame, hist_df: pd.DataFrame, config, run_mode: str = 'daily'):
        """
        Initializes the XMLParser with the given parameters.

        Args:
            min_date (datetime): Minimum date for filtering data.
            max_date (datetime): Maximum date for filtering data.
            new_data_dims (Tuple[int, int]): Dimensions of the new data.
            curr_df (pd.DataFrame): Current dataframe containing the latest data.
            hist_df (pd.DataFrame): Historical dataframe containing past data.
            config (Config): Configuration object containing various settings.
            run_mode (str): Mode of operation, either 'daily' or 'historical'.
        """
        self.transcript_vars = TranscriptVarsData()
        self.config = config
        self.curr_df = curr_df
        self.txt_preprocessor = TextPreprocessor()
        self.run_mode = run_mode
        self.last_parsed_datetime = self.get_last_pasred_time()
        self.min_last_ts_date = min_date
        self.max_last_ts_date = max_date
        self.last_upload_dt = self.curr_df.UPLOAD_DATETIME.max() if not self.curr_df.empty else hist_df.UPLOAD_DATETIME.max()
        self.last_data_dims = new_data_dims
        self.hist_df = hist_df
        self.parsed_df = self.get_parsed_df()

    def get_last_upload_dt(self):
        """
        Retrieves the last upload datetime from the current or historical dataframe.

        Returns:
            datetime: The last upload datetime.
        """
        if self.curr_df.empty:
            last_upload_dt = self.hist_df.UPLOAD_DATETIME.max() if self.run_mode == 'daily' else pd.to_datetime('2024-01-01 00:00:00.000')
        else:
            last_upload_dt = self.curr_df.UPLOAD_DATETIME.max()
        return last_upload_dt
    
    def get_parsed_df(self):
        """
        Parses the current dataframe into a new dataframe with extracted XML data.

        Returns:
            pd.DataFrame: DataFrame containing parsed XML data.
        """
        if not self.curr_df.empty:
            curr_list = self.curr_df.to_numpy().tolist()
            parsed_ct_list = [self.parse_ct_xml(row) for row in curr_list]
        else:
            parsed_ct_list = []
        return pd.DataFrame.from_records(parsed_ct_list)

    def get_last_pasred_time(self):
        """
        Gets the last parsed time in the local timezone.

        Returns:
            datetime: The last parsed time.
        """
        eastern_tzinfo = pytz.timezone(self.config.timezone)
        load_date_time = self.utc_to_local(datetime.now(), eastern_tzinfo)
        load_date_str = datetime.strptime(load_date_time.strftime("%Y-%m-%d %H:%M:%S"), "%Y-%m-%d %H:%M:%S") 
        return load_date_str

    def utc_to_local(self, utc_dt: datetime, tz_var: pytz.timezone) -> datetime:
        """
        Converts UTC datetime to local timezone

        Args:
            utc_dt (datetime): UTC datetime
            tz_var (pytz.timezone): Timezone variable

        Returns:
            datetime: Local datetime
        """
        return utc_dt.replace(tzinfo=timezone.utc).astimezone(tz=tz_var)
    
    def extract_metadata(self, root):
        """
        Extracts metadata from the XML root element.

        Args:
            root (ET.Element): The root element of the XML tree.

        Returns:
            Tuple[str, str, str, str]: A tuple containing status, title, ID, and date.
        """
        status = root.get('product')
        title_xml_path = f'./{self.config.xlmns}meta/{self.config.xlmns}title'
        call_transcript_title = root.find(title_xml_path).text
        call_transcript_id = root.get('id')
        date_xml_path = f'./{self.config.xlmns}meta/{self.config.xlmns}date'
        date = root.find(date_xml_path).text
        return status, call_transcript_title, call_transcript_id, date
    
    def parse_participants(self, root):
        """
        Parses participant information from the XML root element.

        Args:
            root (ET.Element): The root element of the XML tree.
        """
        participants_xml_path = f'./{self.config.xlmns}meta/{self.config.xlmns}participants/'
        for child in root.findall(participants_xml_path):
            if child.get('type') == 'corprep':
                self.transcript_vars.corp_representative_ids.append(child.get('id'))
                if self.txt_preprocessor.check_ceo(child.get('title')):
                    self.transcript_vars.ceo_matches += 1
                self.transcript_vars.company_name.append(child.get('affiliation'))
                self.transcript_vars.company_id.append(child.get('affiliation_entity'))
                self.transcript_vars.ceo_id.append(child.get('id'))
                self.transcript_vars.ceo_unique_id.append(child.get('entity'))
            elif child.get('type') == 'analyst':
                self.transcript_vars.analyst_ids.append(child.get('id'))
    
    def parse_sections(self, root):
        """
        Parses sections from the XML root element.
        Args:
            root (ET.Element): The root element of the XML tree.
        """
        section_xml_path = f'./{self.config.xlmns}body/{self.config.xlmns}section'
        for child in root.findall(section_xml_path):
            if child.get('name') == 'MANAGEMENT DISCUSSION SECTION':
                self._extract_speech(child)
            elif child.get('name') == 'Q&A':
                self._extract_qa(child)

    def parse_company_id(self, root):
        """
        Parses company ID from the XML root element.
        Args:
            root (ET.Element): The root element of the XML tree.
        """
        company_xml_path = f'./{self.config.xlmns}meta/{self.config.xlmns}companies/{self.config.xlmns}company'
        self.transcript_vars.company_id = self.transcript_vars.company_id or [root.find(company_xml_path).text]

    def _create_empty_transcript_dict(self, entity_id: str, sf_date: str, sf_id: str, error: str) -> Dict[str, Any]:
        """
        Creates an empty transcript dictionary
        Args:
            entity_id (str): Entity ID
            sf_date (str): SF Date
            sf_id (str): SF ID
            error (str): Error status
        Returns:
            Dict[str, Any]: Empty transcript dictionary
        """
        return {
            "ENTITY_ID": entity_id, "SF_DATE": sf_date, "SF_ID": sf_id, "ERROR": error,
            "TRANSCRIPT_STATUS": '', "COMPANY_NAME": '', "VERSION_ID": '', "UPLOAD_DT_UTC": '',
            "COMPANY_ID": '', "CALL_NAME": '', "CALL_ID": '', "DATE": '', "CEO_DISCUSSION": '',
            "EXEC_DISCUSSION": '', "ANALYST_QS": '', "CEO_ANS": '', "EXEC_ANS": '',
            "MGNT_DISCUSSION": '', "QA_SECTION": '', "EARNINGS_CALL": '', 'EVENT_DATETIME_UTC': ''
        }

    def _extract_speech(self, section: ET.Element) -> None:
        """
        Extracts speech from the XML section

        Args:
            section (ET.Element): XML section element
            corp_representative_ids (List[str]): Corporate representative IDs
            ceo_id (List[str]): CEO IDs
            ceo_speech (List[str]): CEO speech list
            executive_speech (List[str]): Executive speech list
        """
        for participant in section.findall('./'):
            p_id = participant.get('id')
            if p_id in self.transcript_vars.corp_representative_ids:
                if p_id in self.transcript_vars.ceo_id:
                    for para in participant.findall(f'./{self.config.xlmns}plist/'):
                        if para.text:
                           self.transcript_vars.ceo_speech.append(" ".join(para.text.split()))
                else:
                    for para in participant.findall(f'./{self.config.xlmns}plist/'):
                        if para.text:
                            self.transcript_vars.executive_speech.append(" ".join(para.text.split()))                    
                    

    def _extract_qa(self, section: ET.Element) -> None:
        """
        Extracts Q&A from the XML section

        Args:
            section (ET.Element): XML section element
            analyst_ids (List[str]): Analyst IDs
            corp_representative_ids (List[str]): Corporate representative IDs
            ceo_id (List[str]): CEO IDs
            analysts_questions (List[str]): Analysts' questions list
            ceo_answers (List[str]): CEO answers list
            executive_answers (List[str]): Executive answers list
        """
        for participant in section.findall('./'):
            p_id = participant.get('id')
            if p_id in self.transcript_vars.analyst_ids:
                for para in participant.findall(f'./{self.config.xlmns}plist/'):
                    if para.text:
                        self.transcript_vars.analysts_questions.append(" ".join(para.text.split()))

            if p_id in self.transcript_vars.corp_representative_ids:
                if p_id in self.transcript_vars.ceo_id:
                    for para in participant.findall(f'./{self.config.xlmns}plist/'):
                        if para.text:
                            self.transcript_vars.ceo_answers.append(" ".join(para.text.split()))
                else:
                    for para in participant.findall(f'./{self.config.xlmns}plist/'):
                        if para.text:
                            self.transcript_vars.executive_answers.append(" ".join(para.text.split()))

    def parse_ct_xml(self, xml_list_row: List[Any]) -> Dict[str, Any]:
        """
        Parse XML strings stored in Snowflake
        Args:
            xml_list_row (List[Any]): Call transcript XML row data
        Returns:
            Dict[str, Any]: Dictionary object with call transcript discussion details
        """
        sf_date, sf_id, entity_id = xml_list_row[0], xml_list_row[1], xml_list_row[2]
        xml_str, version_id, upload_dt, event_datetime = xml_list_row[3], xml_list_row[4], xml_list_row[5], xml_list_row[8]

        error = 'no'
        try:
            root = ET.fromstring(xml_str)
        except ET.ParseError as e:
            logging.error(f"ParseError occurred for transcript with ID = {xml_list_row[1]}. Error: {e}")
            error = 'yes'
            return self._create_empty_transcript_dict(entity_id, sf_date, sf_id, error)
        
        status, call_transcript_title, call_transcript_id, date = self.extract_metadata(root )

        if 'correct' not in status.lower():
            return self.parsed_df
        
        if not call_transcript_id: call_transcript_id = xml_list_row[1]

        self.parse_participants(root)
        self.parse_company_id(root)

        try:
            self.parse_sections(root)
        except Exception as ex:
            logging.error(f"Error while parsing XML: {ex}")
            raise

        # merger = 1 if len(set(self.transcript_vars.company_name)) > 1 else 0
        earnings_call_title = self.txt_preprocessor.check_earnings(call_transcript_title)
        management_discussion = ' '.join(self.transcript_vars.ceo_speech) + ' '.join(self.transcript_vars.executive_speech)
        management_answers = ' '.join(self.transcript_vars.ceo_answers) + ' '.join(self.transcript_vars.executive_answers)
        q_and_a_total = [' '.join(self.transcript_vars.analysts_questions) + management_answers]
        self.transcript_vars.company_name = ' '.join(self.transcript_vars.company_name)

        return {
            "ENTITY_ID": entity_id, "SF_DATE": sf_date, "SF_ID": sf_id, "ERROR": error,
            "TRANSCRIPT_STATUS": status, "VERSION_ID": version_id, "UPLOAD_DT_UTC": upload_dt,
            "COMPANY_NAME": self.transcript_vars.company_name, "COMPANY_ID": self.transcript_vars.company_id[0], "CALL_NAME": call_transcript_title,
            "CALL_ID": call_transcript_id, "DATE": date, "CEO_DISCUSSION": self.transcript_vars.ceo_speech, "EXEC_DISCUSSION": self.transcript_vars.executive_speech,
            "ANALYST_QS": self.transcript_vars.analysts_questions, "CEO_ANS": self.transcript_vars.ceo_answers, "EXEC_ANS": self.transcript_vars.executive_answers,
            "MGNT_DISCUSSION": [management_discussion], "QA_SECTION": q_and_a_total, "EARNINGS_CALL": self.transcript_vars.earnings_call,
            'EVENT_DATETIME_UTC': event_datetime
        }
