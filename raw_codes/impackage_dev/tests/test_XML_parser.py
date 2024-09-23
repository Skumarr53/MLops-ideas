# Databricks notebook source
# MAGIC %run ../ETL/XML_parser

# COMMAND ----------

# MAGIC %pip install nutter

# COMMAND ----------

import unittest
from datetime import datetime
import pandas as pd
import pytz
from runtime.nutterfixture import NutterFixture, tag
from unittest.mock import Mock
from xml.etree.ElementTree import Element, SubElement, tostring
# from your_module import XMLParser  # Replace 'your_module' with the actual module name

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

class TestXMLParser(unittest.TestCase):
    def setUp(self):
        # Mock configuration
        self.config = Mock()
        self.config.xlmns = "{http://www.factset.com/callstreet/xmllayout/v0.1}"
        self.config.timezone = 'America/New_York'  # Updated timezone
        
        # Mock dataframes with specified columns and different data
        curr_data = {
            'ENTITY_ID': ['E1', 'E2'],
            'DATE': [datetime(2023, 1, 1), datetime(2023, 1, 2)],
            'ANALYST_QS': ['Q1', 'Q2'],
            'CEO_ANS': ['A1', 'A2'],
            'CEO_DISCUSSION': ['D1', 'D2'],
            'EXEC_ANS': ['EA1', 'EA2'],
            'EXEC_DISCUSSION': ['ED1', 'ED2'],
            'MGNT_DISCUSSION': ['MD1', 'MD2'],
            'QA_SECTION': ['QA1', 'QA2'],
            'CALL_NAME': ['CN1', 'CN2'],
            'COMPANY_NAME': ['Company1', 'Company2'],
            'EARNINGS_CALL': ['EC1', 'EC2'],
            'ERROR': ['no', 'no'],
            'TRANSCRIPT_STATUS': ['correct', 'correct'],
            'SF_DATE': ['2023-01-01', '2023-01-02'],
            'SF_ID': ['SF1', 'SF2'],
            'CALL_ID': ['C1', 'C2'],
            'UPLOAD_DT_UTC': [datetime(2023, 1, 1, 12, 0, 0), datetime(2023, 1, 2, 12, 0, 0)],
            'VERSION_ID': ['V1', 'V2'],
            'EVENT_DATETIME_UTC': [datetime(2023, 1, 1, 15, 0, 0), datetime(2023, 1, 2, 15, 0, 0)],
            'PARSED_DATETIME_EASTERN_TZ': [datetime(2023, 1, 1, 10, 0, 0), datetime(2023, 1, 2, 10, 0, 0)],
            'BACKFILL': [False, False],
            'UPLOAD_DATETIME': [datetime(2023, 1, 1, 12, 0, 0), datetime(2023, 1, 2, 12, 0, 0)]  # Added column
        }
        hist_data = {
            'ENTITY_ID': ['E3', 'E4'],
            'DATE': [datetime(2022, 12, 1), datetime(2022, 12, 2)],
            'ANALYST_QS': ['Q3', 'Q4'],
            'CEO_ANS': ['A3', 'A4'],
            'CEO_DISCUSSION': ['D3', 'D4'],
            'EXEC_ANS': ['EA3', 'EA4'],
            'EXEC_DISCUSSION': ['ED3', 'ED4'],
            'MGNT_DISCUSSION': ['MD3', 'MD4'],
            'QA_SECTION': ['QA3', 'QA4'],
            'CALL_NAME': ['CN3', 'CN4'],
            'COMPANY_NAME': ['Company3', 'Company4'],
            'EARNINGS_CALL': ['EC3', 'EC4'],
            'ERROR': ['no', 'no'],
            'TRANSCRIPT_STATUS': ['correct', 'correct'],
            'SF_DATE': ['2022-12-01', '2022-12-02'],
            'SF_ID': ['SF3', 'SF4'],
            'CALL_ID': ['C3', 'C4'],
            'UPLOAD_DT_UTC': [datetime(2022, 12, 1, 12, 0, 0), datetime(2022, 12, 2, 12, 0, 0)],
            'VERSION_ID': ['V3', 'V4'],
            'EVENT_DATETIME_UTC': [datetime(2022, 12, 1, 15, 0, 0), datetime(2022, 12, 2, 15, 0, 0)],
            'PARSED_DATETIME_EASTERN_TZ': [datetime(2022, 12, 1, 10, 0, 0), datetime(2022, 12, 2, 10, 0, 0)],
            'BACKFILL': [False, False],
            'UPLOAD_DATETIME': [datetime(2022, 12, 1, 12, 0, 0), datetime(2022, 12, 2, 12, 0, 0)]  # Added column
        }
        self.curr_df = pd.DataFrame(curr_data)
        self.hist_df = pd.DataFrame(hist_data)
        
        # Initialize XMLParser
        self.parser = XMLParser(
            min_date=datetime(2022, 1, 1),
            max_date=datetime(2023, 12, 31),
            new_data_dims=(2, 2),
            curr_df=self.curr_df,
            hist_df=self.hist_df,
            config=self.config,
            run_mode='daily'
        )

    def test_get_last_upload_dt(self):
        last_upload_dt = self.parser.get_last_upload_dt()
        self.assertEqual(last_upload_dt, datetime(2023, 1, 2, 12, 0, 0))

    def test_get_parsed_df(self):
        parsed_df = self.parser.get_parsed_df()
        self.assertIsInstance(parsed_df, pd.DataFrame)

    def test_get_last_parsed_time(self):
        last_parsed_time = self.parser.get_last_pasred_time()
        self.assertIsInstance(last_parsed_time, datetime)

    def test_utc_to_local(self):
        utc_dt = datetime(2023, 1, 1, 17, 0, 0, tzinfo=pytz.UTC)
        local_dt = self.parser.utc_to_local(utc_dt, pytz.timezone('America/New_York'))
        self.assertEqual(local_dt, datetime(2023, 1, 1, 12, 0, 0, tzinfo=pytz.timezone('America/New_York')))

    def test_extract_metadata(self):
        root = Element('root', attrib={'product': 'correct', 'id': '123'})
        title = Element('title')
        title.text = 'Test Title'
        date = Element('date')
        date.text = '2023-01-01'
        meta = Element('meta')
        meta.append(title)
        meta.append(date)
        root.append(meta)
        status, call_transcript_title, call_transcript_id, date = self.parser.extract_metadata(root)
        self.assertEqual(status, 'correct')
        self.assertEqual(call_transcript_title, 'Test Title')
        self.assertEqual(call_transcript_id, '123')
        self.assertEqual(date, '2023-01-01')

    def test_parse_participants(self):
        root = Element('root')
        participants = Element('participants')
        participant = Element('participant', attrib={'type': 'corprep', 'id': '1', 'title': 'CEO', 'affiliation': 'Company', 'affiliation_entity': 'Entity', 'entity': 'Entity1'})
        participants.append(participant)
        root.append(participants)
        self.parser.parse_participants(root)
        self.assertIn('1', self.parser.transcript_vars.corp_representative_ids)
        self.assertIn('Company', self.parser.transcript_vars.company_name)

    def test_parse_sections(self):
        root = Element('root')
        body = Element('body')
        section = Element('section', attrib={'name': 'MANAGEMENT DISCUSSION SECTION'})
        body.append(section)
        root.append(body)
        self.parser.parse_sections(root)
        # Add assertions based on the expected behavior of _extract_speech
        self.assertTrue(hasattr(self.parser, 'some_expected_attribute'))  # Replace with actual assertion

    def test_parse_company_id(self):
        root = Element('root')
        company = Element('company')
        company.text = 'CompanyID'
        root.append(company)
        self.parser.parse_company_id(root)
        self.assertEqual(self.parser.transcript_vars.company_id, 'CompanyID')

    def test_create_empty_transcript_dict(self):
        empty_dict = self.parser._create_empty_transcript_dict('entity_id', 'sf_date', 'sf_id', 'error')
        self.assertEqual(empty_dict['ENTITY_ID'], 'entity_id')
        self.assertEqual(empty_dict['SF_DATE'], 'sf_date')
        self.assertEqual(empty_dict['SF_ID'], 'sf_id')
        self.assertEqual(empty_dict['ERROR'], 'error')

    def test_parse_ct_xml(self):
        xml_list_row = ['2023-01-01', 'sf_id', 'entity_id', '<root></root>', 'version_id', 'upload_dt', '', '', 'event_datetime']
        parsed_dict = self.parser.parse_ct_xml(xml_list_row)
        self.assertIsInstance(parsed_dict, dict)

if __name__ == '__main__':
    class XMLParserFixture(NutterFixture):
        def __init__(self):
            self.test_case = TestXMLParser()
            super().__init__()

        @tag('test_get_last_upload_dt')
        def run_test_get_last_upload_dt(self):
            self.test_case.setUp()
            self.last_upload_dt = self.test_case.parser.get_last_upload_dt()

        def assertion_test_get_last_upload_dt(self):
            self.test_case.assertEqual(self.last_upload_dt, datetime(2023, 1, 2, 12, 0, 0))

        @tag('test_get_parsed_df')
        def run_test_get_parsed_df(self):
            self.test_case.setUp()
            self.parsed_df = self.test_case.parser.get_parsed_df()

        def assertion_test_get_parsed_df(self):
            self.test_case.assertIsInstance(self.parsed_df, pd.DataFrame)

        @tag('test_get_last_parsed_time')
        def run_test_get_last_parsed_time(self):
            self.test_case.setUp()
            self.last_parsed_time = self.test_case.parser.get_last_pasred_time()

        def assertion_test_get_last_parsed_time(self):
            self.test_case.assertIsInstance(self.last_parsed_time, datetime)

        @tag('test_utc_to_local')
        def run_test_utc_to_local(self):
            self.test_case.setUp()
            utc_dt = datetime(2023, 1, 1, 17, 0, 0, tzinfo=pytz.UTC)
            self.local_dt = self.test_case.parser.utc_to_local(utc_dt, pytz.timezone('America/New_York'))

        def assertion_test_utc_to_local(self):
            self.test_case.assertEqual(self.local_dt, datetime(2023, 1, 1, 12, 0, 0, tzinfo=pytz.timezone('America/New_York')))

        @tag('test_extract_metadata')
        def run_test_extract_metadata(self):
            self.test_case.setUp()
            ns = self.test_case.config.xlmns
            root = Element(f'{ns}root', attrib={'product': 'correct', 'id': '123'})
            meta = SubElement(root, f'{ns}meta')
            title = SubElement(meta, f'{ns}title')
            title.text = 'Test Title'
            date = SubElement(meta, f'{ns}date')
            date.text = '2023-01-01'
            self.metadata = self.test_case.parser.extract_metadata(root)

        def assertion_test_extract_metadata(self):
            status, call_transcript_title, call_transcript_id, date = self.metadata
            self.test_case.assertEqual(status, 'correct')
            self.test_case.assertEqual(call_transcript_title, 'Test Title')
            self.test_case.assertEqual(call_transcript_id, '123')
            self.test_case.assertEqual(date, '2023-01-01')

        @tag('test_parse_participants')
        def run_test_parse_participants(self):
            self.test_case.setUp()
            root = Element('root')
            participants = Element('participants')
            participant = Element('participant', attrib={'type': 'corprep', 'id': '1', 'title': 'CEO', 'affiliation': 'Company', 'affiliation_entity': 'Entity', 'entity': 'Entity1'})
            participants.append(participant)
            root.append(participants)
            self.test_case.parser.parse_participants(root)

        def assertion_test_parse_participants(self):
            self.test_case.assertIn('1', self.test_case.parser.transcript_vars.corp_representative_ids)
            self.test_case.assertIn('Company', self.test_case.parser.transcript_vars.company_name)

        @tag('test_parse_sections')
        def run_test_parse_sections(self):
            self.test_case.setUp()
            ns = self.test_case.config.xlmns
            root = Element(f'{ns}root')
            body = SubElement(root, f'{ns}body')
            section = ET.SubElement(body, f'{ns}section', attrib={'name': 'MANAGEMENT DISCUSSION SECTION'})
            self.test_case.parser.parse_sections(root)

        def assertion_test_parse_sections(self):
            # Add assertions based on the expected behavior of _extract_speech
            # Example assertion (replace with actual expected behavior):
            self.test_case.assertTrue(hasattr(self.test_case.parser, 'some_expected_attribute'))  # Replace with actual assertion

        @tag('test_parse_company_id')
        def run_test_parse_company_id(self):
            self.test_case.setUp()
            ns = self.test_case.config.xlmns 
            root = Element(f'{ns}root')
            meta = SubElement(root, f'{ns}meta')
            companies = SubElement(meta, f'{ns}companies')
            company = SubElement(companies, f'{ns}company')
            company.text = 'CompanyID'
            self.test_case.parser.parse_company_id(root)

        def assertion_test_parse_company_id(self):
            self.test_case.assertEqual(self.test_case.parser.transcript_vars.company_id, ['CompanyID'])

        @tag('test_create_empty_transcript_dict')
        def run_test_create_empty_transcript_dict(self):
            self.test_case.setUp()
            self.empty_dict = self.test_case.parser._create_empty_transcript_dict('entity_id', 'sf_date', 'sf_id', 'error')

        def assertion_test_create_empty_transcript_dict(self):
            self.test_case.assertEqual(self.empty_dict['ENTITY_ID'], 'entity_id')
            self.test_case.assertEqual(self.empty_dict['SF_DATE'], 'sf_date')
            self.test_case.assertEqual(self.empty_dict['SF_ID'], 'sf_id')
            self.test_case.assertEqual(self.empty_dict['ERROR'], 'error')
        
        def create_test_xml(self, ns):
            ns = self.test_case.config.xlmns 
            root = ET.Element(f'{ns}root', attrib={'product': 'correct'})
            meta = ET.SubElement(root, f'{ns}meta')
            title = ET.SubElement(meta, f'{ns}title')
            title.text = 'Test Title'
            date = ET.SubElement(meta, f'{ns}date')
            date.text = '2023-01-01'
            companies = SubElement(meta, f'{ns}companies')
            company = SubElement(companies, f'{ns}company')
            company.text = 'CompanyID'
            return ET.tostring(root, encoding='unicode')


        @tag('test_parse_ct_xml')
        def run_test_parse_ct_xml(self):
            self.test_case.setUp()
            xml_str = self.create_test_xml(self.test_case.parser.config.xlmns)
            xml_list_row = [
            '2023-01-01', 'sf_id', 'entity_id',
            xml_str,
            'version_id', 'upload_dt', '', '', 'event_datetime'
        ]
            self.parsed_dict = self.test_case.parser.parse_ct_xml(xml_list_row)

        def assertion_test_parse_ct_xml(self):
            self.test_case.assertIsInstance(self.parsed_dict, dict)
            self.test_case.assertEqual(self.parsed_dict['CALL_NAME'], 'Test Title')
            self.test_case.assertEqual(self.parsed_dict['DATE'], '2023-01-01')

result = XMLParserFixture().execute_tests()
print(result.to_string())

# COMMAND ----------


