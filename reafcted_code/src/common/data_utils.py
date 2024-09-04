from dataclasses import dataclass

@dataclass
class ProcessedData:
    call_id: str
    entity_id: str
    date: str
    filt_md: list
    filt_qa: list
    labels_filt_md: list
    labels_filt_qa: list
    upload_dt_utc: str
    event_datetime_utc: str


class DataPreprocessor:
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the DataFrame by applying various transformations."""
        try:
            df['CALL_ID'] = df['CALL_ID'].astype(str)
            df['FILT_MD'] = df['FILT_MD'].apply(ast.literal_eval)
            df['FILT_QA'] = df['FILT_QA'].apply(ast.literal_eval)
            df['SENT_LABELS_FILT_MD'] = df['SENT_LABELS_FILT_MD'].apply(ast.literal_eval)
            df['SENT_LABELS_FILT_QA'] = df['SENT_LABELS_FILT_QA'].apply(ast.literal_eval)
            df['LEN_FILT_MD'] = df['FILT_MD'].apply(len)
            df['LEN_FILT_QA'] = df['FILT_QA'].apply(len)
            return df.sort_values(by='UPLOAD_DT_UTC').drop_duplicates(subset=['ENTITY_ID', 'EVENT_DATETIME_UTC'], keep='first')
        except Exception as e:
            logging.error(f"Error in data preprocessing: {e}")
            raise