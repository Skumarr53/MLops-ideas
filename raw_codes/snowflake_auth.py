# centralized_nlp_package/data_access/snowflake_utils.py
import os
from dotenv import load_dotenv
from cryptography.fernet import Fernet
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
import pandas as pd
from snowflake.connector import connect
from typing import Any
from pathlib import Path
from loguru import logger
from centralized_nlp_package import config
import re

load_dotenv()

ENV = os.getenv('ENVIRONMENT', 'development')


def encrypt_message(message: str) -> bytes:
    """
    Encrypts the provided message using Fernet symmetric encryption.
    """
    encoded_message = message.encode()
    fernet_obj = Fernet(os.getenv('FERNET_KEY'))
    encrypted_message = fernet_obj.encrypt(encoded_message)
    return encrypted_message


def decrypt_message(encrypted_message: bytes) -> str:
    """
    Decrypts the provided encrypted message using Fernet symmetric encryption.
    """
    fernet_obj = Fernet(os.getenv('FERNET_KEY'))
    decrypted_message = fernet_obj.decrypt(encrypted_message)
    return decrypted_message.decode()


def get_private_key() -> str:
    """
    Retrieves and processes the Snowflake private key from AKV.
    
    Returns:
        str: The private key in a format suitable for Snowflake authentication.
    """
    # Retrieve encrypted private key and password from AKV
    key_file = dbutils.secrets.get(scope="id-secretscope-dbk-pr4707-prod-work", key="eds-prod-quant-key")
    pwd = dbutils.secrets.get(scope="id-secretscope-dbk-pr4707-prod-work", key="eds-prod-quant-pwd")
    
    # Load the private key using the retrieved password
    p_key = serialization.load_pem_private_key(
        key_file.encode('ascii'),
        password=pwd.encode(),
        backend=default_backend()
    )
    
    # Serialize the private key to PEM format without encryption
    pkb = p_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()
    )
    
    # Decode and clean the private key string
    pkb = pkb.decode("UTF-8")
    pkb = re.sub("-*(BEGIN|END) PRIVATE KEY-*\n", "", pkb).replace("\n", "")
    
    return pkb


def get_snowflake_connection():
    """
    Establishes a connection to Snowflake using key pair authentication.
    
    Returns:
        conn: A Snowflake connection object.
    """
    private_key = get_private_key()
    
    snowflake_config = {
        'user': decrypt_message(config.lib_config.development.snowflake.user),
        'account': config.lib_config.development.snowflake.account,
        'private_key': private_key,
        'database': config.lib_config.development.snowflake.database,
        'schema': config.lib_config.development.snowflake.schema,
        'timezone': "spark",
        'role': config.lib_config.development.snowflake.role  # Optional if needed
    }

    try:
        conn = connect(**snowflake_config)
        logger.info("Successfully connected to Snowflake using key pair authentication.")
        return conn
    except Exception as e:
        logger.error(f"Failed to connect to Snowflake: {e}")
        raise


def read_from_snowflake(query: str) -> pd.DataFrame:
    """
    Executes a SQL query on Snowflake and returns the result as a pandas DataFrame.
    """
    logger.info("Establishing connection to Snowflake.")
    conn = get_snowflake_connection()
    try:
        logger.debug(f"Executing query: {query}")
        df = pd.read_sql(query, conn)
        logger.info("Query executed successfully.")
    except Exception as e:
        logger.error(f"Error executing query: {e}")
        raise
    finally:
        conn.close()
        logger.info("Snowflake connection closed.")
    return df


def write_to_snowflake(df: pd.DataFrame, table_name: str, if_exists: str = 'append') -> None:
    """
    Writes a pandas DataFrame to a Snowflake table.
    """
    logger.info("Establishing connection to Snowflake.")
    conn = get_snowflake_connection()

    try:
        logger.info(f"Writing DataFrame to Snowflake table: {table_name}")
        df.to_sql(
            table_name,
            con=conn,
            if_exists=if_exists,
            index=False,
            method='multi'  # Use multi-row inserts for efficiency
        )
        logger.info(f"DataFrame written successfully to {table_name}.")
    except Exception as e:
        logger.error(f"Error writing DataFrame to Snowflake: {e}")
        raise
    finally:
        conn.close()
        logger.info("Snowflake connection closed.")


def get_snowflake_options() -> dict:
    """
    Returns a dictionary of Snowflake options for Spark connections using key pair authentication.
    """
    private_key = get_private_key()

    snowflake_options = {
        'sfURL': f"{config.lib_config.development.snowflake.account}.snowflakecomputing.com",
        'sfUser': decrypt_message(config.lib_config.development.snowflake.user),
        'private_key': private_key,
        'sfDatabase': config.lib_config.development.snowflake.database,
        'sfSchema': config.lib_config.development.snowflake.schema,
        "sfTimezone": "spark",
        'sfRole': config.lib_config.development.snowflake.role  # Optional if needed
    }
    return snowflake_options


def read_from_snowflake_spark(query: str, spark) -> Any:
    """
    Executes a SQL query on Snowflake and returns the result as a Spark DataFrame.
    """
    logger.info("Reading data from Snowflake using Spark.")

    snowflake_options = get_snowflake_options()

    try:
        logger.debug(f"Executing query: {query}")
        df_spark = spark.read.format("snowflake") \
            .options(**snowflake_options) \
            .option("query", query) \
            .load()
        logger.info("Query executed successfully and Spark DataFrame created.")
    except Exception as e:
        logger.error(f"Error executing query on Snowflake: {e}")
        raise

    return df_spark


def write_to_snowflake_spark(df, table_name: str, mode: str = 'append') -> None:
    """
    Writes a Spark DataFrame to a Snowflake table.
    """
    logger.info(f"Writing Spark DataFrame to Snowflake table: {table_name}.")
    snowflake_options = get_snowflake_options()

    try:
        df.write.format("snowflake") \
            .options(**snowflake_options) \
            .option("dbtable", table_name) \
            .mode(mode) \
            .save()
        logger.info(f"DataFrame written successfully to {table_name}.")
    except Exception as e:
        logger.error(f"Error writing Spark DataFrame to Snowflake: {e}")
        raise
