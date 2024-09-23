## Database 

## SnowFlakeDBUtility

This class provides the main functionalities for interacting with Snowflake databases.

## Methods

### `__init__(self, environment_str, role_str, schema_str)`
Initializes the SnowFlakeDBUtility object.

### `get_url(self, environment_str)`
Returns the Snowflake URL based on the environment.

### `get_snowflake_auth_options(self)`
Generates and returns a dictionary of authentication options for connecting to Snowflake.

### `utc_to_local(self, utc_date, tz_val)`
Converts UTC time to local time zone.

### `generate_key(self)`
Generates a new Fernet key.

### `encrypt_message(self, message)`
Encrypts a message.

### `decrypt_message(self, encrypted_message)`
Decrypts a message.

### `perform_encryption(self)`
Reads the credentials file, encrypts the credentials, and creates a JSON request using the encrypted credentials.

### `write_to_snowflake_table(self, df, tablename)`
Writes a DataFrame to a Snowflake table.

### `read_from_snowflake(self, query)`
Reads data from Snowflake.

### `query_snowflake(self, query)`
Executes a query on Snowflake.

### `truncate_or_merge_table(self, query)`
Truncates or merges a table in Snowflake.