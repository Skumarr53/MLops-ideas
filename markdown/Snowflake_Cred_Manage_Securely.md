# Securely Managing Snowflake Credentials in Databricks

Managing credentials securely across different environments in Databricks is crucial for maintaining security and compliance. Here’s a step-by-step guide to securely storing and loading credentials for Snowflake across multiple environments in Databricks.

## Step-by-Step Guide

### Step 1: Use Databricks Secrets

Databricks provides a secure way to store and access credentials using its Secrets management feature. You can create secret scopes and store your credentials within these scopes.

#### Create a Secret Scope:
1. Go to the Databricks workspace.
2. Navigate to the "Data" tab.
3. Click on "Create" and select "Secret Scope".
4. Follow the prompts to create a new secret scope (e.g., `snowflake_scope`).

#### Store Secrets:
Use the Databricks CLI to store your Snowflake credentials in the secret scope.

1. Install the Databricks CLI if you haven't already:
bash
   pip install databricks-cli
2. Configure the Databricks CLI with your workspace details:
bash
   databricks configure --token
3. Store the secrets:
``` bash
   databricks secrets put --scope snowflake_scope --key snowflake_username
   databricks secrets put --scope snowflake_scope --key snowflake_password
   databricks secrets put --scope snowflake_scope --key snowflake_account
```
### Step 2: Access Secrets in Your Notebook

You can access the stored secrets in your Databricks notebooks using the `dbutils.secrets` API.

#### Access Secrets:
``` python
snowflake_username = dbutils.secrets.get(scope="snowflake_scope", key="snowflake_username")
snowflake_password = dbutils.secrets.get(scope="snowflake_scope", key="snowflake_password")
snowflake_account = dbutils.secrets.get(scope="snowflake_scope", key="snowflake_account")
```

### Step 3: Configure Snowflake Connection

Use the retrieved secrets to configure your Snowflake connection.

#### Install Snowflake Connector:
bash
pip install snowflake-connector-python
#### Create a Connection Function:

``` python
import snowflake.connector

def get_snowflake_connection():
    conn = snowflake.connector.connect(
        user=snowflake_username,
        password=snowflake_password,
        account=snowflake_account
    )
    return conn
```
### Step 4: Handle Multiple Environments

You can use environment variables or Databricks environment tags to differentiate between environments (e.g., dev, stg, prod).

#### Set Environment Variables:
In your Databricks cluster configuration, set environment variables for each environment. For example, set `ENV` to `dev`, `stg`, or `prod`.

#### Access Environment Variables:
``` python
import os

env = os.getenv("ENV", "dev")
if env == "dev":
    snowflake_username = dbutils.secrets.get(scope="snowflake_scope", key="snowflake_username_dev")
    snowflake_password = dbutils.secrets.get(scope="snowflake_scope", key="snowflake_password_dev")
    snowflake_account = dbutils.secrets.get(scope="snowflake_scope", key="snowflake_account_dev")
elif env == "stg":
    snowflake_username = dbutils.secrets.get(scope="snowflake_scope", key="snowflake_username_stg")
    snowflake_password = dbutils.secrets.get(scope="snowflake_scope", key="snowflake_password_stg")
    snowflake_account = dbutils.secrets.get(scope="snowflake_scope", key="snowflake_account_stg")
elif env == "prod":
    snowflake_username = dbutils.secrets.get(scope="snowflake_scope", key="snowflake_username_prod")
    snowflake_password = dbutils.secrets.get(scope="snowflake_scope", key="snowflake_password_prod")
    snowflake_account = dbutils.secrets.get(scope="snowflake_scope", key="snowflake_account_prod")
```
### Step 5: Use the Connection

Use the connection function to interact with Snowflake.

#### Query Snowflake:
``` python
conn = get_snowflake_connection()
cursor = conn.cursor()
cursor.execute("SELECT CURRENT_USER(), CURRENT_ACCOUNT(), CURRENT_REGION()")
row = cursor.fetchone()
print(row)
cursor.close()
conn.close()
```
## Summary
- **Create Secret Scopes**: Store your Snowflake credentials securely using Databricks secret scopes.
- **Access Secrets**: Use the `dbutils.secrets` API to access the stored secrets in your notebooks.
- **Configure Connection**: Create a function to configure the Snowflake connection using the retrieved secrets.
- **Handle Environments**: Use environment variables or tags to handle multiple environments and retrieve the appropriate credentials.
- **Use the Connection**: Use the connection function to interact with Snowflake securely.

By following these steps, you can securely manage and access Snowflake credentials across different environments in Databricks.
This Markdown format organizes the content into sections, making it easier to read and follow.



```
def singleton(cls):
    """
    A decorator to make a class a Singleton by ensuring only one instance exists.
    """
    instances = {}

    @wraps(cls)
    def get_instance(*args, **kwargs):
        global _spark_session, _sfUtils
        if cls not in instances:
            #logger.info(f"Creating a new instance of {cls.__name__}")
            instances[cls] = cls(*args, **kwargs)
            _spark_session = instances[cls].spark
            _sfUtils = instances[cls].sfUtils
        # else:
            #logger.info(f"Using existing instance of {cls.__name__}")
        return instances[cls]

    return get_instance


@singleton
class SparkSessionManager:
    """
    Singleton class to manage a single Spark session across the application.
    """

    def __init__(self, app_name: str = "SnowflakeIntegration") -> None:
        """
        Initializes the Spark session and configures Snowflake integration.

        Args:
            app_name (str): The name of the Spark application.
        """
        self.spark: SparkSession = SparkSession.builder \
            .appName(app_name) \
            .getOrCreate()

        self.setup_spark()

    def setup_spark(self) -> None:
        """
        Configures Spark session settings and Snowflake integration.
        """
        global _sfUtils
        #logger.info("Configuring Spark session settings...")

        sc = self.spark.sparkContext

        # Configure Snowflake pushdown session
        self.sfUtils = sc._jvm.net.snowflake.spark.snowflake.Utils
        # sc._jvm.net.snowflake.spark.snowflake.SnowflakeConnectorUtils.enablePushdownSession(self.spark)
        sc._jvm.net.snowflake.spark.snowflake.SnowflakeConnectorUtils.enablePushdownSession(sc._jvm.org.apache.spark.sql.SparkSession.builder().getOrCreate())

        # Set the default timezone to UTC
        zone = sc._jvm.java.util.TimeZone
        zone.setDefault(sc._jvm.java.util.TimeZone.getTimeZone("UTC"))

        #logger.info("Spark session configured and Snowflake integration enabled.")


def with_spark_session(func):
    """
    Decorator to ensure that the Spark session is initialized before function execution.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        global _spark_session
        if _spark_session is None:
            #logger.info("Spark session not initialized. Initializing now...")
            SparkSessionManager(app_name="SnowflakeIntegration")
        else:
            print("Spark session already initialized; reusing existing session.")
        return func(*args, **kwargs)
    return wrapper
```
