
sfUtils = sc._jvm.net.snowflake.spark.snowflake.Utils
sc = spark.sparkContext
sc._jvm.net.snowflake.spark.snowflake.SnowflakeConnectorUtils.enablePushdownSession(sc._jvm.org.apache.spark.sql.SparkSession.builder().getOrCreate())
zone = sc._jvm.java.util.TimeZone
zone.setDefault(sc._jvm.java.util.TimeZone.getTimeZone("UTC"))

class SnowFlakeDBUtilityCTS:

  def __init__(self, schema, srcdbname):
    self.schema = schema
    self.srcdbname = srcdbname
    self.options = self.get_snowflake_options()

  def __repr__(self):
    return f"Schema & DB in object : {self.schema} & {self.srcdbname} repectively"
  
  def get_private_key(self) -> str:
    """
    Retrieves and processes the Snowflake private key from AKV.
    
    Returns:
        str: The private key in a format suitable for Snowflake authentication.
    """
    # Retrieve encrypted private key and password from AKV
    key_file = dbutils.secrets.get(scope="id-secretscope-dbk-pr4707-prod-work", key=config.snowflake_key)
    pwd = dbutils.secrets.get(scope="id-secretscope-dbk-pr4707-prod-work", key=config.snowflake_pwd)
    
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
  
  def get_snowflake_options(self):
    private_key = self.get_private_key()

    options = {
            "sfUrl" : "voya.east-us-2.privatelink.snowflakecomputing.com",
            "sfUser" : config.snowflake_service_id, ## service id for SA_EDS_PROD_QUANT
            "pem_private_key": private_key,  
            "sfDatabase" : "EDS_PROD",
            "sfSchema" : config.schema,
            "sfWarehouse" : "WH_EDS_PROD_READ",
            "sfRole": config.snowflake_role,
            "sfTimezone" : "spark"
            }
    return options

  
  def read_from_snowflake(self,query): 

    df = spark.read \
              .format("snowflake") \
              .options(**self.options) \
              .option("query",  query) \
              .load()

    return df
  
  def write_to_snowflake_table(self, df, tablename):

    df\
    .write.format("snowflake")\
    .options(**self.options)\
    .option("dbtable", tablename)\
    .mode("append")\
    .save()
    result="Load Complete"
    return result
  
  def truncate_or_merge_table(self, query):

    df=sfUtils.runQuery(self.options, query)
    result="Truncate or Merge Complete"
    return result