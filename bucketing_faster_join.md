```
data2 = [
    ('A1B2C3D4E5', 100000),
    ('F6G7H8I9J0', 120000),
    ('K1L2M3N4O5', 110000),
    ('P6Q7R8S9T0', 90000)
]

# Step 3: Create DataFrames
df1 = spark.createDataFrame(data1, ["JOB_HASH", "Job_Title"])
df2 = spark.createDataFrame(data2, ["JOB_HASH", "Salary"])

# Step 4: Write DataFrames to bucketed tables
# Specify number of buckets and the column to bucket by (JOB_HASH)
num_buckets = 4
bucketed_df1 = df1.write.bucketBy(num_buckets, "JOB_HASH").saveAsTable("bucketed_jobs1")
bucketed_df2 = df2.write.bucketBy(num_buckets, "JOB_HASH").saveAsTable("bucketed_jobs2")

# Step 5: Read the bucketed tables back into DataFrames
bucketed_jobs1 = spark.read.table("bucketed_jobs1")
bucketed_jobs2 = spark.read.table("bucketed_jobs2")

# Step 6: Perform join operation using the bucketed DataFrames
joined_df = bucketed_jobs1.join(bucketed_jobs2, "JOB_HASH")


```