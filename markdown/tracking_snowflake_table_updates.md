## USER
My SQL table should have been updated with corrected data, but it either wasn't updated or was overwritten with old data, leading to incorrect results when queried. I want to know the history of updates made to the Snowflake table. Is there a way to retrieve this information to identify the root cause? Additionally, can you suggest any other methods to determine the cause of this issue?


## ASSISTANT
To assist you effectively, could you clarify:

- Do you have **Time Travel** enabled in Snowflake for this table? This allows you to query past states of the table.
- Are there **query logs** available (such as using `QUERY_HISTORY` or `ACCESS_HISTORY` views)?
- Do you need help formulating queries to check for updates, deletions, or overwrites?
- Are you looking for a solution within Snowflake only, or would external logging/monitoring be useful?

Let me know, and I'll tailor my approach accordingly.

## USER

I'm not sure if Time Travel is enabled. Please provide solutions for both scenarios.

I don't know if query logs are enabled. Please provide a solution assuming they are enabled.

I need assistance in formulating queries to check for updates, deletions, or overwrites.

I'm open to any kind of solution, whether it's in Snowflake or external. I just want to identify the root cause.

## ASSISTANT
Understood! I will research methods to retrieve the update history of your Snowflake table, covering both cases where **Time Travel** is enabled or not. I'll also provide queries to check for **updates, deletions, or overwrites** and suggest external monitoring/logging solutions if necessary.

I'll get back to you with detailed steps and solutions shortly.

## ASSISTANT
# Investigating Snowflake Table Update History

## 1. Using Time Travel (If Enabled)
Snowflake **Time Travel** lets you query past versions of data if your table is a permanent table with a retention period (default 1 day, up to 90 days on Enterprise edition) [Time Travel & Cloning in Snowflake | by Peggie Mishra | Mar, 2025 | Medium](https://peggie7191.medium.com/time-travel-cloning-in-snowflake-1ddd5a478986#:~:text=Duration%20of%20data%20retrieval%20depends,upon%20multiple%20factors%20like) [Snowflake Time Travel 101: A Comprehensive Overview (2025)](https://www.chaosgenius.io/blog/snowflake-time-travel/#:~:text=Snowflake%20allows%20users%20to%20set,this%20period%20to%2090%20days). This means if incorrect updates occurred recently, you can **“time travel”** to before the change and inspect the data. To use Time Travel, include an **`AT`** or **`BEFORE`** clause in your query’s `FROM` clause, specifying a past timestamp or a query ID [Understanding Snowflake Time Travel [+with Examples and Code]](https://hevodata.com/learn/snowflake-time-travel/#:~:text=specified%20statement%E2%80%99s%20Modifications%2C%20but%20not,including%20them). For example, to retrieve the table’s state as of a specific time or interval, you could run: 

```sql
-- Query the table as it was on a specific timestamp (e.g. March 19, 2025 12:00:00)
SELECT * 
FROM my_db.my_schema.my_table 
AT (TIMESTAMP => '2025-03-19 12:00:00'::TIMESTAMP);
-- Or, using an offset (e.g. 5 minutes ago):
SELECT * 
FROM my_db.my_schema.my_table 
AT (OFFSET => -60*5);  -- 5 minutes back in time
``` 

These queries return the table’s contents at that past point in time [Understanding Snowflake Time Travel [+with Examples and Code]](https://hevodata.com/learn/snowflake-time-travel/#:~:text=specified%20statement%E2%80%99s%20Modifications%2C%20but%20not,including%20them). If you know the **Query ID** of a suspicious statement (from the History tab), you can even query the table **just before** that statement ran. For example: 

```sql
-- Get the table state immediately before a specific query executed
SELECT * 
FROM my_db.my_schema.my_table 
BEFORE (STATEMENT => '8e5d0ca9-005e-44e6-b858-a8f5b37c5726');
``` 

This uses the unique Query ID to rewind the table to the moment right before that query’s changes [Understanding Snowflake Time Travel [+with Examples and Code]](https://hevodata.com/learn/snowflake-time-travel/#:~:text=specified%20statement%E2%80%99s%20Modifications%2C%20but%20not,including%20them). Using these Time Travel queries, you can **compare past vs. present data** to identify what changed. One approach is to export or `JOIN` the results from two different times to pinpoint differences (e.g. find rows that were deleted or updated).

Another powerful technique is to **clone** the table at a past time using Time Travel. For instance, you could run: 

```sql
CREATE OR REPLACE TABLE my_table_clone 
CLONE my_db.my_schema.my_table 
AT (TIMESTAMP => '2025-03-19 12:00:00'::TIMESTAMP);
``` 

This creates a copy of the table exactly as it existed at that time [Understanding Snowflake Time Travel [+with Examples and Code]](https://hevodata.com/learn/snowflake-time-travel/#:~:text=specified%20statement%E2%80%99s%20Modifications%2C%20but%20not,including%20them). You can then query `my_table_clone` and compare it to the current table (using `MINUS` or an `OUTER JOIN` on primary keys) to see which rows differ. Cloning is a convenient way to examine historical data without affecting the original table. Remember that Time Travel data is only available within the retention window – *ensure you perform these analyses before that window expires*. 

## 2. Without Time Travel: Query Logs & Monitoring
If Time Travel is not enabled (e.g. the table is transient or the retention period is 0/expired), you **cannot directly query past data** – once out of the retention window, historical data moves to **Fail-safe** which is not accessible to users [Snowflake Time Travel 101: A Comprehensive Overview (2025)](https://www.chaosgenius.io/blog/snowflake-time-travel/#:~:text=Time%20Travel%20period%20has%20expired,for%20direct%20queries%20or%20restores). 【48†embed_image】Snowflake’s data protection lifecycle illustrates that after the Time Travel period (up to 1–90 days), data goes into a 7-day Fail-safe where **no user queries are allowed** (recovery from Fail-safe requires Snowflake Support) [Snowflake Time Travel 101: A Comprehensive Overview (2025)](https://www.chaosgenius.io/blog/snowflake-time-travel/#:~:text=Time%20Travel%20period%20has%20expired,for%20direct%20queries%20or%20restores). In this scenario, you must turn to **indirect methods** to investigate the issue.

Start by examining **Snowflake’s query logs** to see what happened. Snowflake provides an Account Usage view called **`QUERY_HISTORY`** (and a related table function `INFORMATION_SCHEMA.QUERY_HISTORY()`) that records all queries run, including DML statements. You can search these logs for any **UPDATE, DELETE, MERGE, or INSERT** statements on the target table around the time the data went bad. If using Snowflake Enterprise Edition, the **`ACCESS_HISTORY`** view offers even more detail, showing which tables were touched by each query. By combing through these logs, you can often identify the culprit query or user.

Additionally, consider any **external monitoring** in place. For example, if the table is populated by an ETL job or data pipeline, check that pipeline’s logs to see if it loaded or changed data incorrectly. If you maintain your own audit records (such as an application log of changes or a separate “history” table that tracks updates), review those as well. In absence of Snowflake Time Travel, your **data observability tools** or processes become crucial. It’s a best practice to enable some form of monitoring – for instance, scheduling regular snapshots of critical tables or using a third-party data auditing tool – so that you have backups or logs to compare when something goes wrong. These external references can help reconstruct the sequence of events leading to the bad data.

## 3. Checking Query Logs (QUERY_HISTORY & ACCESS_HISTORY)
Snowflake’s query logs allow you to **audit what queries modified your table**. Using the **`SNOWFLAKE.ACCOUNT_USAGE.QUERY_HISTORY`** view (retained for 365 days), you can filter for any DML statements that targeted your table. For example, the SQL below finds recent inserts, updates, deletes, merges, etc. on a given table:

```sql
SELECT query_start_time, user_name, query_type, rows_inserted, rows_updated, rows_deleted, query_text
FROM SNOWFLAKE.ACCOUNT_USAGE.QUERY_HISTORY
WHERE query_start_time >= '2025-03-19'  -- specify a time range of interest
  AND query_type IN ('INSERT','UPDATE','DELETE','MERGE','TRUNCATE_TABLE','COPY') 
  AND query_text ILIKE '%MY_TABLE%';
``` 

This will list any queries that **inserted/updated/deleted** rows in “MY_TABLE” during the timeframe (the `QUERY_TYPE` field categorizes the query type) [sql - How to get an specific kind/type of query in Snowflake - Stack Overflow](https://stackoverflow.com/questions/75863935/how-to-get-an-specific-kind-type-of-query-in-snowflake#:~:text=select%20start_time%2C%20query_type%2C%20query_text%20from,INSERT%27%2C%20%27DELETE%27%2C%20%27MERGE%27%2C%20%27TRUNCATE_TABLE%27%2C%20%27COPY) [Snowflake Data Audit Trail: A Guide to Data Security](https://www.datasunrise.com/knowledge-center/snowflake-data-audit-trail/#:~:text=schema_name%2C%20table_name%20FROM%20table%28information_schema,UPDATE%27%20OR%20query_type%20%3D%20%27DELETE). The output also shows which user ran the query and how many rows were affected (`ROWS_INSERTED/UPDATED/DELETED`). If you suspect the table was completely replaced or truncated, look for DDL operations too. For example, a `CREATE OR REPLACE TABLE MY_TABLE` would appear with `QUERY_TYPE='CREATE_TABLE'` (and possibly a preceding `DROP TABLE`). Including `COPY` and `TRUNCATE_TABLE` in the filter (as above) ensures you catch bulk loading or truncation events as well. Adjust the time window and filters as needed to pinpoint the exact query – often the **timestamp of the anomaly** is your clue for when to look.

**Access History (Enterprise only)**: If you have Enterprise Edition, the `SNOWFLAKE.ACCOUNT_USAGE.ACCESS_HISTORY` view provides an even more granular audit. Each query’s record in ACCESS_HISTORY contains JSON arrays of **objects accessed or modified** by that query. By flattening these arrays, you can search for your specific table. For instance, to find all queries that modified `MY_TABLE` in the last week:

```sql
SELECT ah.query_start_time, ah.user_name, ah.query_text
FROM SNOWFLAKE.ACCOUNT_USAGE.ACCESS_HISTORY AS ah,
     LATERAL FLATTEN(input => ah.objects_modified) AS mod
WHERE mod.value:objectName = 'MY_DB.MY_SCHEMA.MY_TABLE'
  AND ah.query_start_time >= DATEADD(day, -7, CURRENT_TIMESTAMP())
ORDER BY ah.query_start_time;
``` 

In this example, we **flatten** the `objects_modified` array, which contains entries for any tables that were written to by the query [Snowflake: How to flatten the ACCESS_HISTORY View](https://clark-perucho.medium.com/how-to-flatten-snowflakes-access-history-view-9766f6366d52#:~:text=Snowflake%3A%20How%20to%20flatten%20the,performed%20against%2C%20this%20is). We filter on our fully qualified table name. The result will show any query (and the user and time) that altered that table in the past 7 days. You could similarly flatten `base_objects_accessed` to see even read access, but focusing on `objects_modified` isolates **write operations**. Using Access History in combination with Query History is a best practice for a thorough investigation: Query History gives you the raw SQL and row counts, while Access History confirms which objects were impacted by each query. 

*Best practices:* When querying these log views, try to **narrow the time range** and **specific keywords** to avoid scanning the entire year of history (which can be slow). For example, if you noticed the data issue today, start by looking at queries from the last day or two on that table. Also, remember that Account Usage views have a lag (up to ~45 minutes) for recent queries [Using the Snowflake Query History: 9 Practical Examples](https://select.dev/posts/snowflake-query-history#:~:text=Using%20the%20Snowflake%20Query%20History%3A,data%20much%20sooner%20in), so very recent operations might not appear immediately in `ACCOUNT_USAGE.QUERY_HISTORY` – in such cases, you can use the **Information Schema** (`INFORMATION_SCHEMA.QUERY_HISTORY()` table function) which shows your session’s last 7 days of history in near-real-time. By systematically inspecting these logs, you should be able to identify any erroneous UPDATE or DELETE that was run, who ran it, and when.

## 4. Other Tools and Features (Streams, Change Tracking, etc.)
Beyond Time Travel and logs, Snowflake offers features to **proactively track data changes** which can help diagnose issues. One such feature is **Snowflake Streams** – a *change data capture* mechanism. A **Stream** on a table will record every row-level change (insert, update, delete) made to that table, along with metadata about each change [Introduction to Streams | Snowflake Documentation](https://docs.snowflake.com/en/user-guide/streams-intro#:~:text=A%20stream%20object%20records%20data,introduces%20key%20concepts%20for%20change). By creating a stream (e.g. `CREATE STREAM my_table_stream ON TABLE my_table;`), you can later query the stream like a pseudo-table to see all changes that have occurred since the stream’s last consumption. Each row in the stream represents a change with indicators such as `METADATA$ACTION` (INSERT/DELETE) and `METADATA$ISUPDATE` flags. If a stream had been set up on the table *before* the issue, you could simply query `SELECT * FROM my_table_stream;` to retrieve the history of changes and identify the bad update. Even if you didn’t have one already, consider creating a stream now (or enable one after restoring correct data) to capture any future anomalies.

Another related capability is **Change Tracking**. Snowflake tables can be created or altered with `CHANGE_TRACKING = TRUE`, which allows use of the **CHANGES clause** in SQL queries. The CHANGES clause lets you query the *change tracking metadata* for a table over a time interval **without** a pre-created stream [CHANGES | Snowflake Documentation](https://docs.snowflake.com/en/sql-reference/constructs/changes#:~:text=Change%20tracking%20must%20be%20enabled,in%20this%20topic). In other words, if change tracking was enabled, you can directly ask Snowflake “what changed in this table between time X and time Y.” For example, after enabling change tracking on a table, you might run:

```sql
SELECT * 
FROM my_db.my_schema.my_table 
CHANGES (INFORMATION => DEFAULT) 
AT (TIMESTAMP => '2025-03-19 00:00:00'::TIMESTAMP) 
END (TIMESTAMP => '2025-03-20 00:00:00'::TIMESTAMP);
``` 

This would return the net set of row changes (inserts/updates/deletes) that occurred in the table between March 19 and 20, 2025. (The `INFORMATION => DEFAULT` option requests all change types [CHANGES | Snowflake Documentation](https://docs.snowflake.com/en/sql-reference/constructs/changes#:~:text=Change%20tracking%20must%20be%20enabled,in%20this%20topic). There is also `APPEND_ONLY` if you only care about inserts.) Note that **change tracking must be enabled on the table beforehand** to use this [CHANGES | Snowflake Documentation](https://docs.snowflake.com/en/sql-reference/constructs/changes#:~:text=Change%20tracking%20must%20be%20enabled,in%20this%20topic), and it relies on Time Travel under the covers (honoring the retention period). If you anticipate needing to audit changes regularly, enabling change tracking can be very useful for on-demand diffs.

Aside from Snowflake’s native features, you may employ **external tools** to help catch and diagnose issues:
- **Data observability platforms** (like Monte Carlo, BigEye, etc.) can monitor Snowflake data and alert on anomalies or detect changes in record counts.
- Building a simple **audit table**: e.g., have your ETL processes insert a row into an “audit_log” table for every update they execute (recording the table, timestamp, user, and number of rows changed). This provides a manual trail outside the main table.
- **Periodic backups or snapshots**: Taking nightly snapshots of important tables (e.g., using `CREATE … CLONE` or unloading data to cloud storage) can help you compare yesterday’s data to today’s. This is a last resort for when Time Travel is insufficient, but it can save the day if something slips through.
- **Row-based auditing with Streams**: As mentioned, a stream can feed an audit table. For instance, you could have a task that periodically reads from `my_table_stream` and inserts those change records into an “audit_changes” table for permanent retention. This way you’d have a full history of how each row changed over time, even beyond the normal Time Travel window.

In summary, the best practice for investigating unexpected data changes in Snowflake is to leverage **Time Travel** first (for a quick look at before/after states), then comb through **Query History and Access History logs** to find the responsible operations. If Time Travel isn’t available, those logs and any external audit mechanisms become your primary tools. Going forward, consider enabling features like Streams or Change Tracking on critical tables to **continuously track changes**, and implement regular auditing so that any future data issues can be identified and resolved with minimal guesswork. With the combination of these methods, you can confidently trace how incorrect updates happened and put measures in place to prevent or catch them in the future. 

**Sources:** Snowflake Documentation on Time Travel [Understanding Snowflake Time Travel [+with Examples and Code]](https://hevodata.com/learn/snowflake-time-travel/#:~:text=specified%20statement%E2%80%99s%20Modifications%2C%20but%20not,including%20them) [Understanding Snowflake Time Travel [+with Examples and Code]](https://hevodata.com/learn/snowflake-time-travel/#:~:text=specified%20statement%E2%80%99s%20Modifications%2C%20but%20not,including%20them), Snowflake Account Usage views (QUERY_HISTORY, ACCESS_HISTORY) [Snowflake Data Audit Trail: A Guide to Data Security](https://www.datasunrise.com/knowledge-center/snowflake-data-audit-trail/#:~:text=schema_name%2C%20table_name%20FROM%20table%28information_schema,UPDATE%27%20OR%20query_type%20%3D%20%27DELETE) [sql - How to get an specific kind/type of query in Snowflake - Stack Overflow](https://stackoverflow.com/questions/75863935/how-to-get-an-specific-kind-type-of-query-in-snowflake#:~:text=select%20start_time%2C%20query_type%2C%20query_text%20from,INSERT%27%2C%20%27DELETE%27%2C%20%27MERGE%27%2C%20%27TRUNCATE_TABLE%27%2C%20%27COPY), and best-practice guides on auditing Snowflake data changes [Introduction to Streams | Snowflake Documentation](https://docs.snowflake.com/en/user-guide/streams-intro#:~:text=A%20stream%20object%20records%20data,introduces%20key%20concepts%20for%20change) [CHANGES | Snowflake Documentation](https://docs.snowflake.com/en/sql-reference/constructs/changes#:~:text=Change%20tracking%20must%20be%20enabled,in%20this%20topic).