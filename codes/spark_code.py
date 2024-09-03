# multipliers for converting word monetary values to numeric
multipliers = {
    'billion': 1000000000,
    'million': 1000000,
    'thousand': 1000,
    'b': 1000000000,
    'm': 1000000,
    'k': 1000
}

# ranges for the min/max of hourly wages and annual salaries
hourly_min = 10
hourly_max = 150
hourly_range = (hourly_min, hourly_max)
salary_min = 20000
salary_max = 950000
salary_range = (salary_min, salary_max)

# Regex patterns
salary_pattern = r'(\$[,\d]+\.?\d* [Th]housand)|(\$[,\d]+\.?\d* [MBmb]illion)|(\$[,\d]+\.?\d*-[,\d]+\.?\d*[mbkMBK]?)|(\$[,\d]+\.?\d*\s*-\s*[,\d]+\.?\d*[mbkMBK]?)|(\$[,\d]+\.?\d*[mbkMBK]?)'
high_school = r'[Hh]igh [Ss]chool [Dd]iploma|[Hh]igh [Ss]chool [Dd]egree|[Hh]igh [Ss]chool|GED|CSSD|HSE|ged|cssd|hse|'
associate = r'[Aa]ssociate[\']?s [Dd]egree|[Aa]ssociate [Dd]egree|[Aa]ssociate[\']?s|'
bachelor = r'[Bb]achelor [Oo]f|[Bb]achelor[\']?s [Dd]egree|[Bb]achelor[\']?s|[Bb]achelor|[^a-zA-Z]B[.]?[AS][.]?[^a-zA-Z]|[Cc]ollege [Dd]egree|[Cc]ollege [Dd]iploma|[Uu]niversity [Dd]egree|[Uu]niversity [Dd]iploma|[Aa]ccredited [Cc]ollege|[Cc]ollege|[^a-zA-Z]B[.]?S[.]?N[.]?[^a-zA-Z]|'
master = r'[Mm]aster[\']?s [Dd]egree|[Mm]aster [Oo]f|[Mm]aster[\']?s|[Mm]aster|[^a-zA-Z]M[.]?B[.]?A[.]?[^a-zA-Z]|'
doctoral = r'[Dd]octorate|[Pp][Hh][.]?[Dd][.]?'
degree_pattern = high_school + associate + bachelor + master + doctoral
yoe_pattern = r'1[012345][\+]? year[a-zA-Z\-,\(\) ]*experience|[1-9][\+]? year[a-zA-Z\-,\(\) ]*experience'


degree_pattern = (
    r"(?i)\b(?:high school diploma|high school degree|high school|ged|cssd|hse|"
    r"associate's degree|associate degree|associate's|"
    r"bachelor of|bachelor's degree|bachelor's|bachelor|"
    r"college degree|college diploma|university degree|university diploma|"
    r"accredited college|college|"
    r"master's degree|master of|master's|master|"
    r"doctorate|ph.d|"
    r"B\.?A\.?|B\.?S\.?|M\.?B\.?A\.?|M\.?S\.?)\b"
)

# given list of strings representing monetary values, returns single numeric value representing wage/salary
def extract_salary(dollar_strs, hourly_range=hourly_range, salary_range=salary_range, agg='avg'):
  if len(dollar_strs) == 0:
    return None
  try:
    salary = []
    for dollar_str in dollar_strs:
      dollar_str = dollar_str.strip('$').replace(',', '')
      #$100,00-120,000
      if '-' in dollar_str:
        dollar_str = dollar_str.split('-')[1].strip()
      value = 0
      # $1 billion, $100 million
      if ' ' in dollar_str:
        value = float(dollar_str.split()[0]) * multipliers[dollar_str.split()[1]]
      # $1k, $90k
      elif 'k' == dollar_str[-1] or 'm'==dollar_str[-1] or 'b'==dollar_str[-1]:
        value = float(dollar_str[:-1]) * multipliers[dollar_str[-1]]
      else:
        value = float(dollar_str)
      if hourly_range[0]<=value<=hourly_range[1] or salary_range[0]<=value<=salary_range[1]:
        salary.append(value)
    # currently takes avg but could change to min, max.
    if agg == 'max':
      return max(salary)
    elif agg == 'min':
      return min(salary)
    elif agg == 'avg':
      return sum(salary)/len(salary)
    else:
      return None
  except:
    return None


# converts worded YOE string to numeric value
def yoe_to_value(yoe):
  if len(yoe) == 0:
    return 0
  else:
    yoe_values = []
    for y in yoe:
      y = re.sub(r'[^A-Za-z0-9 ]+', '', y)
      yoe_values += [int(val) for val in re.findall('1[012]|[1-9]', y)]
    return max(yoe_values)

# converts education list to an age
def education_to_age(ed):
  if not ed:
    return 21
  else:
    ed_ages = []
    for e in ed:
      if 'high school' in e or 'ged' in e or 'cssd' in e or 'hse' in e:
        ed_ages.append(19)
      elif 'associate' in e:
        ed_ages.append(21)
      elif 'bachelor' in e or 'b.a.' in e or 'ba' in e or 'college' in e or 'university' in e or 'bsn' in e or 'b.s.n' in e:
        ed_ages.append(23)
      elif 'master' in e or 'mba' in e or 'm.b.a.' in e:
        ed_ages.append(26)
      else:
        ed_ages.append(30)
    return max(ed_ages)

# Register UDFs
extract_salary_udf = udf(extract_salary, FloatType())
yoe_to_value_udf = udf(yoe_to_value, IntegerType())
education_to_age_udf = udf(education_to_age, IntegerType())

naics_to_recode = pd.read_excel('/dbfs/mnt/access_work/job_listing/naics_to_recode.xlsx')
naics_to_recode_dict = {}
for i in range(len(naics_to_recode)):
  naics = str(naics_to_recode['2022_naics'][i]).split(', ') + str(naics_to_recode['2017_naics'][i]).split(', ')
  recode = naics_to_recode['recode_52'][i]
  for code in naics:
    naics_to_recode_dict[code] = recode

def convert_to_recode(code):
  try:
    return naics_to_recode_dict[str(code)]
  except:
    return None

convert_to_recode_udf = udf(convert_to_recode, StringType())


from pyspark.sql.types import StringType

broadcast_pattern = spark.sparkContext.broadcast(degree_pattern)

# Define a function that uses the broadcasted regex pattern
def extract_degree(job_description):
  pattern = broadcast_pattern.value  # Access the broadcasted pattern
  match = re.search(pattern, job_description)
  return match.group(0) if match else None

# Register the function as a UDF
extract_degree_udf = udf(extract_degree, StringType())

job_comp_code_df = job_comp_code_df.withColumn('degrees_str', extract_degree_udf(col('JOB_DESCRIPTION')))
job_comp_code_df = job_comp_code_df.withColumn('high_school', when(col('degrees_str').rlike(high_school), True).otherwise(False))
job_comp_code_df = job_comp_code_df.withColumn('associate', when(col('degrees_str').rlike(associate), True).otherwise(False))
job_comp_code_df = job_comp_code_df.withColumn('bachelor', when(col('degrees_str').rlike(bachelor), True).otherwise(False))
job_comp_code_df = job_comp_code_df.withColumn('master', when(col('degrees_str').rlike(master), True).otherwise(False))
job_comp_code_df = job_comp_code_df.withColumn('doctorate', when(col('degrees_str').rlike(doctoral), True).otherwise(False))
job_comp_code_df.show(5)

# Define a UDF to extract years of experience
def extract_years_of_experience(job_description):
  try:
    if job_description is not None:
        match = re.search(yoe_pattern, job_description)
        return match.group(0) if match else None
    return None
  except Exception as e:
    print(e)
  
extract_years_of_experience_udf = udf(extract_years_of_experience, StringType())

job_comp_code_df = job_comp_code_df.withColumn('yoe_str', regexp_extract(col('JOB_DESCRIPTION'), yoe_pattern, 0))
# job_comp_code_df = job_comp_code_df.withColumn(
#     'yoe_str',
#     when(col('JOB_DESCRIPTION').isNotNull(), regexp_extract(col('JOB_DESCRIPTION'), yoe_pattern, 0)).otherwise(None)
# )
job_comp_code_df.limit(10).collect()
print('doctorate')
job_comp_code_df = job_comp_code_df.withColumn('yoe_value', yoe_to_value_udf(col('yoe_str')))
job_comp_code_df.limit(10).collect()
print('doctorate')
job_comp_code_df = job_comp_code_df.withColumn('age', education_to_age_udf(col('degrees_str')) + col('yoe_value'))
job_comp_code_df.limit(10).collect()
print('doctorate')
job_comp_code_df = job_comp_code_df.withColumn('recode', convert_to_recode_udf(col('NAICS_CODE')))
job_comp_code_df.limit(10).collect()
print('doctorate')