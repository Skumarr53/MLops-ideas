We would like to discuss upgrading an existing repository, which includes renaming it, adding additional branches, and implementing security settings found in other codebase repositories. I will set up a call, and it would be helpful if someone could join to address these needs.


Bea, 
**Company Similarity** 
- The CS pipelines have been deployed to the production environment and will be monitored daily until this week. 
- As discussed, I will create a presentation deck for the ops team, that will be scheduled for next week. 
 
**Topic Modeling** 
- I am nearly finished refactoring the common code. The remaining tasks include writing test cases and updating the documentation for the newly added functions. 
- I am designing a configuration management system to manage the parameters used in the code, which will help improve the maintainability of the common code. 
- Regarding topic modeling, I will continue refactoring the dictionary-based processing pipeline code. 
 
**Optimize NLI Inference** 
- I had a discussion with Yujing about the NLI pipeline validation, there are some discrepancies in the final results. It appears that the logic which   processes inference results in my PySpark code is missing something. I am closely examining that part of the code to resolve the issue.




Certainly! Here’s a brief summary for your team regarding the provided unit testing module:

---

### Purpose of Unit Testing
Unit testing is a software testing practice where individual components or units of code are tested in isolation to ensure they function correctly. The primary goals of unit testing are to:
- Validate that each unit performs as expected.
- Identify bugs early in the development process.
- Ensure code reliability and facilitate safe refactoring by providing a safety net that existing functionality remains intact.

### Overview of the Code
The code defines a test suite for the  `TokenInjection`  class, which is responsible for processing tokens related to entities. The suite is implemented using  `unittest.TestCase`  and  `NutterFixture` , indicating it is designed to work within a specific testing framework. The test suite includes three tests, each targeting specific functionalities of the  `TokenInjection`  class.

### Explanation of Each Test

#### Test 1:  `insert_entity_id_into_tokens` 
- **What**: This test verifies that the  `insert_entity_id_into_tokens`  method correctly inserts the given entity ID into specified positions in a list of tokens.
- **Why**: Ensuring that the entity ID is inserted at the expected positions is crucial for the integrity of the token processing logic.
- **How**: 
  - The  `run_test_insert_entity_id_into_tokens`  method sets up a sample list of tokens and calls the  `insert_entity_id_into_tokens`  method.
  - The  `assertion_test_insert_entity_id_into_tokens`  method then checks that the entity ID is correctly inserted at the first and seventh positions in the first list and the first position in the second list.

#### Test 2:  `process_data_for_word2vec` 
- **What**: This test validates the functionality of the  `process_data_for_word2vec`  method, ensuring it processes a DataFrame correctly.
- **Why**: It is important to confirm that the method can handle and process data effectively, producing expected results.
- **How**:
  - The  `run_test_process_data_for_word2vec`  method creates a DataFrame with entity IDs and filtered data, then calls the  `process_data_for_word2vec`  method.
  - The  `assertion_test_process_data_for_word2vec`  method checks that the resulting DataFrame is not empty, indicating successful processing.

#### Test 3:  `create_dataframe_with_model_vectors` 
- **What**: This test checks that the  `create_dataframe_with_model_vectors`  method can create a DataFrame with model vectors based on a mock model.
- **Why**: Testing this integration is important to ensure that the method interacts correctly with the external model and produces the desired output.
- **How**:
  - The  `run_test_create_dataframe_with_model_vectors`  method creates a mock model using  `MagicMock` , simulating the behavior of a word vector model.
  - It then calls the  `create_dataframe_with_model_vectors`  method with the mock model and a DataFrame.
  - The  `assertion_test_create_dataframe_with_model_vectors`  method verifies that the resulting DataFrame is not empty and checks that the  `NUM_CALLS`  column matches the expected values.

---

This summary should help your team understand the purpose and functionality of the unit tests in the provided module.


few things missing 



I feel like unitests I developed is not robust and not cover all the code logic entirely. I want to review the code throughly and enhance the unittest to make them more robust.   

```
    @tag('insert_entity_id_into_tokens')
    def run_test_insert_entity_id_into_tokens(self):
        tokens_list = [['token1', 'token2', 'token1', 'token2', 'token1', 'token2', 'token1', 'token2'], ['token3', 'token4']]
        self.result = self.processor.insert_entity_id_into_tokens('entity_id', tokens_list)

    def assertion_test_insert_entity_id_into_tokens(self):
        assert self.result[0][0] == 'entity_id'
        assert self.result[0][6] == 'entity_id'
        assert self.result[1][0] == 'entity_id'

    @tag('process_data_for_word2vec') ## start here
    def run_test_process_data_for_word2vec(self):
        df = pd.DataFrame({
            'ENTITY_ID': ['id1', 'id2'],
            'FILT_DATA': ['[["token1", "token2"],["token1", "token2"]]', '[["token3", "token4"]]']
        })
        self.result = self.processor.process_data_for_word2vec(df)

    def assertion_test_process_data_for_word2vec(self):
        self.assertTrue(len(self.result) > 0)

    @tag('create_dataframe_with_model_vectors')
    def run_test_create_dataframe_with_model_vectors(self):
        model = MagicMock()
        model.wv.key_to_index.keys.return_value = ['ID1-E', 'ID2-E']
        model.wv.__getitem__.side_effect = lambda x: [1, 2, 3]
        df = pd.DataFrame({'ENTITY_ID': ['ID1-E', 'ID2-E',  'ID2-E']})
        self.result = self.processor.create_dataframe_with_model_vectors(model, df)
        # print(self.result)

    def assertion_test_create_dataframe_with_model_vectors(self):
        self.assertFalse(self.result.empty)
        assert self.result.NUM_CALLS.tolist() == [1, 2]
```

```
class TokenInjection:
  def __init__(self):
    """
      Initialize the TokenInjection class.
      Args:
        user_config_path (str): Path to the user configuration file.
    """
    self.config = config
    self.dbfs_utility = DBFSUtility()
    self.df_transform_utility = DFTransformUtility()
    self.date_util = DateUtility()
    self.spark_df_utils_obj = SparkDFUtils()
    print("TokenInjection class initialized successfully")

  @staticmethod
  def insert_entity_id_into_tokens(entity_id: str, tokens_list: List[List[str]]) -> List[List[str]]:
    """
      Insert entity ID token into the list of tokens.
      Args:
        entity_id (str): The entity ID to be inserted.
        tokens_list (List[List[str]]): List of token lists.
      Returns:
        List[List[str]]: Updated list of token lists with entity ID inserted.
    """
    try:
      token_list_with_eid = []
      entity_id = entity_id.strip().replace(' ', '_')

      for tokens in tokens_list:
        for i in range(0, len(tokens), 6):
          tokens.insert(i, str(entity_id))
        token_list_with_eid.append(tokens)
      return token_list_with_eid
    except Exception as e:
      print(f"Failed to insert entity ID token: {e}")
      raise

  def process_data_for_word2vec(self, currdf: pd.DataFrame) -> List[List[str]]:
    """
      Process the data to prepare for Word2Vec training.
      Args:
        currdf (pd.DataFrame): The current DataFrame to be processed.
      Returns:
        List[List[str]]: List of token lists ready for Word2Vec training.
    """
    try:
      print("Processing data for Word2Vec training")
      currdf['FILT_DATA'] = currdf['FILT_DATA'].apply(ast.literal_eval)
      currdf['ticker_sents'] = currdf[['ENTITY_ID', 'FILT_DATA']].apply(
          lambda x: self.insert_entity_id_into_tokens(x[0], x[1]), axis=1
      )
      print("Entity IDs token inserted successfully")
      tokens_list = list(itertools.chain.from_iterable(currdf['ticker_sents'].tolist()))
      print("Data processed successfully")
      return tokens_list
    except Exception as e:
      print(f"Failed to process data: {e}")
      raise

  def create_dataframe_with_model_vectors(self, w2v_model: Word2Vec, currdf: pd.DataFrame) -> pd.DataFrame: 
    """
      Create a DataFrame with the model vectors for EIDs.
      Args:
        w2v_model (Word2Vec): Trained Word2Vec model.
        currdf (pd.DataFrame): Current DataFrame with data.
      Returns:
        pd.DataFrame: DataFrame with model vectors for EIDs.
    """
    try:
      print("Creating DataFrame with model vectors for EIDs")
      entity_ids = [
          x for x in w2v_model.wv.key_to_index.keys()
          if x.endswith('-E') and '_' not in x and x.isupper()
      ]
      eid_count_dict = {id: len(currdf[currdf['ENTITY_ID'] == id]) for id in entity_ids}
      self.spark_df_utils_obj.cleanup(currdf)
      end_month = str(datetime.now().month)
      end_year = str(datetime.now().year)
      df = pd.DataFrame({
          'ENTITY_ID': entity_ids,
          'NUM_CALLS': [eid_count_dict[id] for id in entity_ids],
          'EMBED': [w2v_model.wv[x] for x in entity_ids],
          'MODEL_END_DATE': [pd.to_datetime(f"{end_year}-{end_month}-01")] * len(entity_ids),
          'MODEL_START_DATE': [pd.to_datetime(f"{max(2010, int(end_year) - 5)}-{end_month if int(end_year) > 2011 else '01'}-01")] * len(entity_ids)
      })
      print("DataFrame created successfully")
      return df
    except Exception as e:
      print(f"Failed to create DataFrame: {e}")
      raise
```

---------
few updates:

- Yesterday, I discussed the instructor embedding fix with MJ. I implemented the changes that he suggested to the code and tested it today. in today ru  we are getting more counts than usaual. Let’s monitor the results this week. We can schedule a call with Ops next week Monday or later.

- I set up a call with the DevOps team today at 7:30 to discuss the repository upgrade and enabling GitHub Pages features. I received  replies two members of team saying that they are on medical leave. Lets say 
- Regarding topic modeling, I created a branch for a documentation dummy repository and staged all the common NLP package code in it. I'm currently testing the modules. anish 

- The issue with the inference results has been resolved, and I’ve tested the changes for NLI optimization. Yesterday, I shared the updates with Yujing, along with the test results, and she is currently validating them.



--------------

rephrase 
- Topic modelling part I am working on setting up documentation for the package, It almost done it should ready be tommorrow.  
- I am workign on testing part for topic modelling code  
- NLI model finetuning and inference refactoring is complete I will connect with yujing to get it reviewed. Also, I will get detail on the models details she is building.
- I will work on packaging part, dependency management and creting CI/CD pipiline


Requirements for Setting Up New Branch for Limited Release on East1

Hi Naresh,

We are setting up a new branch for the limited release, named quant_lr. Here are the requirements for the new environment in the East1 region:

1. Workspace Mapping: Map this new branch to the acceptance workspace in East1, alongside the quant_stg branch (staging environment).
2. New Schema: Create a new schema for the limited release on Snowflake, including the necessary role configuration. The quant_stg role should be applicable here as well.
3. Blob Container: Set up an additional blob container for storing limited release artifacts.
   
Could you please look into this? Let me know if you need any further details.


----- I have couple of things to discuss 
- on east 1 migration regardig n creation of new branches are we going to create a role to the feature release schema or  role assignmet feature realse schma 
- mlflow i was instaed 


Here’s a rephrased version of your message:

---

Nov 20 -- 17:51 

I have a few topics to discuss:

- **Updates on Topic Modeling**: I have completed the packaging for the topic modeling code and the common NLP code and tested it sucessfully. right now, I will be publishing it on GitHub as a wheel (.whl) binary package, which is lightweight. I am also developing workflows for different topic modeling use cases using refcatored code. Currently, I've put this on hold to focus on integrating the MLflow requirements based on our previous discussions and implementing other necessary changes into common code.

- **Company Similarity Pipelines**: I am working on implementing the changes suggested by Priya for the company similarity pipelines.

**Questions**:

- I have a suggestion regarding MLflow. Instead of logging different base models into the same experiment, why not maintain separate experiments and register the models in the same model registry? This approach would allow us to compare models in the registry that come from different experiments. For instance, if we are comparing different versions of the same base model, like DeBERTa small and large, we can include them in the same experiment. However, when comparing models from different families, such as DeBERTa and LLaMA, keeping them in the same experiment complicates things because the input data formats for NLI differ between DeBERTa and LLaMA, and their training pipelines are also distinct.

- Regarding the East 1 migration and the creation of new branches, will we be creating a role for the feature release schema or for the role assignment feature release schema?

- discuss on tach

#### talking points with Bea
 - I am developing a CI/CD pipeline to automate the deployment from development to production. It can only be tested in the new environment, which I will connect with.
- After testing MLflow, Yujing suggested some enhancements and raised a few issues. I have been working on implementing her suggestions and fixing the issues.
- Today, I connected with Yujing to discuss the workflow steps involved in developing an NLP use case, from fine-tuning on custom labels to backfilling and setting up the daily pipeline.
   


- updates 

- I have completed refacted pipeline notebook modelling usecase compared with origina output that i shared few min back. I am doing similar comparision exverise for nli inferece usecase   
- the mlflow memeory has been resolved innthe updated code  and yujing  
- i am updated the packages with few few additianal funtions that yujing shared and updating  package documentation for the new function that we added recelty in the mlflow utils  and yujing one 
- coming to comsumer topic, yesturday. I spoke to josh, he ask me run new iteration with couple of chages 
  - add netural sentences to the existing finetuning set to match out od sample/ real dataset distribution.
  - try fintuning with lower learning rates 