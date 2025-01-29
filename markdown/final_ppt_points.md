### **Slide Content**

* * *

#### **Slide 1: Code Refactoring**

*   **Structured & Clean Code**
    *   Organize existing pipelines for clarity
*   **Logical Understanding**
    *   Deep comprehension of pipeline logic
*   **Best Practices**
    *   Consistent variable definitions
    *   Simplify complex logic into manageable steps and functions
*   **Modularity & Reusability**
    *   Enhance maintainability through modular design
*   **Packaging Functionalities**
    *   Collate functionalities into reusable and distributable packages

* * *

#### **Slide 2: Emphasis on Refactoring**

*   **Configuration Management**
    *   Maintain config files for dynamic run parameters
*   **Efficient Execution**
    *   Enable quick, single executions across multiple stages
*   **Error Handling**
    *   Streamline handling of fatal and other execution errors

* * *

#### **Slide 3: Data Testing**

*   **Testing Framework Design**
    *   Validate pipeline outputs effectively
*   **Diverse Data Types**
    *   Handle numerics, text, and embeddings
*   **Continuous Validation**
    *   Ensure logic integrity post-refactoring and during PySpark conversion
*   **Scalability Challenges**
    *   Address testing at scale to cover corner cases
*   **Case Study**
    *   Recent pipeline success in output accuracy

* * *

#### **Slide 4: Model Deployment**

*   **Deployment Context**
    *   Rationale behind the need for improved deployment
*   **Approach for New Use Cases**
    *   Step-by-step deployment strategy
*   **NLI Model Overview**
    *   Automated deployment of the Natural Language Inference model
*   **Experimentation Insights**
    *   Parameter usage and customization challenges
*   **Productionization Ease**
    *   Streamlined process for moving models to production

* * *

#### **Slide 5: Topic Modelling Packages**

*   **Project Overview**
    *   Developing a user-friendly topic modelling package
*   **Publication Justification**
    *   Enhance usability and accessibility
*   **User Experience Focus**
    *   Web-based interactive documentation
*   **Package Maintenance**
    *   Inspired by popular libraries like Pandas and NumPy
*   **Resource Link**
    *   [NLP Centralized Package](https://pages.github.voya.net/Voya/data-science-nlp-centralized-nlp-package/_build/html/index.html)

* * *

#### **Slide 6: Optimizing Existing Pipelines**

*   **Identifying Bottlenecks**
    *   Analyze and address slow processing stages
*   **Performance Improvement**
    *   Rewrite logic and transition to PySpark
*   **Transition from Pandas to PySpark**
    *   Enhance production readiness and scalability
*   **Success Stories**
    *   **Consumer Topic Historical Backfill**
        *   Reduced processing time from 36 hrs to ~2 hrs using PySpark
    *   **Job Link Up**
        *   Replaced regex and pandas-driven features with PySpark for efficiency

* * *

### **Detailed Comprehensive Write-Up**

To start, most of my responsibilities focus on simplifying existing processes, optimizing operations through automation, and identifying and eliminating bottlenecks. In this I will talk about the old  practices and what improveements being made in the current for operations 


* * *

#### **Slide 1: Code Refactoring**

previous:
When I joined Voya, I began with code refactoring. The code in our pipelines was highly unorganized, making it difficult to manage, understand the flow, and debug. it simply does the job there there no code structure. We bagan to refactor the code. I will breifly expalin steps involved.



**Logical Understanding** Firstly,  need to review code to get understanding of the underlying logic of the pipeline. This ensures that any refactoring efforts preserve the intended functionality and improve upon the existing design without introducing errors.

then organizing pipelines to ensure they are well-structured and clean. This involves removing redundancies, optimizing code flow, and enhancing readability, making it easier for other team members to understand.

**Best Practices** the refactored code should Adhered to best software development practices is crucial. This includes consistent variable naming conventions, defining clear and meaningful variables, and breaking down complex logic into smaller, manageable functions. Such practices enhance code maintainability and reduce the likelihood of bugs.

**Modularity & Reusability** By designing pipelines in a modular fashion, components can be reused across different projects. This also simplifies maintenance. this step we wanted to take it further. In the some pipelines which share common logic flow we decided that the functionalities collated into comprehensive packages, promoting reusability and ease of distribution to all stakeholders. I will talk abiout this in detail, in coming slide.

* * *

#### **Slide 2: Emphasis on Refactoring**
cofig managemnt. Although this is part of the refactoring process, I want to highlight this particular aspect. Previously, the pipeline run parameters were hardcoded with, which hindered code reusability. Imagine trying to create a new pipeline or a new iteration using existing pipeline code. With hardcoded parameters, you would need to manually identify and change them, which is time-consuming and cumbersome, especially when dealing with multiple notebooks in the pipeline. More importantly, if any parameter is mistakenly left unchanged, it can lead to catsphy. for example forgot to change table name . In the last six months, I've observed issues in at least two pipelines due to such misconfigurations. given these limitations. 


suggested to manage run parameters via config managemnt. 

**Configuration Management** Maintaining separate configuration files allows for dynamic adjustment of run parameters without altering the core codebase. This flexibility facilitates quicker iterations and easier management of different execution environments.

**Efficient Execution** This design can support single executions that can handle multiple stages simultaneously reducing execution time and simplifies the workflow, enabling more efficient processing of data pipelines.

**Error Handling** Main Enhancing error handling mechanisms ensures that fatal and other execution errors are managed gracefully. By improving the robustness of pipelines, we minimize downtime and maintain the integrity of data processing operations.

* * *

#### **Slide 3: Data Testing**


After refactoring existing code for better maintainability or optimizing certain pipelines by transitioning standard Python code to Spark for more efficient operations, it's crucial to ensure thatany code changes or refactoring efforts do not disrupt existing logic


For this A robust testing framework is essential for validating the outputs of original and refactopred pipelines. This framework systematically checks the accuracy and reliability of the results produced at each stage of the pipeline.

 Our testing approach should accommodate various data types, not just simnple data types such as numerical data, string. . Given that our team frequently works with textual data and embeddings, the framework is dsesigned to handle the intricacies of these data forms effectively. I tell you prablem with embeddings they cannot be compared directly many times due precision, we need empley cosine similarity, this is same meticr which shrimukh had talked about during GenAI presentation this computes distance between embedding. ideally, if they are identical thy shopuld overlap but due percision it may not happen that we see if they match 99 percent. and in one of our piline what inovlves training process even this not possible as training rtandomness everytime embedding generated aere different. in that case we have to employ diifernt stargey such to looking into relative distance between embbeding entities for eample if comp A and B are close in original pipleine created embedding they shoudl alos clsoe in refcatored pipline embedding naking sure  although embedding valkues iteslf different    


**Scalability Challenges** One of the primary challenges is scaling tests to handle thousands of records, ensuring that corner cases are not overlooked. Comprehensive testing across different time periods and data scenarios helps in identifying and addressing potential issues before they impact production.

**Case Study** i would like to quote a recent example. we trastioned one pipeline duration val;idation we sampled data vlaidate from narrow timeperiod specifcally form a particaular month it and validn results looked good. but in production, there was mismatch in the expected and pysprk trainistioned code output. this use case perfectly demosatrtes impoetance validation at sacl 

* * *

#### **Slide 4: Model Deployment**

- Whenevr a new use case comes in, befor egetting into old approch we followed. I would like talk about the use case we are solving first. For eaxmple, I am taking Glassdoor review usecxase wherre we need to identify topic review is associated with. topics could salary, work culture, management and many more. We use natural language inference model it an language model which predicts whether text or review in this case is associaterd with the topic. expand...
- if rview discuss salry then respective entailment score, it assocation score you can say,will be high 
- in the ivestment imangement area, use case could be whether magamnets discussion section of call trascpt talks about incresed consumption and reduced consumption in the market which are kind of macro economic indicators. we want to development model that can extarct and idenfiy text that deicsuse either of the topics. this will used to gain understaing of how company perperspective.
- and coming ealrier approch fo sobving use case. it was hifghly manual and not efficient.
  
- talk about training parameters, tarinsets, model versions, and 
- running iterations manual and each sotre artifactes in blob separately. once all thge iteration are run, model evalution on evalution set and comparing diff model was again was different proces s involing manula loading all models and generate metrics at one place managing this was very difficult 

- The current frame which we built addresses all these issues all the steps are taken care with single excution.   

**Deployment Context** The need for an improved model deployment strategy arose from the challenges and limitations of previous experimental approaches. Ensuring seamless deployment is critical for integrating new models into production environments efficiently.

**Approach for New Use Cases** For every new use case, we follow a structured deployment strategy. This involves defining requirements, selecting appropriate models, conducting experiments, and automating the deployment process to ensure consistency and reliability.

**NLI Model Overview** Our Natural Language Inference (NLI) model serves as a foundational component. Automating its deployment ensures that it can be reliably integrated into various applications, enhancing our capability to process and understand natural language data.

**Experimentation Insights** Experimentation involves using specific parameters to fine-tune model performance. Customizing transformer models poses challenges due to their complexity. By managing original code and ensuring it remains manageable, we can effectively tailor models to meet specific requirements.

**Productionization Ease** The refactored deployment process significantly simplifies moving models into production. Enhanced logging and artifact management provide in-depth insights into model performance, facilitating easier troubleshooting and optimization.

**Screenshot Reference** Including a screenshot from the Azure Databricks ML experiments dashboard illustrates the parameters used in experimentation and demonstrates the practical application of our deployment strategy. [View Experimentation](https://adb-2762743938046900.0.azuredatabricks.net/ml/experiments/4009146671706400?o=2762743938046900)

* * *

#### **Slide 5: Topic Modelling Packages**

**Project Overview** Developing a dedicated package for topic modelling aims to streamline and simplify the process for users. This package encapsulates complex topic modelling functionalities into an easy-to-use interface.

**Publication Justification** Publishing the package addresses the need for user-friendly tools in topic modelling. By making advanced techniques accessible, we empower data scientists and analysts to perform sophisticated analyses without extensive technical overhead.

**User Experience Focus** To enhance user experience, the package includes web-based interactive documentation. This approach replaces lengthy traditional documentation, offering a more intuitive and engaging way for users to learn and utilize the package.

**Package Maintenance** Inspired by widely-adopted libraries like Pandas and NumPy, our topic modelling package emphasizes ease of use and reliability. Regular maintenance ensures that the package remains up-to-date with the latest advancements and user needs.

**Resource Link** For an overview and detailed documentation, refer to our centralized NLP package portal: [NLP Centralized Package](https://pages.github.voya.net/Voya/data-science-nlp-centralized-nlp-package/_build/html/index.html). This resource provides comprehensive guidance on using the package effectively.

* * *

#### **Slide 6: Optimizing Existing Pipelines**

**Identifying Bottlenecks** Our team is dedicated to identifying and addressing bottlenecks within existing data pipelines. By analyzing processing stages that consume excessive time, we can target specific areas for optimization.

**Performance Improvement** Improving pipeline performance involves rewriting inefficient logic and transitioning from Pandas to PySpark. PySpark offers superior scalability and performance for large-scale data processing, making it ideal for production environments.

**Transition from Pandas to PySpark** While Pandas is excellent for experimentation and small-scale data manipulation, it often falls short in production due to scalability limitations. Transitioning to PySpark ensures that pipelines can handle larger datasets efficiently and reliably.

**Success Stories**

*   **Consumer Topic Historical Backfill**
    
    *   **Challenge:** The original pipeline required 36 hours to process historical data.
    *   **Solution:** Rewritten entirely in PySpark, processing time was reduced to approximately 2 hours.
    *   **Impact:** Significant improvement in processing speed enabled timely data availability and enhanced operational efficiency.
*   **Job Link Up**
    
    *   **Challenge:** Feature extraction was dependent on regex and Pandas, causing bottlenecks.
    *   **Solution:** Transformed the feature extraction process to PySpark, eliminating the dependency on Pandas.
    *   **Impact:** Streamlined workflow with faster processing times and increased reliability in feature generation.

**Conclusion** Through strategic refactoring and optimization, our pipelines have achieved substantial performance gains. These enhancements not only reduce processing times but also ensure scalability and maintainability, positioning our data infrastructure for future growth and success.

* * *


rephrase crisp and for clarity

⁠Even we too surpreized at first, earlier pandas code was not utilizing  parallezation cxaptility cluster trasitioning to pyspark gave boost 




Hi team, Here is the link to the documentation and the PowerPoint presentation.

PPT:
https://voya0-my.sharepoint.com/:p:/g/personal/santhosh_kumar3_voya_com/EVDq5TTdL9lGo4pO7vRsCukBWBU1VIhFDAAOFZD5Y3feew?e=9zLmh8

Package Documentation:
https://pages.github.voya.net/Voya/data-science-nlp-centralized-nlp-package/_build/html/index.html
