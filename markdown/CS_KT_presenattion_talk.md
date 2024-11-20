**Title Slide**:

Good day everyone, and welcome! Today, I’ll be presenting the KT session on our project Company Similarity Analysis. This project has been developed with a focus on quantitatively analyzing companies through innovative NLP methodologies. The goal of this session is to ensure a smooth handover of this project to the operations team, enabling them to manage and run workflows effectively going forward. Let's begin without further ado. Feel free to ask any questions as we go.

## Slide 2 - Embeddings for Company Representation**:

Let's start by talking about what we mean by "embeddings." Essentially, embeddings are a way of representing unstructured data, such as documents, into numerical formats that we can work with quantitatively. Here, we have applied an embedding algorithm to company documents—these could be earnings call transcripts, annual reports, or even press releases. By converting these documents into a set of numbers, or vectors, we capture complex relationships and similarities that may exist between different firms.

The idea here is that companies with similar embeddings are closer to each other in this numerical space, meaning they are similar in characteristics like markets, products, or technologies. This helps us efficiently analyze and compare companies even when we have large sets of unstructured data.

## Slide 3 - Visualizing Company Similarity**:

Next, let's see how these embeddings translate into meaningful visual representations. We built a universe of embeddings for over 9,000 companies, using their earnings call transcripts. On the left side, you can see a visualization—essentially a map—showing these firms clustered based on their similarity. Clusters for sectors like *Minerals & Geo-exploration*, *Telecommunications*, and *Health & Biosciences* have emerged organically, showcasing the effectiveness of these embeddings in capturing true similarities among companies.

On the right side, we have some practical examples. For instance, the three closest firms to Amazon are Walmart, Wayfair, and Alphabet Inc. This indicates that our model is capturing sensible relationships between companies.

## Slide 4 - Methodology: Two Embedding Types**:

For this project, we used two complementary embedding methodologies. Let me walk you through each:

- **Token Insertion (TI)**: This approach provides a slower-moving signal by using data from the past five years of earnings calls. It was developed entirely in-house and is run quarterly. Since it leverages historical data, it helps capture longer-term relationships.

- **Instructor LM**: On the other hand, Instructor LM is based on a state-of-the-art, open-source language model that is pretrained and run daily. It focuses on providing a more immediate signal by using the latest earnings call transcript.

These two methods complement each other—one focusing on a more stable historical perspective and the other on real-time changes.

## Slide 5 - Data Flow Chart**:

This slide illustrates the flow of data for the Instructor Embedding and Token Injection workflows, which are the primary focus of this presentation. Both workflows draw input from the *Call Transcript workflow*.

The *Call Transcript workflow* extracts and analyzes text from the transcripts, concentrating on the management discussions and Q&A sections. The extracted text is stored in the *CTS_FUND_COMBINED_SCORES_H* table.

The *Instructor Embedding workflow* directly sources data from this table. This workflow processes the earnings call data to generate embeddings that represent the latest company information.

For the *Token Injection workflow*, there is an intermediate step that handles text preprocessing, which involves tokenization and lemmatization of the text. This step is crucial because it cleans and standardizes the text, removing unnecessary elements like stop words, and reducing words to their root forms, which makes the subsequent embedding process more accurate and efficient. The preprocessed text is stored in *CT_TL_STG_1*.

This preprocessed data is then utilized by the *Token Injection workflow* to generate company embeddings that capture long-term relationships, which are stored in the corresponding table for further use.

On this slide, you can see two different shapes—the boxes represent workflows, which are distinct processes or scripts we execute, while the ellipses represent tables where data is stored at various stages.

## Slide 6 - Instructor Embedding Pipeline (Daily)**:

Now let’s zoom in on the Instructor Embedding Pipeline, which runs daily. First, we start by sampling the latest daily transcripts from the earnings call data. We use a pre-trained model to process these transcripts into smaller chunks, keeping each chunk to about 200 words. This makes it easier for our model to handle and generate embeddings efficiently.

Once chunked, the embeddings are generated for each part, and we average these embeddings to represent the company as a whole for that day. This ensures we have an up-to-date numerical representation for each firm, which gets stored for further analysis.

## Slide 7 - Token Insertion Embedding Pipeline (Quarterly)**:

Moving on to the Token Insertion Embedding Pipeline—this is a quarterly process. Here, we work with the most recent five years of transcripts. We first split these transcripts into sentences, tokenize, remove stop words, and lemmatize the text.

We then insert unique company identifiers into each sentence at regular intervals, which allows us to understand relationships across firms better when we run our Word2Vec model. This insertion of company tokens effectively allows the model to "learn" about each firm’s unique context within the broader industry. We then generate embeddings that represent these firms, and a similarity matrix is created to analyze firm-to-firm relationships.

## Slide 8 - Token and Lemmatization Workflow Detail**:

Here are the details for the *Token and Lemmatization workflow*. This is a monthly process, running on the first day of every month at 6:00 AM Eastern Time. The purpose is to preprocess the transcripts by tokenizing and lemmatizing, making sure the data is ready for the token insertion and embedding processes.

## Slide 9 - Instructor Embedding Workflow Detail**:

The *Instructor Embedding Workflow* runs daily at 4:30 AM Eastern Time. The goal here is to maintain a real-time understanding of the firm's similarity landscape by processing the most recent earnings calls. The workflow is available in Databricks, and the main script link is provided for reference.

## Slide 10 - Token Injection Workflow Detail**:

Finally, we have the *Token Injection workflow*, which runs quarterly. This workflow uses the token insertion methodology we discussed earlier to maintain an updated similarity matrix over longer-term trends. It runs at 5:00 AM on the first day of the quarter, capturing a broader historical context for our embeddings.

Thank you all for your attention. With these pipelines and workflows established, we have a solid, reliable mechanism for analyzing company similarity both from a historical and a real-time perspective. I am now happy to take any questions or discuss how we can integrate these processes into your operational workflows moving forward.

