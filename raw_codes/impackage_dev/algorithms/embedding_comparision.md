
``` python
from pyspark.sql import functions as F
from pyspark.ml.linalg import VectorUDT
from pyspark.sql.types import DoubleType

def compare_embeddings(df1, df2, id_col='ENTITY_ID', emb_col='EMBEDDING'):
    # Rename embedding columns
    df1 = df1.select(id_col, F.col(emb_col).alias('embedding1'))
    df2 = df2.select(id_col, F.col(emb_col).alias('embedding2')) #
    
    # Join DataFrames on ENTITY_ID
    joined = df1.join(df2, on=id_col)
    
    # UDF to compute cosine similarity
    def cosine_similarity(v1, v2):
        dot = float(v1.dot(v2))
        norm1 = float(v1.norm(2))
        norm2 = float(v2.norm(2))
        return dot / (norm1 * norm2) if norm1 and norm2 else None
    
    cosine_udf = F.udf(cosine_similarity, DoubleType())
    
    # Calculate cosine similarity
    compared = joined.withColumn('cosine_similarity', cosine_udf('embedding1', 'embedding2'))
    
    # Show summary statistics
    compared.select('cosine_similarity').describe().show()
    
    return compared

# Usage
# compared_df = compare_embeddings(df1, df2)

```


Yes, cosine similarity between embeddings of the same entity from different runs should be high even if the embedding vectors themselves differ. Here's why:

1.  **Directional Consistency**:
    
    *   **Cosine Similarity** measures the angle between two vectors, focusing on their direction rather than magnitude. If embeddings are trained under the same conditions, their directions should capture similar semantic relationships, resulting in high cosine similarity.
2.  **Stochastic Variability**:
    
    *   **Random Initialization and Sampling**: Word2Vec training involves random initialization and stochastic processes (like negative sampling). These introduce variability in the actual embedding values but typically preserve the overall structure and relationships.
3.  **Expected Similarity**:
    
    *   **High Similarity Scores**: While exact values will differ, you should observe high cosine similarity scores (e.g., above 0.8 or 0.9) for the same entities across runs.
    *   **Consistency Across Entities**: Most entities should exhibit high similarity, confirming that differences are due to randomness rather than coding errors.
4.  **Practical Verification**:
    
    *   **Statistical Summary**: Use the provided function to compute and review summary statistics of cosine similarities. Consistently high averages with low variance support the embeddings' reliability.

**Example Interpretation**:
---------------------------

*   **Average Cosine Similarity**: An average close to 1 indicates strong directional alignment.
*   **Low Variance**: Minimal fluctuation across entities suggests stable embedding behavior.

**Conclusion**: High cosine similarity confirms that embeddings maintain their semantic integrity across different training runs, validating that differences in embedding values are due to the inherent randomness of the training process rather than any implementation issues.