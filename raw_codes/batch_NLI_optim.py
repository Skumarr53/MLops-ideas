# Define the schema for the inference results
inference_schema = ArrayType(StructType([
    StructField("label", StringType(), False),
    StructField("score", FloatType(), False)
]))


@pandas_udf(inference_schema, PandasUDFType.SCALAR_ITER)
def inference_udf(iterator):
    global nli_pipeline
    if nli_pipeline is None:
        nli_pipeline = initialize_nli_pipeline(enable_quantization=ENABLE_QUANTIZATION)
    
    for batch_num, batch in enumerate(iterator, start=1):
        logger.info(f"Processing MD inference batch {batch_num} with {len(batch)} rows.")
        try:
            # Flatten the list of text pairs in the batch
            flat_text_pairs = [pair for sublist in batch for pair in sublist]
            logger.debug(f"Batch {batch_num}: Total text pairs to infer: {len(flat_text_pairs)}")
            
            # Perform inference in batch
            results = nli_pipeline(
                flat_text_pairs,
                padding=True,
                top_k=None,
                batch_size=BATCH_SIZE,  # Adjusted batch size
                truncation=True,
                max_length=512
            )
            logger.debug(f"Batch {batch_num}: Inference completed with {len(results)} results.")

            
            # Split results back to original rows
            split_results = []
            idx = 0
            for pairs in batch:
                if len(pairs):
                    split_results.append(results[idx:idx+len(pairs)])
                    idx += len(pairs)
                else:
                    split_results.append([])
            
            yield pd.Series(split_results)
        except Exception as e:
            logger.error(f"Error in MD inference batch {batch_num}: {e}")
            # Yield empty results for this batch to continue processing
            yield pd.Series([[] for _ in batch])