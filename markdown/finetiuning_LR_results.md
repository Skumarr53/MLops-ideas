Here is a table displaying the percentage of entailment score predictions below various threshold values, ranging from 0.1 to 0.9. These scores are derived from an NLI model trained with different hyperparameters and datasets, specifically NonNeutral_train and TrueRep_train. Based on these figures, I would like to hear your observations and insights, particularly regarding how the parameters and datasets influence the distribution of the scores.


run | topic | <0.1 | <0.2 | <0.3 | <0.4 | <0.5 | <0.6 | <0.7 | <0.8 | <0.9 |  | <1.0 | learning_rate | epoch
                                                        
deberta-v3-large-zeroshot-v2_CT_NonNeutral_train_CT_TrueRep_test_param_set6 | This text is about a weak consumer or reduced consumption. | 0.969 | 0.973 | 0.975 | 0.977 | 0.978 | 0.980 | 0.982 | 0.984 | 0.988 | 0.018 | 1 | 0.0000001 | 5
deberta-v3-large-zeroshot-v2_CT_NonNeutral_train_CT_TrueRep_test_param_set7 | This text is about a weak consumer or reduced consumption. | 0.947 | 0.962 | 0.965 | 0.969 | 0.973 | 0.975 | 0.976 | 0.978 | 0.983 | 0.036 | 1 | 0.000001 | 5
deberta-v3-large-zeroshot-v2_CT_NonNeutral_train_CT_TrueRep_test_param_set8 | This text is about a weak consumer or reduced consumption. | 0.970 | 0.970 | 0.971 | 0.971 | 0.973 | 0.973 | 0.973 | 0.973 | 0.973 | 0.003 | 1 | 0.00002 | 5
deberta-v3-large-zeroshot-v2_CT_NonNeutral_train_CT_TrueRep_test_param_set9 | This text is about a weak consumer or reduced consumption. | 0.972 | 0.973 | 0.974 | 0.974 | 0.974 | 0.975 | 0.975 | 0.975 | 0.975 | 0.003 | 1 | 0.00001 | 5
deberta-v3-large-zeroshot-v2_CT_NonNeutral_train_CT_TrueRep_test_param_set10 | This text is about a weak consumer or reduced consumption. | 0.898 | 0.903 | 0.907 | 0.909 | 0.915 | 0.920 | 0.923 | 0.934 | 0.945 | 0.047 | 1 | 0.0001 | 5
                                            0.021 |  |  | 
                                                        
deberta-v3-large-zeroshot-v2_CT_TrueRep_train_CT_TrueRep_test_param_set6 | This text is about a weak consumer or reduced consumption. | 0.973 | 0.974 | 0.975 | 0.975 | 0.975 | 0.975 | 0.977 | 0.979 | 0.982 | 0.009 | 1 | 0.0000001 | 5
deberta-v3-large-zeroshot-v2_CT_TrueRep_train_CT_TrueRep_test_param_set7 | This text is about a weak consumer or reduced consumption. | 0.975 | 0.976 | 0.976 | 0.976 | 0.976 | 0.976 | 0.976 | 0.976 | 0.976 | 0.001 | 1 | 0.000001 | 5
deberta-v3-large-zeroshot-v2_CT_TrueRep_train_CT_TrueRep_test_param_set8 | This text is about a weak consumer or reduced consumption. | 0.977 | 0.977 | 0.977 | 0.977 | 0.977 | 0.977 | 0.977 | 0.977 | 0.979 | 0.002 | 1 | 0.00002 | 5
deberta-v3-large-zeroshot-v2_CT_TrueRep_train_CT_TrueRep_test_param_set9 | This text is about a weak consumer or reduced consumption. | 0.979 | 0.979 | 0.979 | 0.979 | 0.979 | 0.979 | 0.979 | 0.980 | 0.980 | 0.001 | 1 | 0.00001 | 5
deberta-v3-large-zeroshot-v2_CT_TrueRep_train_CT_TrueRep_test_param_set10 | This text is about a weak consumer or reduced consumption. | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 1 | 0.0001 | 5                                                        
                                                        
                                                        
                                                        
run | topic | <0.1 | <0.2 | <0.3 | <0.4 | <0.5 | <0.6 | <0.7 | <0.8 | <0.9 |  | learning_rate | epoch | 
                                                        
deberta-v3-large-zeroshot-v2_CT_NonNeutral_train_CT_TrueRep_test_param_set6 | This text is about a strong consumer or increased consumption. | 0.928 | 0.961 | 0.971 | 0.973 | 0.976 | 0.979 | 0.979 | 0.980 | 0.986 | 0.057 | 0.0000001 | 5 | 
deberta-v3-large-zeroshot-v2_CT_NonNeutral_train_CT_TrueRep_test_param_set7 | This text is about a strong consumer or increased consumption. | 0.956 | 0.968 | 0.971 | 0.974 | 0.977 | 0.977 | 0.979 | 0.982 | 0.983 | 0.027 | 0.000001 | 5 | 
deberta-v3-large-zeroshot-v2_CT_NonNeutral_train_CT_TrueRep_test_param_set8 | This text is about a strong consumer or increased consumption. | 0.972 | 0.973 | 0.974 | 0.975 | 0.975 | 0.975 | 0.975 | 0.975 | 0.975 | 0.003 | 0.00002 | 5 | 
deberta-v3-large-zeroshot-v2_CT_NonNeutral_train_CT_TrueRep_test_param_set9 | This text is about a strong consumer or increased consumption. | 0.975 | 0.976 | 0.977 | 0.977 | 0.977 | 0.977 | 0.977 | 0.977 | 0.977 | 0.002 | 0.00001 | 5 | 
deberta-v3-large-zeroshot-v2_CT_NonNeutral_train_CT_TrueRep_test_param_set10 | This text is about a strong consumer or increased consumption. | 0.945 | 0.947 | 0.949 | 0.949 | 0.950 | 0.951 | 0.953 | 0.953 | 0.958 | 0.014 | 0.0001 | 5 | 
                                                        
deberta-v3-large-zeroshot-v2_CT_TrueRep_train_CT_TrueRep_test_param_set6 | This text is about a strong consumer or increased consumption. | 0.977 | 0.978 | 0.979 | 0.979 | 0.980 | 0.980 | 0.980 | 0.980 | 0.982 | 0.005 | 0.0000001 | 5 | 
deberta-v3-large-zeroshot-v2_CT_TrueRep_train_CT_TrueRep_test_param_set7 | This text is about a strong consumer or increased consumption. | 0.979 | 0.979 | 0.979 | 0.979 | 0.979 | 0.979 | 0.979 | 0.979 | 0.979 | 0.000 | 0.000001 | 5 | 
deberta-v3-large-zeroshot-v2_CT_TrueRep_train_CT_TrueRep_test_param_set8 | This text is about a strong consumer or increased consumption. | 0.982 | 0.982 | 0.982 | 0.982 | 0.982 | 0.982 | 0.982 | 0.982 | 0.982 | 0.001 | 0.00002 | 5 | 
deberta-v3-large-zeroshot-v2_CT_TrueRep_train_CT_TrueRep_test_param_set9 | This text is about a strong consumer or increased consumption. | 0.980 | 0.980 | 0.980 | 0.980 | 0.980 | 0.980 | 0.980 | 0.980 | 0.980 | 0.000 | 0.00001 | 5 | 
deberta-v3-large-zeroshot-v2_CT_TrueRep_train_CT_TrueRep_test_param_set10 | This text is about a strong consumer or increased consumption. | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.0001 | 5 | 
