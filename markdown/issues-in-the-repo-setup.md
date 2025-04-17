

- quant to prod movement should not happen. 
- Dont have permission to clone git repos in all env
- pcant pull changes 
- dont have permission to pull changes. should be grated for all team members.
- changes can be pushed to only quant branch.


good
- require one aprrover other than authour


Bea, I will connect with Andrei separately to discuss on the branching rules yet to implement and rules that do not align with expectations. 



Points discussed with Adrei:
- Not all the members have permission to clone/create repos in the quant, quant_stg and qunat_lr on databricks. I should check this with Arun.
- No matter where we make changes either in prod or non prod  branches but changes should be allowed to be pushed to only quant branch. Andrie confirmed this is already oin place and This will be tested when 1 point resolved.
- Merg rquest creating should be possible as per defined below direction. if it si not adhered creating should not be allowed.
- Authour should not be allowed to merge require and atleat one external approver. even though authour can approve changes he cannout merge change without another approver.


Meeting notes with Andrei:
 
- Not all team members currently have permission to clone or create repositories in the quant, quant_stg, and quant_lr workspaces on Databricks. This needs to be checked with Arun.
 
- Regardless of whether changes are made in production or non-production branches, they must only be allowed to  pushed to the quant branch. Andrei confirmed that this policy is already in place and will be tested once the above permission issue is resolved.

- Merge requests should only be created following the defined branch flow as per standard guidelines. Any deviation from this flow should result in the merge request being blocked. Andrei is currently working on enforcing this rule.
 QUANT > QUANT_STG, QUANT_STG > QUANT_LR, QUANT_STG or QUANT_LR > QUANT_LIVE

- The author of a merge request must not be allowed to merge their own changes. At least one external approver is required. Now, Although the author can approve the their changes, merging is only be permitted with an additional approver. This has been tested

Mlfow current setup shows experiment as name suggests here we experiment with different combination of model parameres and datsets creted by startergy. All the modles form each iteration are logged in the mlflow. for consumer topic use we have 3 such iterations coming from this will eanable us to compare models and select the best one based on the defined metirc precision. this is the candidate with higeht precision so this will be move dto model registry and we promote to staging. this model will be used sample of historical data for validation by end user/ consumer and results will be shared with them. End user post validtes and make sure results are good and expected then they will give approval. post which we will promote to production.




Hi Bea, Tommorrow I logging in and out early due to soem personal work in the evening. I would not be able to attend topic modelling call. I will share the update separately. 




I am moving all my databricks work to new environment. I am downloading the notebooks locally and then upload them in the new environment maually however I am worried about mlflow model and artifacts. I am affraid to download and upload manually as I suspect that could corrupt the models. these models asre transfomrer based models resitered in model registry. is there way to migrate these in safe way. porvide me complete plan guide to achieve this.