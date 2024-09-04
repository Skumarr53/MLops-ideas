I am suggesting following approch for deploying code changes to production 

- development is done in local system in separte feature branch and tested 
- then branch pushed to remote branch which triggeres unittesting and report generated 
- reviewer review report then merges feature branch with main branch which triggers cicd thta deploys this in stging branch and then integration testing.
- report integration test report is generated and gets submitted to manager for approval
- post approving manager or developer deploys to prod by click action 
- CI pipeline is triggered for every feature branch push, not just MR.
- feature flags to control relase of new features.
- Implement an automated approval workflow within your CI/CD tool (e.g., GitLab, Jenkins, GitHub Actions) that notifies the manager or designated approvers when integration tests pass. Consider using a dashboard to visualize test results, code coverage, and deployment status.

Carefully review the approch and refine the approch in there is gap or imporved further to be aligned with best practices