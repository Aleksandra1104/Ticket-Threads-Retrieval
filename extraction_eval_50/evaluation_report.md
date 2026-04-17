# Extraction Evaluation Report

## Dataset
- Total tickets: 50
- Eligible tickets: 43
- Extracted tickets: 42
- Evaluation scope: resolved_only

## Extraction Metrics
- Precision: 1.0
- Recall: 0.9767
- F1: 0.9882
- True positives: 42
- False positives: 0
- False negatives: 1
- True negatives: 7

## Category Metrics
- Fine-grained evaluated predictions: 42
- Fine-grained correct predictions: 39
- Fine-grained accuracy: 0.9286
- Family-level evaluated predictions: 42
- Family-level correct predictions: 40
- Family-level accuracy: 0.9524

## Text Similarity Metrics
### Issue Text (predicted query vs best-matching ground-truth title)
- Average token F1: 0.3562
- Average sequence similarity: 0.4372
- Average embedding similarity: 0.6239

### Resolution Text (predicted positive vs ground-truth resolution summary)
- Average token F1: 0.2629
- Average sequence similarity: 0.412
- Average embedding similarity: 0.6737

## Confidence
- Average confidence: 0.9
- Count: 42

## Skipped Ticket Reasons
- ground_truth resolution_state is not resolved: unresolved: 6
- ground_truth resolution_state is not resolved: partial: 1
- model marked ticket unusable: 1

## Category Distributions
### Ground Truth Categories
- account_locked: 2
- api_failure: 3
- application_crash: 1
- browser_issue: 2
- data_recovery: 3
- disk_space_full: 2
- email_issue: 3
- internet_access: 2
- mfa_issue: 6
- onboarding_offboarding: 4
- password_reset: 2
- peripheral_issue: 1
- phishing_report: 2
- printer_issue: 3
- server_unavailable: 1
- software_install: 2
- vpn_issue: 1
- wifi_connectivity: 1
- workstation_failure: 1

### Predicted Categories
- account_locked: 2
- api_failure: 3
- application_crash: 1
- browser_issue: 2
- data_recovery: 2
- disk_space_full: 2
- email_issue: 3
- internet_access: 2
- mfa_issue: 6
- onboarding_offboarding: 4
- password_reset: 2
- peripheral_issue: 1
- permission_issue: 2
- phishing_report: 2
- printer_issue: 3
- server_unavailable: 1
- shared_drive_issue: 1
- vpn_issue: 1
- wifi_connectivity: 1
- workstation_failure: 1

### Ground Truth Families
- Backend: 4
- Hardware: 5
- IAM: 14
- Networking: 4
- Security: 2
- Software: 8
- Storage: 5

### Predicted Families
- Backend: 4
- Hardware: 5
- IAM: 16
- Networking: 4
- Security: 2
- Software: 6
- Storage: 5

## Fine-Grained Category Confusion Matrix
- account_locked -> account_locked:2
- api_failure -> api_failure:3
- application_crash -> application_crash:1
- browser_issue -> browser_issue:2
- data_recovery -> data_recovery:2, shared_drive_issue:1
- disk_space_full -> disk_space_full:2
- email_issue -> email_issue:3
- internet_access -> internet_access:2
- mfa_issue -> mfa_issue:6
- onboarding_offboarding -> onboarding_offboarding:4
- password_reset -> password_reset:2
- peripheral_issue -> peripheral_issue:1
- phishing_report -> phishing_report:2
- printer_issue -> printer_issue:3
- server_unavailable -> server_unavailable:1
- software_install -> permission_issue:2
- vpn_issue -> vpn_issue:1
- wifi_connectivity -> wifi_connectivity:1
- workstation_failure -> workstation_failure:1

## Family-Level Confusion Matrix
- Backend -> Backend:4
- Hardware -> Hardware:5
- IAM -> IAM:14
- Networking -> Networking:4
- Security -> Security:2
- Software -> IAM:2, Software:6
- Storage -> Storage:5

## Example Fine-Grained Category Errors
- TDX-10001: gt=software_install, pred=permission_issue, gt_family=Software, pred_family=IAM, conf=0.9
- TDX-10008: gt=data_recovery, pred=shared_drive_issue, gt_family=Storage, pred_family=Storage, conf=0.85
- TDX-10040: gt=software_install, pred=permission_issue, gt_family=Software, pred_family=IAM, conf=0.9

## Example Family Errors
- TDX-10001: gt_family=Software, pred_family=IAM, gt=software_install, pred=permission_issue
- TDX-10040: gt_family=Software, pred_family=IAM, gt=software_install, pred=permission_issue

## Worst Issue Matches
- TDX-10026: issue_emb=0.0307, pred='New hire missing email, VPN, and ERP app access.', best_gt='Offboarding request follow-up'
- TDX-10044: issue_emb=0.3214, pred='VPN connection fails after authentication', best_gt='Remote access issue'
- TDX-10046: issue_emb=0.3455, pred='API response format changed, breaking downstream parser.', best_gt='Service-to-service API error'
- TDX-10035: issue_emb=0.347, pred='onboarding request incomplete', best_gt='New hire access missing'
- TDX-10033: issue_emb=0.4102, pred='Keyboard and mouse keep disconnecting from the dock.', best_gt='Peripheral issue'

## Worst Resolution Matches
- TDX-10038: res_emb=0.3759, pred='Cause: Sender impersonation from outside. Fix: Quarantined message and blocked similar patterns.', gt='The reported email was phishing and was quarantined. Because the user did not click or submit credentials, the account did not require recovery steps.'
- TDX-10020: res_emb=0.3874, pred='Cause: Cached credentials issue. Fix: Clear cached credentials and sign in again.', gt='The issue came from an outdated Outlook credential cache. After clearing cached credentials and re-adding the account, email access returned.'
- TDX-10027: res_emb=0.3874, pred='Cause: Cached credentials issue. Fix: Clear cached credentials and sign in again.', gt='The issue came from an outdated Outlook credential cache. After clearing cached credentials and re-adding the account, email access returned.'
- TDX-10008: res_emb=0.4731, pred='Cause: Backup snapshots not used. Fix: Recovered copy from backup and placed back in shared folder.', gt='The requested data was recoverable from backup or snapshot history. Restoring the prior version returned the missing file.'
- TDX-10014: res_emb=0.483, pred='Cause: File deleted. Fix: Recovered copy from backup and placed back in shared folder.', gt='A previous file version was available in backup retention. Recovering that copy restored the needed data. The user expressed urgency during the conversation.'
