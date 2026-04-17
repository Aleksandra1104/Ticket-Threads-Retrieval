# Extraction Evaluation Report

## Dataset
- Total tickets: 200
- Eligible tickets: 144
- Extracted tickets: 130
- Evaluation scope: resolved_only

## Extraction Metrics
- Precision: 1.0
- Recall: 0.9028
- F1: 0.9489
- True positives: 130
- False positives: 0
- False negatives: 14
- True negatives: 56

## Category Metrics
- Fine-grained evaluated predictions: 130
- Fine-grained correct predictions: 47
- Fine-grained accuracy: 0.3615
- Family-level evaluated predictions: 130
- Family-level correct predictions: 64
- Family-level accuracy: 0.4923

## Text Similarity Metrics
### Issue Text (predicted query vs best-matching ground-truth title)
- Average token F1: 0.3693
- Average sequence similarity: 0.4687
- Average embedding similarity: 0.6274

### Resolution Text (predicted positive vs ground-truth resolution summary)
- Average token F1: 0.2467
- Average sequence similarity: 0.3691
- Average embedding similarity: 0.6731

## Confidence
- Average confidence: 0.9146
- Count: 130

## Skipped Ticket Reasons
- ground_truth resolution_state is not resolved: partial: 31
- ground_truth resolution_state is not resolved: unresolved: 25
- invalid issue_category: browser_issue: 3
- invalid issue_category: database_issue: 2
- invalid issue_category: driver_issue: 2
- invalid issue_category: antivirus_issue: 1
- invalid issue_category: data_recovery: 1
- invalid issue_category: disk_space_full: 1
- invalid issue_category: encryption_issue: 1
- invalid issue_category: malware_issue: 1
- invalid issue_category: onboarding_issue: 1
- model marked ticket unusable: 1

## Category Distributions
### Ground Truth Categories
- account_locked: 5
- api_failure: 5
- application_crash: 6
- browser_issue: 2
- data_recovery: 5
- database_connection: 3
- disk_space_full: 4
- email_issue: 5
- encryption_issue: 3
- internet_access: 4
- malware_infection: 1
- mfa_issue: 10
- mobile_device_issue: 3
- onboarding_offboarding: 6
- password_reset: 7
- peripheral_issue: 4
- permission_issue: 4
- phishing_report: 8
- printer_issue: 12
- server_unavailable: 2
- shared_drive_issue: 2
- software_install: 6
- voip_telephony: 7
- vpn_issue: 6
- wifi_connectivity: 6
- workstation_failure: 4

### Predicted Categories
- account_locked: 20
- email_issue: 5
- hardware_issue: 10
- network_issue: 29
- other: 6
- password_reset: 9
- permission_issue: 13
- phishing_report: 9
- printer_issue: 12
- software_access: 11
- vpn_issue: 6

### Ground Truth Families
- Backend: 10
- Hardware: 23
- IAM: 32
- Networking: 23
- Security: 12
- Software: 19
- Storage: 11

### Predicted Families
- Hardware: 12
- IAM: 42
- Networking: 6
- Other: 6
- Security: 9
- Software: 5
- Unknown: 50

## Fine-Grained Category Confusion Matrix
- account_locked -> account_locked:5
- api_failure -> network_issue:3, software_access:2
- application_crash -> software_access:6
- browser_issue -> network_issue:1, other:1
- data_recovery -> other:5
- database_connection -> password_reset:1, permission_issue:2
- disk_space_full -> network_issue:4
- email_issue -> email_issue:5
- encryption_issue -> hardware_issue:2, software_access:1
- internet_access -> network_issue:4
- malware_infection -> phishing_report:1
- mfa_issue -> account_locked:9, password_reset:1
- mobile_device_issue -> network_issue:2, permission_issue:1
- onboarding_offboarding -> account_locked:6
- password_reset -> password_reset:7
- peripheral_issue -> hardware_issue:4
- permission_issue -> permission_issue:4
- phishing_report -> phishing_report:8
- printer_issue -> printer_issue:12
- server_unavailable -> network_issue:2
- shared_drive_issue -> network_issue:2
- software_install -> permission_issue:6
- voip_telephony -> network_issue:5, software_access:2
- vpn_issue -> vpn_issue:6
- wifi_connectivity -> network_issue:6
- workstation_failure -> hardware_issue:4

## Family-Level Confusion Matrix
- Backend -> IAM:3, Unknown:7
- Hardware -> Hardware:12, IAM:1, Unknown:10
- IAM -> IAM:32
- Networking -> Networking:6, Unknown:17
- Security -> Security:9, Unknown:3
- Software -> IAM:6, Other:1, Software:5, Unknown:7
- Storage -> Other:5, Unknown:6

## Example Fine-Grained Category Errors
- TDX-10001: gt=software_install, pred=permission_issue, gt_family=Software, pred_family=IAM, conf=0.9
- TDX-10002: gt=wifi_connectivity, pred=network_issue, gt_family=Networking, pred_family=Unknown, conf=0.9
- TDX-10004: gt=mfa_issue, pred=account_locked, gt_family=IAM, pred_family=IAM, conf=0.9
- TDX-10005: gt=mfa_issue, pred=password_reset, gt_family=IAM, pred_family=IAM, conf=0.95
- TDX-10007: gt=data_recovery, pred=other, gt_family=Storage, pred_family=Other, conf=0.85
- TDX-10009: gt=disk_space_full, pred=network_issue, gt_family=Storage, pred_family=Unknown, conf=0.9
- TDX-10010: gt=mfa_issue, pred=account_locked, gt_family=IAM, pred_family=IAM, conf=0.9
- TDX-10011: gt=server_unavailable, pred=network_issue, gt_family=Backend, pred_family=Unknown, conf=0.9
- TDX-10012: gt=disk_space_full, pred=network_issue, gt_family=Storage, pred_family=Unknown, conf=0.95
- TDX-10014: gt=data_recovery, pred=other, gt_family=Storage, pred_family=Other, conf=0.9

## Example Family Errors
- TDX-10001: gt_family=Software, pred_family=IAM, gt=software_install, pred=permission_issue
- TDX-10002: gt_family=Networking, pred_family=Unknown, gt=wifi_connectivity, pred=network_issue
- TDX-10007: gt_family=Storage, pred_family=Other, gt=data_recovery, pred=other
- TDX-10009: gt_family=Storage, pred_family=Unknown, gt=disk_space_full, pred=network_issue
- TDX-10011: gt_family=Backend, pred_family=Unknown, gt=server_unavailable, pred=network_issue
- TDX-10012: gt_family=Storage, pred_family=Unknown, gt=disk_space_full, pred=network_issue
- TDX-10014: gt_family=Storage, pred_family=Other, gt=data_recovery, pred=other
- TDX-10019: gt_family=Networking, pred_family=Unknown, gt=internet_access, pred=network_issue
- TDX-10022: gt_family=Hardware, pred_family=Unknown, gt=workstation_failure, pred=hardware_issue
- TDX-10031: gt_family=Backend, pred_family=Unknown, gt=api_failure, pred=network_issue

## Worst Issue Matches
- TDX-10141: issue_emb=0.0, pred='Installer fails due to incorrect deployment scope', best_gt=''
- TDX-10026: issue_emb=0.0307, pred='New hire missing email, VPN, and ERP app access.', best_gt='Offboarding request follow-up'
- TDX-10121: issue_emb=0.0307, pred='New hire missing email, VPN, and ERP app access.', best_gt='Offboarding request follow-up'
- TDX-10094: issue_emb=0.2449, pred='Softphone client unregistered, preventing outbound calls.', best_gt='Meeting room audio problem'
- TDX-10179: issue_emb=0.2633, pred='Permission denied accessing shared team folder', best_gt='Cannot open department drive'

## Worst Resolution Matches
- TDX-10038: res_emb=0.3759, pred='Cause: Sender impersonation from outside. Fix: Quarantined message and blocked similar patterns.', gt='The reported email was phishing and was quarantined. Because the user did not click or submit credentials, the account did not require recovery steps.'
- TDX-10071: res_emb=0.387, pred='Cause: Cached credentials issue. Fix: Clear cached credentials and sign in again.', gt='The mailbox itself was healthy, but Outlook had stale local credentials. Clearing the cached credentials and reauthenticating restored access. The user expressed urgency during the conversation. A secondary issue related to voip_telephony was also reported.'
- TDX-10020: res_emb=0.3874, pred='Cause: Cached credentials issue. Fix: Clear cached credentials and sign in again.', gt='The issue came from an outdated Outlook credential cache. After clearing cached credentials and re-adding the account, email access returned.'
- TDX-10027: res_emb=0.3874, pred='Cause: Cached credentials issue. Fix: Clear cached credentials and sign in again.', gt='The issue came from an outdated Outlook credential cache. After clearing cached credentials and re-adding the account, email access returned.'
- TDX-10108: res_emb=0.4718, pred='Cause: Local application cache corrupted. Fix: Reset local application cache.', gt='Crash logs showed corrupted local app data. Rebuilding the local profile resolved the repeated freezes. A secondary issue related to encryption_issue was also reported.'
