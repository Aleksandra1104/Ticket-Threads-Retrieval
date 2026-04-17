from __future__ import annotations

from typing import List

from ticket_memory.simulation.core.models import IssueVariant


def issue_variant(
    family: str,
    root_cause_id: str,
    titles: List[str],
    user_openers: List[str],
    clarify_agent: List[str],
    clarify_user: List[str],
    first_try_agent: List[str],
    first_try_user_fail: List[str],
    diagnosis_agent: List[str],
    fix_agent: List[str],
    resolution_summary: List[str],
    environment_details: List[str],
    signals: List[str],
    tags: List[str],
) -> IssueVariant:
    return IssueVariant(
        family=family,
        root_cause_id=root_cause_id,
        titles=titles,
        user_openers=user_openers,
        clarify_agent=clarify_agent,
        clarify_user=clarify_user,
        first_try_agent=first_try_agent,
        first_try_user_fail=first_try_user_fail,
        diagnosis_agent=diagnosis_agent,
        fix_agent=fix_agent,
        resolution_summary=resolution_summary,
        environment_details=environment_details,
        signals=signals,
        tags=tags,
    )


IT_SUPPORT_VARIANTS: List[IssueVariant] = []


IT_SUPPORT_VARIANTS.extend([
    issue_variant(
        family="account_locked",
        root_cause_id="failed_attempt_lockout",
        titles=["Cannot log in", "Account locked", "Unable to access account", "Sign-in issue"],
        user_openers=[
            "I cannot log into my account even though I believe the password is correct.",
            "My account says it is locked and I cannot access email or the portal.",
            "I was trying to sign in and now it says my account is locked.",
        ],
        clarify_agent=[
            "Are you seeing a specific lockout message or a generic password error?",
            "Can you confirm whether the system says the account is locked after failed attempts?",
        ],
        clarify_user=[
            "It specifically says the account is locked.",
            "The message says my account is locked after too many attempts.",
        ],
        first_try_agent=[
            "Please wait a few minutes and then try signing in once more.",
            "Please try one more sign-in attempt and let me know whether the message changes.",
        ],
        first_try_user_fail=[
            "I tried that and it still says the account is locked.",
            "I waited and tried again, but I still cannot sign in.",
        ],
        diagnosis_agent=[
            "I checked the authentication logs and confirmed the account is in a locked state.",
            "I reviewed the account status and found a lockout from repeated failed sign-ins.",
        ],
        fix_agent=[
            "I unlocked the account and you should be able to sign in now.",
            "I cleared the lockout and confirmed the account can authenticate again.",
        ],
        resolution_summary=[
            "The account was locked after failed sign-in attempts. The lockout was cleared and access was restored.",
            "Authentication logs showed a lockout state. The account was unlocked and sign-in access was restored.",
        ],
        environment_details=["This started this morning.", "The issue is affecting both email and the portal."],
        signals=["It happens on both my laptop and phone.", "I never got in after the last failed prompt."],
        tags=["iam", "login"],
    ),
    issue_variant(
        family="password_reset",
        root_cause_id="expired_reset_token",
        titles=["Password reset help", "Forgot password", "Cannot reset password", "Reset link not working"],
        user_openers=[
            "I forgot my password and the reset link is not working for me.",
            "I need help resetting my password for my account.",
            "I cannot get the password reset email to work.",
        ],
        clarify_agent=[
            "Are you receiving the reset email, or is the link itself failing?",
            "Can you confirm whether the reset email arrives and what happens when you click it?",
        ],
        clarify_user=[
            "I get the email, but the link says it is invalid.",
            "The email arrives, but when I click the link it says the token is expired.",
        ],
        first_try_agent=[
            "Please request a new reset email and use only the newest link.",
            "Please close the older reset emails and try only the latest reset URL.",
        ],
        first_try_user_fail=[
            "I tried the newest one and it still fails.",
            "I requested another reset email and the latest link still does not work.",
        ],
        diagnosis_agent=[
            "I checked the reset status and found the previous token had expired.",
            "I reviewed the account and saw the reset token was no longer valid.",
        ],
        fix_agent=[
            "I generated a fresh password reset link and it should work now.",
            "I issued a new reset email and removed the expired token from the account.",
        ],
        resolution_summary=[
            "The prior reset token had expired. A fresh password reset link was issued and the reset flow worked again.",
            "The reset failure was caused by an invalid or expired token. A new reset link was generated and the process was restored.",
        ],
        environment_details=["The old email arrived about ten minutes earlier.", "This is for my main staff account."],
        signals=["The browser opens the page and then says invalid token.", "It fails before I can enter a new password."],
        tags=["iam", "credentials"],
    ),
    issue_variant(
        family="mfa_issue",
        root_cause_id="token_desync",
        titles=["MFA code not working", "Cannot complete two-factor sign-in", "Authenticator app issue"],
        user_openers=[
            "My authenticator code keeps getting rejected when I try to sign in.",
            "I cannot get past the MFA prompt even though the password works.",
            "The two-factor login is failing on my phone app.",
        ],
        clarify_agent=[
            "Is the code rejected immediately, or does the prompt time out?",
            "Are you using the mobile authenticator app, SMS, or a hardware token?",
        ],
        clarify_user=[
            "It accepts my password and then says the authenticator code is invalid.",
            "I am using the app on my phone and the code is rejected right away.",
        ],
        first_try_agent=[
            "Please check that the time on your phone is set automatically and then try a fresh code.",
            "Please refresh the authenticator app and enter the newest code only.",
        ],
        first_try_user_fail=[
            "I synced the phone clock and it still rejects the code.",
            "I tried the newest code again and it still says invalid.",
        ],
        diagnosis_agent=[
            "I checked the MFA registration and found the token had drifted out of sync.",
            "I reviewed the second-factor enrollment and the device token is out of sync with the identity platform.",
        ],
        fix_agent=[
            "I reset the MFA enrollment and had you register the authenticator app again. Sign-in worked after that.",
            "I re-synced the MFA token and the verification prompt now accepts the code.",
        ],
        resolution_summary=[
            "The MFA device token was out of sync. Re-syncing or re-registering the authenticator restored sign-in.",
            "Second-factor verification was failing because the enrolled authenticator had drifted out of sync. Resetting MFA resolved the issue.",
        ],
        environment_details=["This started after I changed phones.", "I can sign in with password only until the second step appears."],
        signals=["Backup codes are not with me right now.", "SMS fallback is not configured on this account."],
        tags=["iam", "mfa", "security"],
    ),
    issue_variant(
        family="permission_issue",
        root_cause_id="missing_security_group",
        titles=["No access to shared folder", "Permission denied", "Cannot open department drive", "Shared drive access issue"],
        user_openers=[
            "I get permission denied when I open the department shared drive.",
            "I cannot access the shared folder that my team uses.",
            "The system says I do not have permission to open the project folder.",
        ],
        clarify_agent=[
            "Can you confirm which shared folder path you are trying to access?",
            "Which folder are you opening, and did you have access to it previously?",
        ],
        clarify_user=[
            "It is the project folder under the department shared drive and I used to have access.",
            "It is the shared team folder and I could open it before this week.",
        ],
        first_try_agent=[
            "Please sign out and back into your computer once, then try opening the folder again.",
            "Please disconnect and reconnect the shared drive once and let me know if access changes.",
        ],
        first_try_user_fail=[
            "I tried that and I still get the same permission denied message.",
            "That did not help. I still cannot open the folder.",
        ],
        diagnosis_agent=[
            "I checked the access groups for that folder and found your account was missing the required security group.",
            "I reviewed your group membership and confirmed the folder permissions are tied to a group your account does not currently have.",
        ],
        fix_agent=[
            "I added the correct access group to your account and the folder should open now.",
            "Once the required security group was applied to your account, access was restored.",
        ],
        resolution_summary=[
            "The user was missing the required security group for the shared folder. Adding the proper access group restored access.",
            "Missing folder permissions caused the access error. The correct group membership was applied and access was restored.",
        ],
        environment_details=["This affects a project folder that my whole team uses.", "I had access earlier in the week."],
        signals=["The top-level drive opens, but one subfolder does not.", "A teammate beside me can still open it."],
        tags=["iam", "storage", "authorization"],
    ),
    issue_variant(
        family="onboarding_offboarding",
        root_cause_id="provisioning_workflow_delay",
        titles=["New hire access missing", "Account provisioning incomplete", "Offboarding request follow-up"],
        user_openers=[
            "Our new hire still does not have the right accounts provisioned.",
            "I submitted an onboarding request, but the employee cannot access required systems.",
            "The offboarding request was submitted, but some access still appears active.",
        ],
        clarify_agent=[
            "Is this for onboarding or deprovisioning, and which systems are affected?",
            "Can you share the employee start or departure date and the main access that is missing or still active?",
        ],
        clarify_user=[
            "This is for onboarding and they are missing email, VPN, and the ERP app.",
            "It is an onboarding request for someone who started today and the account setup looks incomplete.",
        ],
        first_try_agent=[
            "Please confirm whether the manager approval and HR record have both completed.",
            "Please check whether the onboarding task shows fully approved in the portal.",
        ],
        first_try_user_fail=[
            "The approvals are complete, but the accounts are still not ready.",
            "Everything looks approved on our side, but the user is still missing access.",
        ],
        diagnosis_agent=[
            "I reviewed the provisioning workflow and one downstream account creation task was still queued.",
            "I checked the onboarding automation and found the identity workflow did not finish pushing all entitlements.",
        ],
        fix_agent=[
            "I re-ran the provisioning task and the missing access is now assigned.",
            "I completed the failed downstream provisioning step and the required accounts are now available.",
        ],
        resolution_summary=[
            "The onboarding workflow had not fully completed a downstream provisioning task. Re-running the workflow restored the missing access.",
            "Provisioning was incomplete because one automated entitlement step had stalled. Restarting that task completed the user setup.",
        ],
        environment_details=["The employee started today.", "Manager approval was submitted yesterday afternoon."],
        signals=["Email exists now, but the business app does not.", "The access request number is already approved."],
        tags=["iam", "provisioning"],
    ),
])


IT_SUPPORT_VARIANTS.extend([
    issue_variant(
        family="malware_infection",
        root_cause_id="endpoint_antivirus_detection",
        titles=["Antivirus alert on laptop", "Possible malware infection", "Suspicious pop-ups on workstation"],
        user_openers=[
            "My laptop showed an antivirus alert and I am worried it may be infected.",
            "I started seeing suspicious pop-ups and the machine is behaving strangely.",
            "Security software flagged something on my workstation and I need help.",
        ],
        clarify_agent=[
            "Did the antivirus quarantine the item already, or is the alert still active?",
            "Are you seeing pop-ups only, or is the machine also unusually slow or unstable?",
        ],
        clarify_user=[
            "The alert is still active and the machine also seems slower than normal.",
            "I saw the antivirus warning first, and then the system started behaving oddly.",
        ],
        first_try_agent=[
            "Please disconnect the device from the network and leave it powered on while I review the alert.",
            "Please stop using the machine for anything else and keep the security alert visible if possible.",
        ],
        first_try_user_fail=[
            "It is disconnected now, but the alert is still there.",
            "I stopped using it and the warning remains active.",
        ],
        diagnosis_agent=[
            "I reviewed the endpoint security console and it shows a malware detection that needs remediation.",
            "I checked the antivirus event and the device has an active malicious file detection.",
        ],
        fix_agent=[
            "We isolated the device, removed the detected threat, and verified that the endpoint is now clean.",
            "After running the remediation steps from the security console, the malware detection cleared and the device returned to normal.",
        ],
        resolution_summary=[
            "Endpoint security detected malware on the device. Isolating the machine and completing remediation removed the threat.",
            "The workstation issue was caused by a confirmed malware detection. Security remediation cleared the malicious file and restored the endpoint.",
        ],
        environment_details=["The warning appeared shortly after opening an attachment.", "Performance became noticeably worse at the same time."],
        signals=["The antivirus alert is still visible.", "The machine feels slower and less responsive."],
        tags=["security", "malware", "endpoint"],
    ),
    issue_variant(
        family="encryption_issue",
        root_cause_id="bitlocker_recovery_key_required",
        titles=["BitLocker recovery prompt", "Encrypted drive issue", "Cannot unlock secure laptop"],
        user_openers=[
            "My laptop booted into a BitLocker recovery prompt and I cannot get in.",
            "The encrypted drive is asking for a recovery key after reboot.",
            "I cannot unlock the company laptop because the disk encryption screen appeared.",
        ],
        clarify_agent=[
            "Did this happen after a BIOS update, hardware change, or unexpected restart?",
            "Are you looking at a BitLocker or FileVault recovery screen right now?",
        ],
        clarify_user=[
            "Yes, it is a BitLocker screen asking for the recovery key.",
            "It happened after a reboot and now I only see the recovery prompt.",
        ],
        first_try_agent=[
            "Please stay on the recovery screen while I verify the device record and recovery key.",
            "Please do not keep retrying random keys while I check the encryption record.",
        ],
        first_try_user_fail=[
            "I am still at the recovery screen and cannot continue.",
            "I do not have the recovery key myself.",
        ],
        diagnosis_agent=[
            "I checked the device record and the encryption platform has a valid recovery key stored.",
            "I reviewed the endpoint inventory and the device is prompting for BitLocker recovery after a security-state change.",
        ],
        fix_agent=[
            "I provided the recovery key and the device unlocked successfully. I also confirmed the protector state afterwards.",
            "Using the stored recovery key got the laptop booted again, and the encryption protector was then resynced.",
        ],
        resolution_summary=[
            "The device required its stored BitLocker recovery key after a security-state change. Unlocking with the recovery key restored access.",
            "Disk encryption recovery was triggered by a platform state change. Retrieving the valid recovery key and resyncing the protector resolved the issue.",
        ],
        environment_details=["This happened after the device restarted overnight.", "I am blocked before Windows loads."],
        signals=["The screen is specifically asking for a recovery key.", "I cannot reach the normal login screen."],
        tags=["security", "encryption", "bitlocker"],
    ),
    issue_variant(
        family="server_unavailable",
        root_cause_id="service_restart_required",
        titles=["Cannot reach internal server", "Application server unavailable", "Production environment down"],
        user_openers=[
            "I cannot reach one of our internal servers right now.",
            "The production environment for an internal app appears unavailable.",
            "The server-hosted application is down from my side and from other users as well.",
        ],
        clarify_agent=[
            "Is the outage limited to one service, or does the whole server seem unreachable?",
            "Are other users reporting the same outage, or does it appear isolated to you?",
        ],
        clarify_user=[
            "It seems to affect other users too, not just me.",
            "The application and the server endpoint both appear down.",
        ],
        first_try_agent=[
            "Please retry once while I check whether the service is already recovering.",
            "Please confirm the exact server or environment name while I verify its current status.",
        ],
        first_try_user_fail=[
            "It is still down and not responding.",
            "The retry did not help. The service is still unavailable.",
        ],
        diagnosis_agent=[
            "I checked the service health and one critical application service on the server had stopped.",
            "I reviewed the server monitoring and the application service is down even though the host itself is reachable.",
        ],
        fix_agent=[
            "I restarted the failed service and the environment came back online.",
            "After restoring the stopped application service, the server-hosted application became reachable again.",
        ],
        resolution_summary=[
            "The outage was caused by a stopped application service on the server. Restarting that service restored availability.",
            "The server issue traced back to a failed application service rather than a full host outage. Restarting the service resolved the incident.",
        ],
        environment_details=["This is affecting a production environment.", "At least one coworker sees the same outage."],
        signals=["The server endpoint times out.", "Multiple users appear impacted."],
        tags=["backend", "server", "outage"],
    ),
    issue_variant(
        family="database_connection",
        root_cause_id="expired_connection_secret",
        titles=["Database connection failing", "Application cannot reach database", "SQL timeout issue"],
        user_openers=[
            "An internal application can no longer connect to its database.",
            "We are seeing database connection failures and timeouts in the app.",
            "The service is up, but every database call seems to fail.",
        ],
        clarify_agent=[
            "Are you seeing authentication failures, connection timeouts, or both?",
            "Did this begin after a deployment or secret rotation?",
        ],
        clarify_user=[
            "The logs show authentication failures when the app connects to the database.",
            "It started after a recent change window and the app cannot authenticate.",
        ],
        first_try_agent=[
            "Please retry once while I verify whether the database itself is healthy.",
            "Please confirm the application environment so I can check its connection settings.",
        ],
        first_try_user_fail=[
            "The retry still fails with the same database error.",
            "The database calls are still failing after another test.",
        ],
        diagnosis_agent=[
            "I checked the application configuration and the database secret being used has expired.",
            "I reviewed the connection settings and the service is trying to authenticate with an outdated database credential.",
        ],
        fix_agent=[
            "I updated the application to use the current database secret and connections are succeeding now.",
            "After refreshing the expired database credential in the service configuration, the app connected successfully.",
        ],
        resolution_summary=[
            "Database connectivity was failing because the application was using an expired secret. Updating the credential restored access.",
            "The service could not authenticate to the database due to an outdated connection secret. Refreshing the secret resolved the issue.",
        ],
        environment_details=["The application tier still appears online.", "This started after a scheduled maintenance window."],
        signals=["The errors mention database authentication.", "The host is up, but queries time out or fail."],
        tags=["backend", "database", "credentials"],
    ),
    issue_variant(
        family="api_failure",
        root_cause_id="upstream_contract_change",
        titles=["Internal API integration failing", "Service-to-service API error", "Microservice call broken"],
        user_openers=[
            "One internal service is failing when it calls another API.",
            "We are seeing errors in an integration between two internal systems.",
            "A microservice API call is failing and blocking the workflow.",
        ],
        clarify_agent=[
            "Are the failures authentication-related, timeouts, or response-format errors?",
            "Did the issue begin after a recent deployment to either service?",
        ],
        clarify_user=[
            "The logs suggest the API response format changed and the client no longer handles it.",
            "This started after a recent deployment to the upstream service.",
        ],
        first_try_agent=[
            "Please retry the workflow once while I compare the latest request and response logs.",
            "Please confirm the source and target services so I can review the latest traces.",
        ],
        first_try_user_fail=[
            "The workflow still fails and the error is unchanged.",
            "Another test run hit the same integration error.",
        ],
        diagnosis_agent=[
            "I reviewed the traces and the upstream API contract changed in a way the client service is not handling.",
            "The service logs show a response payload change from the upstream API that breaks the downstream parser.",
        ],
        fix_agent=[
            "I rolled the client configuration forward to the current API contract and the integration started working again.",
            "After updating the downstream service to match the upstream response format, the API calls succeeded.",
        ],
        resolution_summary=[
            "The API failure was caused by an upstream contract change that the client service was not prepared for. Updating the client handling restored the integration.",
            "Service-to-service calls were failing because the upstream API response changed. Aligning the downstream parser with the new contract resolved the issue.",
        ],
        environment_details=["This affects an automated workflow between internal systems.", "The failure appeared after a deployment."],
        signals=["The error is in service logs rather than a user interface.", "The workflow breaks on one API call consistently."],
        tags=["backend", "api", "integration"],
    ),
])


IT_SUPPORT_VARIANTS.extend([
    issue_variant(
        family="application_crash",
        root_cause_id="corrupt_local_profile_cache",
        titles=["Application keeps crashing", "Program not responding", "Business app freezes on launch"],
        user_openers=[
            "One of my work applications keeps freezing when I open it.",
            "The program launches and then becomes not responding.",
            "The business app crashes as soon as I try to use it.",
        ],
        clarify_agent=[
            "Does the crash happen on launch or after you open a specific file or module?",
            "Are other users seeing the same issue, or does it seem limited to your profile?",
        ],
        clarify_user=[
            "It opens and freezes before I can do much with it.",
            "It seems limited to me because my teammate can still use the app.",
        ],
        first_try_agent=[
            "Please fully close the application and reopen it without loading recent files.",
            "Please reboot once and then launch the app again before opening any documents.",
        ],
        first_try_user_fail=[
            "I tried that and it still freezes right away.",
            "After rebooting, the app still becomes unresponsive.",
        ],
        diagnosis_agent=[
            "I checked the local app data and the profile cache for the application appears corrupted.",
            "I reviewed the crash logs and they point to corrupted local profile data for the application.",
        ],
        fix_agent=[
            "I reset the local application cache and the program now opens normally.",
            "After rebuilding the application's local profile data, the crashes stopped.",
        ],
        resolution_summary=[
            "The application was crashing because its local profile cache had become corrupted. Resetting the cache restored normal operation.",
            "Crash logs showed corrupted local app data. Rebuilding the local profile resolved the repeated freezes.",
        ],
        environment_details=["Other apps on the laptop seem fine.", "The issue started after the app updated."],
        signals=["It freezes before I can open any real work files.", "Another coworker can still use the same system."],
        tags=["software", "crash"],
    ),
    issue_variant(
        family="browser_issue",
        root_cause_id="corrupt_browser_cache",
        titles=["Browser certificate warning", "Website only fails in one browser", "Browser keeps erroring"],
        user_openers=[
            "I keep getting certificate or privacy warnings in the browser for an internal site.",
            "The website works for others, but my browser keeps failing.",
            "One browser keeps breaking on a work site even though the site is up.",
        ],
        clarify_agent=[
            "Is this happening in one browser only or in all browsers?",
            "Do you see a certificate warning, login loop, or a blank page?",
        ],
        clarify_user=[
            "It seems limited to one browser and I see either a warning or a login loop.",
            "Another browser works, but my main one keeps failing on the same site.",
        ],
        first_try_agent=[
            "Please open an incognito window and test the site there.",
            "Please clear the browser cache for that site and test again.",
        ],
        first_try_user_fail=[
            "I tried a private window and it still behaves strangely.",
            "I cleared the cache and it still loops or warns.",
        ],
        diagnosis_agent=[
            "I reviewed the browser state and it looks like the local cache and cookies for that site are corrupted.",
            "The issue appears to be tied to stale site data in the browser rather than the service itself.",
        ],
        fix_agent=[
            "I had the corrupted browser profile data cleared and the site loads normally now.",
            "After fully resetting the affected browser cache and cookies for that site, the warning stopped.",
        ],
        resolution_summary=[
            "The browser problem was caused by stale or corrupt local site data. Clearing the affected cache and cookies restored access.",
            "The site itself was healthy; the issue came from corrupted browser cache or cookie state on the endpoint. Resetting that data resolved it.",
        ],
        environment_details=["Other users can open the site.", "The site sometimes works in another browser."],
        signals=["The problem is limited to one browser profile.", "The page either loops or shows a trust-style warning."],
        tags=["software", "browser", "cache"],
    ),
    issue_variant(
        family="shared_drive_issue",
        root_cause_id="stale_drive_mapping",
        titles=["Shared drive not mapping", "Cannot reach network drive", "SharePoint or OneDrive path issue"],
        user_openers=[
            "My shared drive is not mapping when I sign in.",
            "I cannot reach the team network drive from File Explorer.",
            "The SharePoint or synced drive path is missing from my workstation.",
        ],
        clarify_agent=[
            "Is this a traditional mapped drive, SharePoint sync, or OneDrive shortcut issue?",
            "Do you see the drive letter missing entirely, or is it there but inaccessible?",
        ],
        clarify_user=[
            "The drive letter is missing entirely after sign-in.",
            "It is a mapped network drive and it no longer reconnects.",
        ],
        first_try_agent=[
            "Please sign out and back in once, then wait a minute and check whether the mapping returns.",
            "Please disconnect any stale mapping for that drive and then test again.",
        ],
        first_try_user_fail=[
            "I signed out and it still does not remap.",
            "I removed the old mapping and it still does not come back.",
        ],
        diagnosis_agent=[
            "I checked the drive mapping policy and your workstation had a stale mapping entry.",
            "I reviewed the logon mapping process and the stored drive path on the workstation was outdated.",
        ],
        fix_agent=[
            "I refreshed the drive mapping policy and the shared drive appeared again.",
            "After removing the stale entry and reapplying the mapping, the drive reconnected successfully.",
        ],
        resolution_summary=[
            "The shared drive problem was caused by a stale local mapping entry. Refreshing the mapping policy restored access.",
            "An outdated drive mapping on the workstation prevented the share from reconnecting. Reapplying the mapping resolved the issue.",
        ],
        environment_details=["This used to reconnect automatically at sign-in.", "Only one shared location appears missing."],
        signals=["The drive letter never comes back after login.", "The path exists for teammates."],
        tags=["storage", "sharepoint", "drive_mapping"],
    ),
    issue_variant(
        family="data_recovery",
        root_cause_id="restorable_backup_snapshot",
        titles=["Need deleted file restored", "Recover previous file version", "Data restore request"],
        user_openers=[
            "I need a file restored because it was deleted or overwritten.",
            "Can IT help recover an older version of a document from backup?",
            "A shared document was changed and we need the previous copy back.",
        ],
        clarify_agent=[
            "Do you need a deleted file restored or an earlier version of an existing file?",
            "Can you share the file path and roughly when the last good version existed?",
        ],
        clarify_user=[
            "It is an overwritten file and we need the version from yesterday.",
            "The file was deleted from the shared folder earlier today.",
        ],
        first_try_agent=[
            "Please check the recycle bin or version history once in case the file is already available there.",
            "Please confirm the exact folder path and filename before I start the restore.",
        ],
        first_try_user_fail=[
            "I already checked and the version I need is not there.",
            "The recycle bin does not have the file anymore.",
        ],
        diagnosis_agent=[
            "I checked the backup snapshots and there is a recoverable copy from before the change.",
            "I reviewed version history and backup retention, and the file can be restored from a prior snapshot.",
        ],
        fix_agent=[
            "I restored the earlier file version to the original location and you should see it now.",
            "I recovered the requested copy from backup and placed it back in the shared folder.",
        ],
        resolution_summary=[
            "The requested data was recoverable from backup or snapshot history. Restoring the prior version returned the missing file.",
            "A previous file version was available in backup retention. Recovering that copy restored the needed data.",
        ],
        environment_details=["The change happened earlier today.", "This is on a team share rather than a local desktop folder."],
        signals=["We need yesterday's version specifically.", "The normal recycle bin did not help."],
        tags=["storage", "backup", "restore"],
    ),
    issue_variant(
        family="disk_space_full",
        root_cause_id="local_profile_storage_buildup",
        titles=["Disk space full", "Laptop storage almost full", "No free space left on drive"],
        user_openers=[
            "My laptop says the disk is almost full and apps are starting to fail.",
            "I am getting low storage warnings and cannot save files properly.",
            "The local drive is basically full and performance is getting bad.",
        ],
        clarify_agent=[
            "Is this the local system drive or a shared server location that is full?",
            "Are you mainly seeing save failures, performance issues, or both?",
        ],
        clarify_user=[
            "It is the local C drive and I am seeing both warnings and save failures.",
            "The laptop is slow and several programs complain about free space.",
        ],
        first_try_agent=[
            "Please empty the recycle bin and close any large apps before checking the free space again.",
            "Please restart the laptop once and then confirm whether the storage warning remains.",
        ],
        first_try_user_fail=[
            "I did that and there is still barely any space available.",
            "The warning is still there and the drive remains almost full.",
        ],
        diagnosis_agent=[
            "I checked the storage breakdown and the local profile cache is consuming far more space than expected.",
            "I reviewed the device storage and most of the space is tied up in temporary local profile data.",
        ],
        fix_agent=[
            "I cleared the oversized local cache and temp files, and the device now has usable free space again.",
            "After cleaning the profile cache and temporary files, the drive freed up enough space and save operations worked again.",
        ],
        resolution_summary=[
            "Low disk space was caused by accumulated temporary or profile cache data. Cleaning that data restored usable free space.",
            "The endpoint storage issue came from excessive local cache growth. Clearing temporary data resolved the disk-space problem.",
        ],
        environment_details=["This has been getting worse over the last few days.", "The issue is on the laptop itself, not a network share."],
        signals=["Saving files has started to fail.", "Windows shows almost no free space remaining."],
        tags=["storage", "disk", "performance"],
    ),
    issue_variant(
        family="phishing_report",
        root_cause_id="suspicious_sender_impersonation",
        titles=["Suspicious email reported", "Possible phishing message", "Question about malicious email"],
        user_openers=[
            "I received a suspicious email and I think it may be phishing.",
            "An email asking for credentials looks fake and I wanted to report it.",
            "I clicked nothing yet, but the message looks suspicious and urgent.",
        ],
        clarify_agent=[
            "Did you click any links, open attachments, or enter credentials?",
            "Can you tell me whether the sender looked internal or clearly external?",
        ],
        clarify_user=[
            "I did not click anything. I just want to make sure it is safe.",
            "The sender looked internal at first glance, but the message seems off.",
        ],
        first_try_agent=[
            "Please do not interact with the email further and report it using the phishing button if available.",
            "Please leave the message in place for the moment and do not open any attachments.",
        ],
        first_try_user_fail=[
            "I reported it, but I want to know whether my account is still safe.",
            "I did not open it, but I am still concerned because it looked convincing.",
        ],
        diagnosis_agent=[
            "I checked the message headers and it appears to be sender impersonation from outside the organization.",
            "I reviewed the security traces and the email matches a known phishing pattern.",
        ],
        fix_agent=[
            "We quarantined the message and blocked similar sender patterns. Since you did not interact with it, no further account action was needed.",
            "The message was confirmed as phishing and removed from mailboxes. No compromise steps were required because you did not click the link.",
        ],
        resolution_summary=[
            "The reported email was phishing and was quarantined. Because the user did not click or submit credentials, the account did not require recovery steps.",
            "Security review confirmed the message was malicious sender impersonation. The email was removed and similar messages were blocked.",
        ],
        environment_details=["The email claimed to be urgent and asked me to log in.", "It mentioned payroll or invoice review."],
        signals=["I did not click the link.", "The sender display name looked familiar, but the address did not."],
        tags=["security", "phishing", "email"],
    ),
])


IT_SUPPORT_VARIANTS.extend([
    issue_variant(
        family="peripheral_issue",
        root_cause_id="docking_station_firmware",
        titles=["Docking station not detecting monitors", "Keyboard or mouse keeps disconnecting", "Peripheral issue"],
        user_openers=[
            "My docking station is not detecting both monitors anymore.",
            "The keyboard and mouse keep disconnecting from the dock.",
            "I plugged everything in, but the peripherals keep dropping in and out.",
        ],
        clarify_agent=[
            "Is this mainly affecting the dock, monitors, or USB accessories?",
            "Did the peripherals stop working all at once or one by one?",
        ],
        clarify_user=[
            "The monitors and keyboard are both connected through the dock and both are unstable.",
            "It seems centered around the docking station because direct laptop connections are fine.",
        ],
        first_try_agent=[
            "Please fully power-cycle the dock and reconnect it after thirty seconds.",
            "Please disconnect the dock from power and the laptop, then reconnect it cleanly.",
        ],
        first_try_user_fail=[
            "I power-cycled the dock and the monitors still do not stay connected.",
            "That did not help. The dock still drops the peripherals.",
        ],
        diagnosis_agent=[
            "I checked the dock model and it needs a firmware refresh to stabilize those ports.",
            "I reviewed the peripheral errors and they point to outdated docking station firmware.",
        ],
        fix_agent=[
            "I updated the docking station firmware and the peripherals stayed connected after retesting.",
            "After applying the current dock firmware, the monitors and USB devices remained stable.",
        ],
        resolution_summary=[
            "The peripheral instability was caused by outdated docking station firmware. Updating the dock resolved the disconnects.",
            "Monitor and USB drops were traced to stale dock firmware. Refreshing the firmware restored stable peripheral connections.",
        ],
        environment_details=["This started after moving to another desk.", "Directly plugging a monitor into the laptop works."],
        signals=["The same dock affects multiple peripherals.", "The laptop itself seems fine without the dock."],
        tags=["hardware", "dock", "peripherals"],
    ),
    issue_variant(
        family="printer_issue",
        root_cause_id="stale_printer_mapping",
        titles=["Printer offline", "Cannot print", "Printer not working", "Print jobs stuck"],
        user_openers=[
            "I cannot print to the office printer and it shows offline.",
            "The department printer is unavailable from my computer.",
            "Print jobs are stuck and the printer looks offline.",
        ],
        clarify_agent=[
            "Which printer are you trying to use, and is anyone nearby able to print to it?",
            "Can you confirm the printer name and whether this happens with all print jobs?",
        ],
        clarify_user=[
            "It is the second-floor office printer and my jobs just sit in the queue.",
            "It is the shared department printer and the queue is stuck on my machine.",
        ],
        first_try_agent=[
            "Please remove the stuck print jobs from the queue and try sending a small test page.",
            "Please clear the local print queue and attempt one new test print.",
        ],
        first_try_user_fail=[
            "The test page still did not print and it still looks offline.",
            "I cleared the queue, but the printer still does not work from my computer.",
        ],
        diagnosis_agent=[
            "I reviewed the printer queue and connection and found the local printer mapping had become stale.",
            "I checked the printer mapping on your machine and the local queue is pointing to an old print path.",
        ],
        fix_agent=[
            "I removed and re-added the printer, and the test page printed successfully after that.",
            "After clearing the queue and reinstalling the printer mapping, printing succeeded.",
        ],
        resolution_summary=[
            "The printer issue was caused by a stuck queue or stale local printer mapping. Re-adding the printer and clearing the queue restored printing.",
            "A stale local printer mapping prevented printing. After reinstalling the printer and clearing the queue, printing worked again.",
        ],
        environment_details=["Other people can print to it right now.", "This is the shared printer near the finance area."],
        signals=["The queue just says error or offline.", "The printer appears available to other staff."],
        tags=["hardware", "printer"],
    ),
    issue_variant(
        family="mobile_device_issue",
        root_cause_id="mdm_policy_noncompliant",
        titles=["Company phone not syncing", "Managed mobile device issue", "Tablet cannot access work apps"],
        user_openers=[
            "My company phone stopped syncing work email and Teams.",
            "The managed tablet says it is noncompliant and work apps will not open.",
            "I cannot use the company mobile device for work apps anymore.",
        ],
        clarify_agent=[
            "Is the device iPhone, Android, or a managed tablet?",
            "Are work apps failing to launch, or is the issue mainly email and sign-in?",
        ],
        clarify_user=[
            "It is an iPhone and the work apps say the device is not compliant.",
            "It is mainly email and Teams, and both now say the device needs attention.",
        ],
        first_try_agent=[
            "Please open the company portal app and let it complete a fresh compliance check.",
            "Please reconnect the device to Wi-Fi, open the management app, and sync policies once.",
        ],
        first_try_user_fail=[
            "I opened the portal app and it still says the device is not compliant.",
            "I synced the device and work apps still will not open.",
        ],
        diagnosis_agent=[
            "I checked the MDM console and the device is blocked because its compliance policy has not updated correctly.",
            "I reviewed the mobile device record and it is stuck in a noncompliant policy state.",
        ],
        fix_agent=[
            "I refreshed the device policy in MDM and the work apps can open again now.",
            "After resyncing the device record and forcing policy evaluation, the phone returned to a compliant state.",
        ],
        resolution_summary=[
            "The company-managed mobile device was blocked by a stale noncompliant policy state. Refreshing the MDM policy restored access.",
            "Mobile work apps were failing because the device record was stuck as noncompliant in MDM. Re-evaluating the device policy resolved the issue.",
        ],
        environment_details=["This started after the phone updated overnight.", "The personal apps still work fine."],
        signals=["The company portal app shows a warning banner.", "Only managed work apps are affected."],
        tags=["hardware", "mobile", "mdm"],
    ),
    issue_variant(
        family="email_issue",
        root_cause_id="stale_outlook_credentials",
        titles=["Cannot access email", "Mailbox not loading", "Outlook keeps asking for password", "Email issue"],
        user_openers=[
            "My email inbox is not loading and I cannot read messages.",
            "Outlook keeps asking for credentials and my mailbox will not open.",
            "I cannot access my email from the desktop app.",
        ],
        clarify_agent=[
            "Is this happening in the web mailbox, the desktop app, or both?",
            "Can you confirm whether this affects Outlook desktop only or webmail too?",
        ],
        clarify_user=[
            "Webmail seems okay, but the desktop Outlook app will not open the mailbox correctly.",
            "It is mainly happening in Outlook on my computer. The web version looks fine.",
        ],
        first_try_agent=[
            "Please fully close Outlook and reopen it once more.",
            "Please sign out of Outlook, close the app, and open it again.",
        ],
        first_try_user_fail=[
            "I tried that and it is still asking for credentials.",
            "That did not help. Outlook still keeps prompting me and the mailbox will not load.",
        ],
        diagnosis_agent=[
            "I reviewed the account and the mailbox is healthy. The problem looks like stale local credentials in Outlook.",
            "I checked the mailbox status and it looks healthy, so the issue appears to be with the local Outlook profile.",
        ],
        fix_agent=[
            "After clearing the cached credentials and signing in again, Outlook opened the mailbox normally.",
            "The Outlook credential cache was cleared and the account was re-added. The mailbox now loads correctly.",
        ],
        resolution_summary=[
            "The mailbox itself was healthy, but Outlook had stale local credentials. Clearing the cached credentials and reauthenticating restored access.",
            "The issue came from an outdated Outlook credential cache. After clearing cached credentials and re-adding the account, email access returned.",
        ],
        environment_details=["Webmail still works normally.", "The desktop client started failing this morning."],
        signals=["The prompt keeps coming back after I enter the password.", "Only the desktop app is affected."],
        tags=["software", "email", "outlook"],
    ),
    issue_variant(
        family="software_install",
        root_cause_id="missing_admin_deployment_scope",
        titles=["Software install request failed", "Cannot install application", "New software needed"],
        user_openers=[
            "I need a software package installed, but the self-service install is failing.",
            "The application update keeps failing on my laptop.",
            "I requested new software and the installer will not complete.",
        ],
        clarify_agent=[
            "Is this a new software request, or is an existing app update failing?",
            "Do you see a permissions error, package error, or a general install failure?",
        ],
        clarify_user=[
            "This is an approved software request, but the installer says I do not have permission.",
            "It starts the install and then fails before completion.",
        ],
        first_try_agent=[
            "Please retry the installation from Software Center once more and note whether the same error appears.",
            "Please restart the laptop and try the managed installer one more time.",
        ],
        first_try_user_fail=[
            "I retried it and the install still fails the same way.",
            "After the reboot it still says the package cannot be installed.",
        ],
        diagnosis_agent=[
            "I checked the deployment and your device was not in the approved install scope for that package.",
            "I reviewed the software deployment policy and the application was not targeted correctly to your machine.",
        ],
        fix_agent=[
            "I corrected the deployment scope and the install completed successfully after that.",
            "After assigning the package properly to your device, the installation finished without errors.",
        ],
        resolution_summary=[
            "The software install failed because the package deployment scope did not include the device. Correcting the deployment restored installation.",
            "Installation was blocked by an incorrect software assignment policy. Once the package was targeted correctly, the install succeeded.",
        ],
        environment_details=["This is for an approved business application.", "The request was already approved by my manager."],
        signals=["The installer starts but then stops with a permissions-style error.", "Self-service shows the app, but it will not complete."],
        tags=["software", "deployment"],
    ),
])


IT_SUPPORT_VARIANTS.extend([
    issue_variant(
        family="vpn_issue",
        root_cause_id="outdated_vpn_profile",
        titles=["VPN not connecting", "Cannot connect to VPN from home", "Remote access issue", "VPN connection failed"],
        user_openers=[
            "My VPN will not connect from home and I need access to internal systems.",
            "I am remote and the VPN connection keeps failing.",
            "The VPN client says connection failed every time I try to connect.",
        ],
        clarify_agent=[
            "Are you seeing the failure before sign-in, during authentication, or after it starts connecting?",
            "Can you confirm whether the VPN client shows an authentication error or a connection error?",
        ],
        clarify_user=[
            "I can enter my credentials, but then the VPN says the connection cannot be established.",
            "It fails after authentication and says connection failed.",
        ],
        first_try_agent=[
            "Please sign out of the VPN client, restart it, and try connecting once more.",
            "Please disconnect fully, reopen the VPN client, and try the connection again.",
        ],
        first_try_user_fail=[
            "I tried that and restarted my laptop, but it still fails.",
            "I signed out and back in and the VPN still will not connect.",
        ],
        diagnosis_agent=[
            "I reviewed the remote access configuration and found the VPN profile on your machine is expired.",
            "I checked the VPN configuration and your client is still using an outdated profile.",
        ],
        fix_agent=[
            "I sent the updated VPN profile and had you reconnect with it. The VPN connected successfully after that.",
            "Once the VPN client profile was updated, the connection completed successfully.",
        ],
        resolution_summary=[
            "The VPN client was using an outdated or expired profile. Updating the VPN configuration restored remote access.",
            "The connection failure was caused by an outdated VPN profile. Updating the client configuration restored VPN access.",
        ],
        environment_details=["This started while working from home this morning.", "I need access to internal finance systems."],
        signals=["The sign-in succeeds, but the tunnel never comes up.", "The same credentials work on the web portal."],
        tags=["network", "remote_access"],
    ),
    issue_variant(
        family="wifi_connectivity",
        root_cause_id="expired_wireless_certificate",
        titles=["Office Wi-Fi keeps dropping", "Cannot join secure wireless", "Wireless certificate error"],
        user_openers=[
            "My laptop keeps disconnecting from the office Wi-Fi.",
            "I cannot join the secure office wireless network anymore.",
            "The SSID is there, but I get a certificate or trust error when I connect.",
        ],
        clarify_agent=[
            "Is this happening on the corporate secure SSID or the guest network?",
            "Do you see a password prompt, certificate warning, or just repeated disconnects?",
        ],
        clarify_user=[
            "It is the secure staff Wi-Fi and it drops every few minutes.",
            "I am seeing a certificate warning when I try to join the secure SSID.",
        ],
        first_try_agent=[
            "Please forget the network and reconnect to it once.",
            "Please disable and re-enable Wi-Fi, then try joining the secure SSID again.",
        ],
        first_try_user_fail=[
            "I forgot the network and it still gives me the same warning.",
            "That did not help. It reconnects briefly and then drops again.",
        ],
        diagnosis_agent=[
            "I checked the device policy and found the wireless certificate on the laptop had expired.",
            "I reviewed the endpoint configuration and the machine certificate used for secure Wi-Fi is no longer valid.",
        ],
        fix_agent=[
            "I pushed a new wireless certificate and had you reconnect. The secure Wi-Fi stayed connected after that.",
            "After renewing the device certificate and reapplying the Wi-Fi profile, the connection stabilized.",
        ],
        resolution_summary=[
            "Secure Wi-Fi authentication was failing because the endpoint certificate had expired. Renewing the certificate restored wireless access.",
            "The laptop was using an expired machine certificate for the secure SSID. Updating the certificate and Wi-Fi profile resolved the issue.",
        ],
        environment_details=["This only seems to happen in the office.", "The guest Wi-Fi works, but the secure one does not."],
        signals=["My phone can join guest Wi-Fi fine.", "The error mentions trust or certificate validation."],
        tags=["network", "wifi"],
    ),
    issue_variant(
        family="internet_access",
        root_cause_id="proxy_misconfiguration",
        titles=["Cannot browse websites", "Internet access blocked", "Proxy error in browser"],
        user_openers=[
            "I cannot browse normal websites from my work laptop.",
            "Every site is failing to load and I keep seeing a proxy error.",
            "The internet looks up, but my browser cannot reach most pages.",
        ],
        clarify_agent=[
            "Are you able to reach any internal websites, or is everything failing?",
            "What exact error do you see in the browser when a page fails?",
        ],
        clarify_user=[
            "Internal sites are okay, but external websites fail with a proxy error.",
            "The browser says it cannot connect to the proxy server.",
        ],
        first_try_agent=[
            "Please close the browser, reconnect to the network, and test one site again.",
            "Please try another browser and let me know if the same proxy error appears.",
        ],
        first_try_user_fail=[
            "I tried another browser and it still fails the same way.",
            "I reconnected to the network and still cannot open outside websites.",
        ],
        diagnosis_agent=[
            "I checked the device network settings and found the proxy configuration is pointing to an old address.",
            "I reviewed the browser and system proxy settings and they are misconfigured.",
        ],
        fix_agent=[
            "I corrected the proxy settings and external browsing works again now.",
            "After resetting the old proxy configuration and reapplying the managed settings, internet access returned.",
        ],
        resolution_summary=[
            "External internet access was blocked by an outdated or incorrect proxy configuration. Resetting the managed proxy settings restored browsing.",
            "The browser issue was caused by stale proxy settings on the endpoint. Reapplying the correct network configuration resolved the problem.",
        ],
        environment_details=["This started right after docking in the office.", "Teams still works, but browser traffic does not."],
        signals=["Internal portals still open.", "The error specifically mentions a proxy server."],
        tags=["network", "proxy", "browser"],
    ),
    issue_variant(
        family="voip_telephony",
        root_cause_id="stale_softphone_registration",
        titles=["Desk phone not registering", "Softphone cannot place calls", "Meeting room audio problem"],
        user_openers=[
            "My desk phone shows unregistered and I cannot place calls.",
            "The softphone client signs in, but audio and calls do not connect.",
            "The meeting room phone system is online, but nobody can hear us.",
        ],
        clarify_agent=[
            "Is this a physical phone, a softphone client, or a conference room system?",
            "Are inbound calls failing, outbound calls failing, or is the issue only with audio?",
        ],
        clarify_user=[
            "It is the softphone client and outbound calls fail right away.",
            "This is my desk phone and it says unregistered on the display.",
        ],
        first_try_agent=[
            "Please sign out of the phone client or reboot the desk phone once and test again.",
            "Please reboot the device and then try one test call.",
        ],
        first_try_user_fail=[
            "I rebooted it and the phone still shows unregistered.",
            "I signed out and back in, but calls still do not connect.",
        ],
        diagnosis_agent=[
            "I checked the telephony platform and found the device registration had gone stale.",
            "I reviewed the phone provisioning record and the endpoint is not properly registered with the call manager.",
        ],
        fix_agent=[
            "I re-registered the device on the call manager and test calls are working again.",
            "After refreshing the phone provisioning record, the client registered successfully and calls connected.",
        ],
        resolution_summary=[
            "The voice endpoint had a stale registration in the telephony platform. Refreshing the registration restored calling.",
            "Phone service was failing because the device registration had expired or become stale. Re-registering the endpoint resolved it.",
        ],
        environment_details=["This affects my main business line.", "It started after the phone client updated."],
        signals=["Calls fail immediately without ringing.", "The display shows unregistered."],
        tags=["network", "telephony", "voice"],
    ),
    issue_variant(
        family="workstation_failure",
        root_cause_id="device_driver_crash",
        titles=["Laptop blue screen", "Computer crashes on startup", "Workstation keeps rebooting"],
        user_openers=[
            "My work laptop blue-screened and now it keeps rebooting.",
            "The workstation crashes shortly after I sign in.",
            "My computer powers on, but then fails with a blue screen.",
        ],
        clarify_agent=[
            "Does the crash happen before sign-in, after sign-in, or while opening a certain app?",
            "Are you seeing any stop code on the blue screen?",
        ],
        clarify_user=[
            "It usually crashes a minute or two after I sign in.",
            "I saw a stop code briefly, but the system rebooted before I could capture it.",
        ],
        first_try_agent=[
            "Please try booting once more and leave it at the desktop without opening anything.",
            "Please disconnect the dock and external devices, then boot the laptop again.",
        ],
        first_try_user_fail=[
            "It still crashed even with nothing open.",
            "I disconnected everything and it still blue-screened.",
        ],
        diagnosis_agent=[
            "I reviewed the crash data and it points to a faulty driver update on the device.",
            "I checked the endpoint logs and the recent driver package is causing system instability.",
        ],
        fix_agent=[
            "I rolled back the problematic driver and the workstation remained stable after reboot.",
            "After removing the bad driver package and applying the approved version, the device stopped crashing.",
        ],
        resolution_summary=[
            "The workstation crashes were caused by a faulty driver update. Rolling back to the approved driver restored stability.",
            "System instability came from a recent driver package on the endpoint. Reverting that driver resolved the blue-screen failures.",
        ],
        environment_details=["The issue started after the last reboot prompt.", "It happens even before I open most apps."],
        signals=["No external monitor is required to trigger it.", "The laptop restarts almost immediately after the crash."],
        tags=["hardware", "endpoint", "bsod"],
    ),
])

