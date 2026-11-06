# Acme Corp IT Security Policy (version 4.2)

## 1. Passwords and authentication

Employees must rotate passwords every 90 days. Passwords must be at least 14 characters long and may not reuse any of the previous 12 passwords. Multi-factor authentication (MFA) is required for all remote access and for all administrator accounts. Contractors may request an MFA exception from the Security Office, but an exception expires after 30 days.

## 2. Data protection

Customer data must be encrypted at rest with AES-256 and in transit with TLS 1.3. Backups are encrypted with AES-256, and restore tests are performed every quarter. Audit logs are retained for seven years. Company laptops must use full-disk encryption, and personal USB storage devices are prohibited on company laptops.

## 3. Incident response

All security incidents must be reported to the Security Office within 24 hours of discovery. The Security Office classifies each incident by severity and responds within the target times shown below.

| Severity | Description | Target response time |
| --- | --- | --- |
| P1 | Active data breach or outage of a critical system | 15 minutes |
| P2 | Suspected breach or major degradation of service | 1 hour |
| P3 | Contained policy violation | 8 hours |
| P4 | Low-risk request or question | 3 days |

## 4. Third-party vendors

Vendors with access to customer data must complete a security review before signing a contract and annually after that. Vendor access is revoked within 24 hours of contract termination. Vendors may not subcontract work involving customer data without written approval from the Security Office.
