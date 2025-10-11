# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability, please report it responsibly.

**Do NOT open a public GitHub issue for security vulnerabilities.**

Instead, please use [GitHub Security Advisories](https://github.com/vadimgodev/face-recognition/security/advisories/new) to privately report vulnerabilities. This is GitHub's built-in private vulnerability reporting feature.

When reporting, please include:
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

We will acknowledge receipt within 48 hours and aim to provide a fix within 7 days for critical issues.

## Scope

Security concerns for this project include:

- **Authentication bypass** -- Circumventing API token or Basic Auth layers
- **Biometric data exposure** -- Unauthorized access to face embeddings or images
- **Injection attacks** -- SQL injection, command injection, or path traversal
- **Anti-spoofing bypass** -- Defeating liveness detection with adversarial inputs
- **Credential exposure** -- Secrets leaked in logs, responses, or error messages

## Biometric Data Handling

This software processes biometric data (face images and embeddings). Deployers should be aware of:

- Face embeddings stored in PostgreSQL can be used to identify individuals
- Images stored locally or in S3 contain personally identifiable information
- Applicable regulations may include GDPR (EU), BIPA (Illinois), CCPA (California), and others
- Deployers are responsible for obtaining appropriate consent and implementing data retention policies
- The anti-spoofing system uses passive detection only (single image analysis) and should not be the sole security measure for high-security applications

## Security Best Practices for Deployment

- Change all default credentials (`SECRET_KEY`, database passwords, Basic Auth)
- Use HTTPS in production (Traefik with Let's Encrypt is pre-configured)
- Restrict `ALLOWED_ORIGINS` to your actual domain(s)
- Set `APP_ENV=production` and `DEBUG=false`
- Use strong, randomly generated API tokens (32+ characters)
- Enable liveness detection for enrollment (`LIVENESS_ON_ENROLLMENT=true`)
- Regularly rotate credentials and API tokens
- Monitor access logs for suspicious activity
- Back up face data and encryption keys securely

## Supported Versions

| Version | Supported |
|---------|-----------|
| Latest  | Yes       |
