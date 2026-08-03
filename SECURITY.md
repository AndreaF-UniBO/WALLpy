# Security policy

## Reporting a vulnerability

Please report suspected vulnerabilities privately to Andrea Fiorini at `andrea.fiorini6@unibo.it`. Do not open a public issue that includes credentials, personal data, or an exploit that has not yet been mitigated.

## Data handling

PyWALL processes images locally and does not upload them. The application does not download model weights implicitly. The supplied checkpoint script performs an explicit HTTPS download from Meta and rejects a file whose SHA-256 checksum does not match the documented value. Users are responsible for verifying that archaeological imagery, datasets, and derived outputs may lawfully be processed and shared.

Credentials, `.env` files, model weights, datasets, generated results, and local databases must not be committed to the repository.
