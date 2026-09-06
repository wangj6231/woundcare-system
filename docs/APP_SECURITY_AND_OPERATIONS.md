# WoundCare+ App Security and Operations

## Product boundary

WoundCare+ is a clinical decision-support tool. The detection → classification
result is advisory only. A nurse must select the patient, review the image and
model output, write the treatment observation, and explicitly save the EMR.

## Implemented controls

- Bearer sessions are stored as SHA-256 token hashes in SQLite, with an expiry,
  revocation timestamp, and last-seen timestamp. Tokens are kept in browser
  `sessionStorage`, not local storage.
- Role-based access control: `nurse` can assess assigned records; `head_nurse`
  can review the team and audit trail; `admin` can manage accounts and model
  loading.
- Nurse access is enforced against each patient record's `assigned_nurse`; an
  unassigned record is visible only to a head nurse or administrator.
- Passwords use bcrypt. New deployments do not create a known default account;
  set `WOUNDCARE_BOOTSTRAP_PASSWORD` once before first login, then remove it.
- Patient fields, wound images, AI text, and treatment notes are encrypted with
  Fernet at rest. The key is supplied by `WOUNDCARE_FERNET_KEY` in production.
- Every login, AI prediction, EMR write, user change, and model load is written
  to `audit_log`.
- A record can only be marked `reviewed` when the user explicitly confirms the
  human review step. Low-confidence and fallback predictions remain visibly
  flagged in the professor preview UI.
- `WOUNDCARE_PROFESSOR_PREVIEW=true` seeds only two synthetic demonstration
  cases in a separate database and enables the invited-review feedback flow.
- Head-nurse-approved, de-identified care recommendations are stored in
  `rag_guidance`. Retrieval is lexical and provenance-aware; it augments the
  nurse review screen but never changes model weights or automatically writes
  an EMR. Each recommendation can be deactivated and carries its evidence
  level, source, author, and timestamps.
- CORS is allow-listed, credentials are disabled, and security headers include
  CSP, `X-Frame-Options`, `X-Content-Type-Options`, and a restrictive
  `Permissions-Policy`.
- Uploads are restricted to JPG/PNG/WebP and 10 MB. Model loading is restricted
  to approved local model directories and `.pt` files.
- SQLite uses foreign keys, WAL mode, a busy timeout, parameterized queries,
  and transaction rollback on failure.
- The server binds to `127.0.0.1` by default. Public tunnelling is intentionally
  not started by the application.

## Deployment checklist

1. Generate a Fernet key with `python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"`.
2. Set `WOUNDCARE_ENV=production`, a strong bootstrap password, the Fernet key,
   exact allowed origins, and exact allowed hosts.
3. Start the API behind TLS and a reverse proxy; do not expose Uvicorn directly.
4. Change the bootstrap admin password immediately and remove the bootstrap
   environment variable.
5. Back up `healthcare.db` and the Fernet key together. Losing the key makes
   encrypted clinical fields unrecoverable.
6. Review `/api/audit` during pilot operation and define a retention policy with
   the hospital/IRB owner before production use.
7. For an external professor review, use the separate preview database and the
   deployment procedure in `docs/PROFESSOR_PREVIEW_DEPLOYMENT.md`; never reuse
   a clinical database, development key, or sealed test images.
