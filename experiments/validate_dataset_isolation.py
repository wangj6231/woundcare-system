"""Pure manifest audit; never opens images or starts a development runner.

Callers must pin source/identity manifests and supply truthful metadata. This
does NOT retrofit legacy runners or prove isolation from an unavailable sealed
hash inventory. Unknown patient/family/source identities remain explicit.
"""
from collections import defaultdict
from pathlib import Path
import re


def _token(value):
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def validate_dataset_isolation(rows, *, allowed_roots, approved_sources=(),
                               protected_hashes=None, source_held_out=False):
    """Return failures for known overlap and restricted roles, not a blanket PASS.

    ``protected_hashes`` is a trusted sealed metadata set, not image paths. None
    means protection against renamed/copied test pixels was NOT verified.
    pHash groups must be precomputed from a frozen similarity protocol; raw pHash
    distance is not silently treated as an equivalence relation here.
    """
    roots = [Path(p).resolve() for p in allowed_roots]
    approved = {_token(s) for s in approved_sources}
    fields = ["sha256", "md5", "perceptual_group", "frame_family_id", "derived_family_id",
              "patient_id", "source_case_id", "video_id"]
    if source_held_out:
        fields.append("source_dataset")
    known, violations, ids = {f: defaultdict(set) for f in fields}, [], set()
    for index, row in enumerate(rows):
        def issue(code, **extra):
            violations.append({"row": index, "sample_id": row.get("sample_id"), "code": code, **extra})
        ident = row.get("sample_id")
        if not ident or ident in ids:
            issue("MISSING_OR_DUPLICATE_SAMPLE_ID")
        ids.add(ident)
        role = row.get("usage_role")
        if role not in {"train", "val"}:
            issue("NON_DEVELOPMENT_ROLE")
        if not any(row.get(f) and _token(row[f]) not in {"unknown", "unverified"} for f in ("sha256", "md5")):
            issue("MISSING_EXACT_CONTENT_IDENTITY")
        source = _token(row.get("source_dataset", ""))
        if "co2wound" in source or _token(row.get("source_role", "")) == "historicalexternalbenchmark":
            issue("HISTORICAL_EXTERNAL_SOURCE")
        if not source or source not in approved:
            issue("SOURCE_NOT_ADMITTED")
        raw_path = row.get("path") or row.get("image_path_relative")
        if not raw_path:
            issue("MISSING_PATH")
        else:
            p = Path(raw_path).resolve()
            raw_parts = re.split(r"[\\/]", str(raw_path))
            tokens = {_token(x) for x in (*p.parts, *raw_parts)}
            if any("co2wound" in t for t in tokens):
                issue("HISTORICAL_EXTERNAL_PATH")
            if tokens & {"test", "testing", "blindtest", "lockedtest", "holdout"}:
                issue("LOCKED_TEST_PATH")
            if not any(p == r or r in p.parents for r in roots):
                issue("OUTSIDE_DEVELOPMENT_ALLOWLIST")
        for field in fields:
            value = row.get(field)
            if value and _token(value) not in {"unknown", "unverified"}:
                known[field][str(value)].add(role)
        if protected_hashes is not None and any(row.get(f) in protected_hashes for f in ("md5", "sha256")):
            issue("PROTECTED_CONTENT_HASH")
    for field, groups in known.items():
        for value, roles in groups.items():
            if len(roles) > 1:
                violations.append({"code": "CROSS_SPLIT_IDENTITY", "field": field, "identity": value,
                                   "roles": sorted(str(x) for x in roles)})
    missing = {f: sum(not row.get(f) or _token(row[f]) in {"unknown", "unverified"} for row in rows) for f in fields}
    return {"passed": bool(rows) and not violations, "violations": violations, "rows": len(rows),
            "missing_identity_fields": missing,
            "patient_verdict": ("PATIENT_LEVEL_INDEPENDENCE_NOT_VERIFIED" if missing["patient_id"]
                                else "KNOWN_PATIENT_IDS_CHECKED"),
            "locked_content_verdict": ("NOT_VERIFIED_NO_TRUSTED_HASH_INVENTORY" if protected_hashes is None
                                       else "CHECKED_AGAINST_SUPPLIED_HASH_INVENTORY"),
            "scope": "Manifest/path and known-identity audit only; not global enforcement or provenance verification."}
