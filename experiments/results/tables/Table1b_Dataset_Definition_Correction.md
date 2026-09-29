# Table 1b — Classification dataset terminology correction (2026-09-20)

| Quantity | Count | Meaning / evidence level |
|---|---:|---|
| Raw source images | 431 | Historical documentation only; original source pool was not reopened or independently audited in Phase A.5. Publisher, URL, version and license remain unverified. |
| Prepared/balanced image files | 768 | Derived dataset count reported by historical project documents; **not** 768 raw-source images. |
| Development files | 720 | Current train 679 + val 41; current development snapshot was audited in Phase A. |
| Internal locked test files | 48 | Historical split record only in this phase; no test directory enumeration or pixel access. |
| Development unique MD5 content groups | 383 | Current development snapshot: 177 singleton + 206 duplicate-content groups. |
| Redundant development instances | 337 | 543 files within duplicate groups − 206 group representatives. |

This correction supplements, and does not overwrite, historical Table 1. The 383 content groups are not patients; no patient-independence claim follows from MD5 grouping. Phase A.5 performed no new training, inference or test access.
