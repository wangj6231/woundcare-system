# D-Seg-07 Docker Runtime Verification

- Verification status: `PASS`
- Docker image: `heartexlabs/label-studio:1.23.0@sha256:20cec817e63144adec9f23d699bf4f33ce4249a56eb183ae4853d20fbd10fd93`
- Container: `dseg07-label-studio`
- Host binding: `127.0.0.1:8083`
- Container health: `healthy`
- Login page: HTTP 200
- Unauthenticated local-image request: HTTP 401 (expected access control)
- Images visible inside container: 383
- `/label-studio/files`: read-only
- `/label-studio/data`: writable persistent bind mount
- `/label-studio/exports`: writable persistent bind mount
- Test images used: 0
- Blind test used: false
- External test status: `LOCKED_NOT_ACCESSED`

The delivery ZIP intentionally excludes the initialized `label_studio_data` contents. It therefore contains no local user account, password, session, or in-progress annotation database.
