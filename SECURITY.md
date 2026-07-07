# Security Policy

## Supported versions

This project is maintained on the `main` branch, which always carries the latest security patched dependencies. The most recent release receives security updates. Older snapshots are not separately maintained.

## Reporting a vulnerability

Please do not open a public issue for security problems.

Report a vulnerability privately through GitHub: open the repository Security tab and choose "Report a vulnerability". This creates a private advisory visible only to the maintainer.

You can expect an initial response within a few days. If the report is confirmed, a fix will be prepared and released, and you will be credited in the advisory unless you prefer otherwise. If it is declined, you will receive a short explanation.

## Scope

This policy covers the application code in this repository. Dependency vulnerabilities are tracked with pip-audit in CI and patched by updating the pinned versions. You are welcome to flag any that slip through.

## Known advisories

- **chromadb (PYSEC-2026-311, GHSA-f4j7-r4q5-qw2c, critical):** a pre authentication code injection in the ChromaDB HTTP **server**, present only in server versions 1.0.0 and later, reachable when a request sets `trust_remote_code` true against the collections endpoint. There is no patched release. This project is not exposed on either deployment path: the default and the demo run Chroma in embedded mode with no server, and the optional Docker deployment pins the server image to `chromadb/chroma:0.6.3`, which is below the affected range. The advisory is scoped to the CI scan on the Python client package, for which no patch exists and which is not the vulnerable surface, so it is listed as an accepted, documented exception. The pin will move to a patched release as soon as one ships.
