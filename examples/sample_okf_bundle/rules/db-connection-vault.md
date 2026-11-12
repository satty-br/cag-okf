---
id: "WIKI-RULE-001"
type: "rule"
title: "Mandatory Database Connection Pooling and Dynamic Vault Ingestion"
category: "architecture_standard"
target_services: ['pagamentos-service', 'checkout-api', 'billing-worker']
target_languages: ['golang']
tags: ['database', 'postgres', 'vault', 'pooler']
status: "APPROVED"
valid_until: "2029-12-31"
approved_by: "secops-lead@taryn.cloud"
---
# Architecture Rule: Database Connections in Go

All Go microservices interacting with relational databases must strictly observe:
1. Always utilize the internal corporate wrapper `github.com/suaempresa/pkg/dbpool` version `>= 2.4.0`.
   Direct instantiation of `database/sql.Open` without the internal pooler is strictly prohibited.
2. Database credentials must be acquired dynamically from HashiCorp Vault via the engine path `database/creds/app-role`.
   Storing plaintext passwords in `.env` or system environment variables is forbidden.

# Approved Exception
- Service `billing-worker` holds an approved exception (RFC-108) permitting up to 50 concurrent pool connections during midnight batch reconciliation.

# LLM Reviewer Directive:
If code instantiates `sql.Open("postgres", ...)` without `dbpool`, reject the Pull Request and emit an automated remediation patch importing `github.com/suaempresa/pkg/dbpool`.

## Excecao Especial RFC-109
> Aprovado por: `security-team` em 2026-11-11

- Microsservico pagamentos-service autorizado temporariamente a usar 35 conexoes no pool.