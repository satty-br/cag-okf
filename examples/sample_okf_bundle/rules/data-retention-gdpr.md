---
id: "WIKI-RULE-003"
type: "rule"
title: "PII Masking and 90-Day Retention Policy for Payment Records"
category: "compliance"
target_services: ["pagamentos-service", "billing-worker"]
target_languages: ["golang", "python"]
approved_by: "dpo@taryn.cloud.corp"
approved_at: "2026-07-10"
valid_until: "2027-07-10"
status: "APPROVED"
tags: ["compliance", "lgpd", "gdpr", "pii", "encryption"]
---

# Compliance Rule: Customer Data Privacy & Retention

1. Payment card numbers (PAN) and national tax IDs (CPF/SSN) must never be logged in plaintext stdout or unstructured loggers.
2. Log messages containing customer metadata must pass through the `masker.RedactPII()` filter.
3. Raw database tables storing payment payloads must configure automated partition dropping after 90 days.
