---
id: "WIKI-PLAY-001"
type: "playbook"
title: "SecOps P1 Incident Response and Autonomous Isolation Playbook"
category: "incident_response"
target_services: ["all"]
target_languages: ["all"]
approved_by: "ciso@taryn.cloud.corp"
approved_at: "2026-06-01"
valid_until: "2028-06-01"
status: "APPROVED"
tags: ["secops", "incident", "isolation", "p1"]
---

# SecOps Incident Response Playbook: Severity P1

When active credential exfiltration or unauthorized root escalation is confirmed:
1. Immediately isolate affected workloads via NetworkSecurityGroup / iptables quarantine.
2. Invalidate Vault lease tokens linked to the breached service identity in under 60 seconds.
3. Capture memory dump and disk snapshot before terminating suspect container instances.
