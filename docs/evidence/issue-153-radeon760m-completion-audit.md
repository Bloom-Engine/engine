# Issue #153 Windows completion audit

Scope: the user replaced the unavailable RTX 4080 machine with this Windows Radeon 760M host. The requested qualification accepts strict failures when complete evidence is returned. It does not require silently making every image pass, weakening thresholds, or claiming RTX 4080 certification.

| Requirement | Authoritative evidence | State |
| --- | --- | --- |
| Start from requested renderer revision | Branch ancestry includes `4508eaff082b849203ac60f9d8f3a327b71d95c9`; source.patch records Windows corrections | Verified, with necessary execution fixes |
| Clean measured source | Both final result.json files name `8309985bf9f5358c38e5dd2cbc0e328d2bccb6f8`, `git_dirty=false` | Verified |
| Complete pinned assets and approved baselines | Asset check PASS; original hashes unchanged; Bistro revision `7c9f9f9ac0915024ccf3dddbccd8bfc643a42607` | Verified |
| Strict full nine-case run without report-only | Two full result.json files, all nine IDs, `report_only=false`, exit code 1 each | Verified; 7 pass, 2 visual failures |
| Correct available host identity | Native telemetry: Windows, Vulkan, AMD Radeon 760M Graphics, driver 24.12.1 | Verified for user-approved substitute hardware |
| Hardware ray-query in required cases | Sponza/Bistro native `ssgi_trace_backend=hw-ray-query` | Verified |
| Correct pixel dimensions and measurement evidence | Final PNG dimensions match every manifest case; telemetry and intermediate captures exist for all nine | Verified |
| Host quietness records | Both preflights and all 18 postflights accepted | Verified CPU snapshots; continuous background GPU/process-start monitoring not claimed |
| Preserve baselines, thresholds and noise bounds | Git diff against pinned source changes none; full result records strict visual failures | Verified |
| Repeatability | repro-check PASS, stable metadata equal, 257/257 artifacts byte-identical | Verified |
| Seeded regressions remain detectable | Five negative controls DETECTED, exit 0 | Verified |
| Diagnose remaining strict failures | Original baseline-source renderer produces byte-identical Sponza/skinned-alpha images on this host; controls preserved | Verified that failures predate intervening source changes under this configuration; precise portability cause unresolved |
| Runnable final local state | Temporary anisotropy reduction reverted; affected executables rebuilt; restored captures byte-identical to final qualification | Verified |
| Complete downloadable evidence | Versioned ZIP, CRC check, SHA-256 sidecar; report and source patch included | Published as a GitHub prerelease; uploaded size and SHA-256 verified against GitHub asset metadata |
| GitHub review and issue handoff | Draft PR #154 and reports posted on #153 and #128 | Published with explicit user authorization; PR draft state, integration base, and published bodies verified through the GitHub API |

The original RTX 4080 certification remains unperformed; this hardware substitution is documented rather than represented as that certification. Radeon performance numbers are measured, with no invented hard budget. The evidence permits renderer work to proceed on this Windows machine.

Published on 2026-09-10: [draft PR #154](https://github.com/Bloom-Engine/engine/pull/154), [evidence prerelease](https://github.com/Bloom-Engine/engine/releases/tag/quality-evidence-153-radeon760m-20260910), [#153 report](https://github.com/Bloom-Engine/engine/issues/153#issuecomment-5622704289), and [#128 report](https://github.com/Bloom-Engine/engine/issues/128#issuecomment-5622704740).

The uploaded v2 ZIP is 267,221,734 bytes with SHA-256 `aa099676ecd801ebb7ed586ddb9f9732b8f913f13a6d926604ad193b078d82a4`. Its source snapshot and release tag remain pinned to `5dc8277e7f6b0ca69908e8b34c33f99cdbe15ba0`; this publication audit was updated afterward. The archive therefore preserves the audit as it stood before publication.
