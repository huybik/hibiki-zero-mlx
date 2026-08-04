# Resume-validation corrective action

The first post-run clean validation exposed a runner-only bug: the batch record stores its plan attestation under `plan.sha256`, while the resume validator looked for a nonexistent top-level `plan_sha256`. Generation results and hashes were unaffected, but an interrupted run would reject a completed atomic batch instead of resuming it.

The owning validator was corrected to read the actual schema boundary. The initially executed script is preserved at commit `8b7cfae` and SHA-256 `8191098997b93e7b2e86a8c178cdfbe898c934d80fafcfec6dec516a1e57f28b`; plans created by that exact revision remain accepted. A clean validation then checked all 120 benchmark batch directories and all output WAV hashes, plus exact coverage of the 10,950-row production plan.
