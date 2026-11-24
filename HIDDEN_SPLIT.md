# Planned hidden v1 split

A future version can add a new set ID with separate public and hidden split
entries. The scorer registry already selects a versioned set and named
split. Public entries identify the benchmark manifest, its SHA-256, the
normalization and the metric protocol.

The hidden entry would expose the submission contract, permitted carrier
IDs, protocol and a scoring-service identifier. It would declare
`reference_access: server_only` and omit public reference paths. The local
CLI rejects that descriptor before fetching assets. The scoring service
would bind a separate immutable private manifest for hidden images,
references and masks, apply the same versioned metric code in an isolated
worker and return aggregate results. Require exact carrier coverage,
submission size limits and a private per-submission receipt keyed by
manifest and prediction digests.

Publish leaderboards from reviewed aggregate receipts with the set
version, split, model metadata and both report views. Limit feedback to
aggregates and bound repeated submissions to reduce adaptation to the
hidden set. Retire contaminated splits with a version change. Server,
credentials, submissions and leaderboard infrastructure are future work.
