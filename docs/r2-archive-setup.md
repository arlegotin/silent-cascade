# R2 archive setup

## Provisioning status — 2026-09-17

The repository owner authorized Cloudflare R2 setup through their authenticated
Chrome session. The following infrastructure is provisioned and locally tested;
this is **not** a claim that the pilot supports remote artifact storage yet.

| Setting | Verified value |
| --- | --- |
| Bucket | `silent-cascade-artifacts` |
| Location | Western Europe; default jurisdiction |
| Storage class | Standard |
| Public access | Disabled |
| Local AWS CLI profile | `silent-cascade-r2` |
| Credential scope | Object Read & Write on this bucket only |
| Credential expiry | March 17, 2027 |

No existing bucket was modified. No experiment artifacts were uploaded, moved,
deleted or pruned. No training, hosted computation or CI/CD was started.

## Credentials and local access

The private credential file is outside the repository at
`~/.config/silent-cascade/r2-credentials.json`, with mode `0600` inside a `0700`
directory. The dedicated profile in `~/.aws/config` uses `credential_process` to
read that file and sets the account-specific HTTPS R2 endpoint and region `auto`.
The account endpoint and all credential values stay in local configuration, not
in this public repository. Do not print the credential file, run the credential
process directly, include it in archives, or enable credential-bearing debug logs.

Secret export was encrypted in the browser and decrypted directly into the
private local file. Plaintext keys were not emitted into chat or Git. The
credential is not a global Cloudflare key and cannot administer buckets. Expiry
or revocation must cause an uploader to stop safely, never discard pending data.

These read-only commands were verified on the configured Mac:

```sh
aws --profile silent-cascade-r2 --no-cli-pager s3api head-bucket --bucket silent-cascade-artifacts
aws --profile silent-cascade-r2 --no-cli-pager s3api list-objects-v2 --bucket silent-cascade-artifacts --max-keys 1 --no-paginate --query '{KeyCount: KeyCount, IsTruncated: IsTruncated}'
```

The listing is limited to one service page and prints no object names; see the
[AWS CLI pagination documentation](https://docs.aws.amazon.com/cli/latest/reference/s3api/list-objects-v2.html).
These are storage-administration commands, not offline scientific commands. Do
not make primary training or evaluation initialize this profile or a cloud SDK.
Other machines require their own authorized local credential configuration.

## Connectivity verification

Verification ran against software checkout
`61d26c58d68a843f598729df7546c89cdefabec0` without source changes:

- A new 50-byte, non-scientific text probe was uploaded with `If-None-Match: *`.
- Downloaded bytes matched the original using `cmp` and SHA-256:
  `aff28149582b38ba2890abb8f45eaa2ad10f61aca501340ac6368c7e20a34c91`.
- The single test object was deleted and listing returned `KeyCount: 0`.
- `HeadBucket` succeeded for the project bucket and was denied for the
  account's existing unrelated bucket; no objects in that bucket were accessed.

This is provisioning/connectivity evidence only. It is not a bandwidth benchmark,
a complete archive durability test, a tested overwrite-conflict policy, or a
Phase 4 acceptance artifact. No spending cap or automatic deletion policy was
configured. Standard storage and request charges follow
[Cloudflare's R2 pricing](https://developers.cloudflare.com/r2/pricing/).

## Remaining Phase 4 storage work

The trainer, verifier and replay readers still assume locally available files.
The historical 4.41 TB preflight remains unchanged and must not be bypassed merely
because this bucket exists. The owner approved the phase-scoped implementation
plan below; its remaining integration and qualification work must establish the
following before the production pilot can use R2:

1. A bounded local spool and a separate, explicit network-enabled archiver;
   primary training/evaluation remain offline with zero foundation-model calls.
2. Immutable artifact bundles, complete manifests, content hashes and verified
   remote readback before any permitted eviction of a local archived copy.
3. Safe backpressure and checkpoint/resume when disk headroom, credentials,
   network connectivity or an explicit archive budget prevent further progress.
4. Bounded rehydration for artifact verification and replay; preserve every
   required failure, adverse result and provenance record.
5. Measured local working-space requirements, transfer throughput, interrupted
   upload/recovery tests and local verification before changing the preflight.

R2 is an archive destination, not a mounted filesystem or a substitute for
retention, scientific gates, reproducible evidence or CPU replay.

The approved [storage design](superpowers/specs/2026-09-17-phase-4-r2-archive-design.md)
and [Phase 4 implementation-plan amendment](superpowers/plans/2026-09-17-phase-4-r2-archive.md)
now specify per-episode publication, streaming readers, paged journals/catalogs,
recovery and verification. Revision 2 replaces the rejected 64 GiB proposal with
the user's **10 GiB total local disk ceiling**, including protected headroom:
normal allocation <=6.5 GiB, emergency allocation <=8 GiB, protected headroom 2 GiB.
It does not require a whole evaluation to be local or an additional 16 GiB reserve.
The user approved the revised plan for autonomous execution on 2026-09-17. These
remain implementation and measurement requirements, not proven archive functionality.

## Implementation checkpoint — 2026-09-18

Tasks 1 and 2 passed their scoped implementation reviews through `de88d0b`:
immutable bounded bundles/catalogs, create-only transport, verified readback,
durable remote-byte reservations and conservative local eviction. Transport
tests used local doubles; the native macOS descriptor-safety test ran separately
because the managed sandbox denies the required descriptor reopen.

Task 3 passed its scoped reviews through `4d2e031`: finite supervisor, offline
child boundary, single-episode leases and shared local-allocation ledger.
The final correction passed 22 supervisor/offline covering tests; the preceding
53-test covering run checked ledger, supervisor, session and offline behavior.
These focused runs are not a complete local quality-gate pass. Offline provenance
requires trusted Git >=2.45 with native no-lazy-fetch support; the executable is
pinned privately before child launch. Older Apple Git is not a fallback.

Task 4 producer integration passed its scoped reviews through `7150b45`:
per-episode publication, shared crash-weight ownership, bounded journal segments,
control snapshots and stopped-writer handoff. These checks used local transport
doubles, not R2.

The first Task 5 milestone passed its scoped review through `045d234`: cold
evaluation scanning, reports, training envelopes and journal readers authenticate
one bounded unit at a time. Full Task 5 remains incomplete: cold partial/recovery
integration, broader caller coverage and replay/continuation checks are still due.
Two minor review findings remain tracked for that integration: duplicated metric
finalization rules and a stronger same-binding mutation regression.

The Task 6 engineering-accounting prelude is implemented through `0515d9c`.
Its latest focused run passed 65 checks; independent review closed the initial
bootstrap admission, repeated-scan, page-serialization and allocation-geometry
findings. These are scoped local checks, not provider qualification or a complete
quality-gate pass. Existing test files remain charged to the same 10 GiB ceiling,
and unsupported fixtures stay local.

The metadata-lifecycle correction passed independent review at `1a28272`, with
96 focused checks passing. Stable immutable leaves and a bounded authenticated
index now preserve earlier generations without rewriting every surviving record
after each small removal. The existing single-unit receipt/recovery boundary and
all storage ceilings are unchanged.

A subsequent read-only full-workspace snapshot measured 6,168,109,056 allocated
bytes. Its prospective bootstrap bound was 85,363,646 bytes, leaving 725,849,154
bytes below the normal ceiling at that instant. For an explicitly identified
731-file subset, conservative retained-inventory publication was about 501 MB
against 2.35 GB of potentially reclaimable payload. This removes the measured
negative-reclamation problem for that subset, but excludes other archival costs
and is not actual admission. Another 67 candidate files remain unadmitted.

The narrow early-provider qualification driver is being implemented before the
first real bootstrap. It will share the existing finite sizing calculations and
exercise the fixed probe, interrupted upload, create-only collision/readback and
exact restore under one local/remote history. Independent driver review, actual
shared-workspace admission and provider qualification still precede real
engineering archival and verified-before-eviction. No cleanup, larger quota or
scientific change is authorized by these engineering checks.

CLI/preflight integration and the remaining real R2 storage qualification are
pending. No experiment evidence has been uploaded or deleted during this
implementation. The new source has not yet passed a complete local `make verify`
run or the production storage gate.

This is implementation progress, not a Phase 4 result: the seed-11 production
pilot remains unstarted. The 10 GiB ceiling includes test output, temporary
files, operational metadata, recovery copies and protected headroom.
