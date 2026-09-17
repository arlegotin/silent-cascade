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
because this bucket exists. An explicitly user-approved, phase-scoped Superpowers
implementation plan or amendment is required before source/configuration changes.
It must cover the following before the production pilot can use R2:

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

The proposed [storage design](superpowers/specs/2026-09-17-phase-4-r2-archive-design.md)
and [Phase 4 implementation-plan amendment](superpowers/plans/2026-09-17-phase-4-r2-archive.md)
now specify the producer, journal, lease, recovery and verification changes. They
remain pending explicit plan approval; the commands and limits in that plan are
implementation requirements, not already available archive functionality.
