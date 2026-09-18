# Completed cold authentication: finite fixture admission

Scope: existing Task5 completed authentication/reuse, not full gate, provider,
recovery or production training. Read-only bound agents: task5_checkout_bound
and task5_training_bound. No numerical work was used to derive these bounds.

## Categories and common authority

Existing operational ledger only. Two new immutable source fixtures (RED/GREEN)
with one genuine four-update phase4_smoke fit each; subsequent controller check
rereads GREEN using its own authenticated source and does not train again.
Scratch/source administration, scientific spool/backing, cold single-file cache,
metadata and logs use their already-bound roots from inception. Old outputs and
retained6195077120B remain charged; no new baseline, quota or nested allowance.

Proposed aggregate increments: spool128MiB, cache16MiB, scratch32MiB,
metadata33562624B, logs2MiB, pinned/emergency0. These are capacity reservations,
not authorization to execute until guard coverage and exact command are checked.

## Checkout proof

Per-checkout ceiling12MiB. Static inventory at review:177regular dense single-link
files/24directories; logical3955245B, rounded4349952B. Derived peak10776576B:

| Component | Bytes |
| --- | ---: |
| Source worktree, rounded | 4349952 |
| Loose Git objects, zlib worst-case rounded | 4853760 |
| Sixteen manifest/audit files | 311296 |
| One24KiB publisher temp hardlink | 24576 |
| Git index/locks/refs/config | 131072 |
| 270directories,4KiB each | 1105920 |

Recompute against actual source before the first write; require4KiB allocation
unit and <=12MiB. Hash/lstat source inventory and reject drift, links, sparse or
special files. Git: empty local template, fixed SHA1/main/identity, disabled
hooks/signing/autoGC/system/global config. Maximum227objects:177initialblobs,
24trees+commit,16outputblobs+7trees+2commits. Maximum tree body1708B; final193path
v2index raw20048B, allowance32KiB plus32KiBlock plus64KiB other metadata.
Data publishers use the real unchanged writer under a fixture-owned prewrite
guard:8manifestcopies<=24KiB each,4debugreports<=24KiB,4indexes<=256B, exactpaths.
All partial bytes retained on any guard failure. Per-command additional test,
temp and coordination allowance2MiB; two commands total28MiB within32MiBgrant.

## Scientific output proof and explicit limitation

Model771090float32parameters in71tensors plus18boolbufferbytes: weight body
3084378B, AdamW body6169004B, trained checkpoint body9253382B before RNG/header.
CPU execution can also capture MPS RNG; natural format limits are much larger
than observed files. The current serializers therefore DO NOT establish an
unconditional64MiB maximum for every possible smoke result.

The fixture must instead enforce64MiB live allocation before every real write,
with16MiB per-input ceiling to bound a cold lease,256live file names, bounded
directories, actual block-rounded prospective bytes and full temporary/copy/
replacement coexistence. Checkpoints are written pending and copied to final;
the pending input stays charged. Keep normal scientific checkpoint retention,
but never manually remove data to fit. A guard rejection FAILS the fixture and
preserves its prefix; it never changes production caps or fabricates completion.

Closed counts:4updates/updatejournals,2validations/validationjournals,32episodes,
2componentreports,2validationrecords,3checkpoints/6checkpointpayloadwrites,
3checkpointindexreplacements,3exported+2runtimeweights,32each neural sidecars,
trajectories and telemetry,at most32crashpairs+2crashindexes,1indexshard and
1attemptresult. Smoke cannot promote beyond one_hop; no robustness extra run.
Static normal names<=195, base directories10; temporary/failure/root-result and
lease/backing names require explicit extra allowance inside the same cap.

Guard coverage must include all imported pilot-byte-publisher aliases including
pilot_evidence_types, trainer's descriptor publisher, atomic publisher common
leaf, and direct .rows.pending.jsonl handle.write appends. Bound original and
partial error output too. Do not call parent ledger during atomic transient
hardlinks. Cold rename transfers allocation between admitted categories; it does
not duplicate payload. Small damaged controls and outer logs have fixed caps.

The implementer must report exact coverage/command before RED admission. Exact
original/current-source identity, real decoders and every semantic comparison
remain unchanged. Independent review and final same-ledger check are required.
