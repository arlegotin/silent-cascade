# Numerical reader task review, round0

Reviewer task5_cold_numerics_review, base471d181, headf0b3c81.
Spec: issues found. Quality: needs fixes. No critical or minor findings.

## Important findings (paths normalized to repository-relative references)

- **Important — partial verification bypasses mandatory closure safety.** When
  any expected artifact is unavailable, `verify_numeric_evidence` skips
  `read_numeric_report` and enters the partial branch without scanning resident
  inventory (pilot_evidence.py:973,985). Consequently, extra files are ignored,
  while `_numeric_exists` follows resident symlinks and `_numeric_path` delegates
  to a reader that yields an existing resident path directly
  (pilot_verification.py:1155,1167; readers.py:21). The required symlink/extra
  rejection occurs only in the complete reader's closure check
  (pilot_verification.py:1181,1736). This violates the plan's requirements to
  reject escapes, extras, and symlinks. Add common resident-inventory and symlink
  validation before selecting complete versus partial verification, then add
  partial-plus-extra and partial-plus-symlink regressions.
- **Important — machine-specific path committed to Git.** The versioned report
  contains a machine-specific absolute prefix (task5-cold-numerics-report.md:70), contrary
  to the binding requirement that machine-specific paths remain outside Git.
  Replace it with a repository-relative or symbolic stage root; retain the
  absolute execution record only in controller-owned external evidence.

## Verified strengths and unresolved external evidence

The reviewer confirmed filtered exhaustive discovery, real runtime scanning,
sequential tensor/operations leases and genuine complete/partial equality tests.
No tests were rerun. Controller must establish frozen custody, allocation plus
headroom, zero provider/scientific execution and its fresh9test result from
the operational ledger. Those facts are not inferable from the diff alone.

## Controller technical check before fix

The partial extra-inventory gap is real. The reviewer did not inspect the low
archive codec: actual consumed file reads already use descriptor-relative
O_NOFOLLOW at every parent and final component (eventflow/archive_io.py:18-43).
Thus arbitrary traversal through a consumed symlink is not established by this
review. The required whole-input invariant still fails for unconsumed extras,
including symlinks and dangling symlinks, in the partial branch. Fix the shared
inventory boundary without weakening existing codecs or modifying generic
archive readers. Preserve missing/unavailable labels and inspect available
evidence even when another dependency is absent.
