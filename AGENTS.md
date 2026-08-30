# Repository workflow

All implementation work in this repository—including planning, code changes,
tests, refactors, bug fixes, and release preparation—must be performed through
the Superpowers workflow.

Before taking any implementation action:

1. Invoke `superpowers:using-superpowers`.
2. Invoke every Superpowers skill applicable to the task.
3. For new functionality or behavior changes, invoke
   `superpowers:brainstorming` before implementation.
4. For multi-step implementation, invoke `superpowers:writing-plans` before
   touching source code.
5. Use `superpowers:test-driven-development` for every feature or bug fix.
6. Use `superpowers:systematic-debugging` before proposing or applying a fix for
   any failure or unexpected behavior.
7. Use the applicable Superpowers execution and review skills while carrying
   out an approved plan.
8. Before reporting completion, invoke
   `superpowers:verification-before-completion` and run the relevant checks.

Do not implement work outside this workflow. If a required Superpowers skill is
unavailable, stop and ask the user to install or enable it.

## Silent Cascade contract

- The Python package is named `silent_cascade`; the CLI executable is
  `silent-cascade`.
- The canonical design specification is
  `docs/superpowers/specs/2026-08-30-silent-cascade-design.md`.
- An approved Superpowers implementation plan is required before creating or
  modifying source, test, build, configuration, workflow, or release files.
- A plan is approved only after the user explicitly approves it in the task
  thread. Approval of this design specification is not approval of an
  implementation plan.
- Execute the program through phase-scoped Superpowers plans. Do not replace
  those plans with one monolithic repository-wide implementation pass.
- The scientific protocol, controls, gates, and claim boundaries in the
  canonical specification may not be weakened without explicit user approval
  and a versioned specification change.
- Primary scientific execution must remain offline and record exactly zero
  foundation-model calls.
