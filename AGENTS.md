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

By default, always work in current branch and make meaningful commits along the way.

## Autonomous delivery and standing approval

- The coding agent's primary objective is to deliver verified results. Do not
  turn routine implementation work into a stream of questions or permission
  requests.
- Treat the user's task request as standing approval for all reasonable,
  reversible, in-scope implementation actions and best-judgment rulings. This
  includes dependency compatibility corrections, configuration updates,
  version bumps, documented deviations, and plan adjustments needed to keep
  the requested work executable, provided they do not weaken the scientific
  protocol or expand the task into materially different work.
- Never ask a question merely to transfer a decision to the user when the
  specification, repository evidence, tests, or established engineering
  practice identify a best answer. Make that decision, record the rationale in
  the active ledger or deviation log when material, and continue.
- A repository document's routine “owner approval required” language does not
  create a user round trip when the agent can choose a safe, compliant,
  reversible correction. Treat this standing instruction as that approval and
  preserve the change in versioned history.
- Ask the user only when a higher-priority instruction requires it, or when the
  remaining choice is genuinely unresolved and would cause an irreversible,
  destructive, security-sensitive, externally visible, or materially
  scope-changing outcome. Do not ask “should I continue?” during an approved
  plan.
- When blocked, investigate systematically, choose the strongest supported
  path, document the ruling and its tradeoff, and keep delivering.

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
