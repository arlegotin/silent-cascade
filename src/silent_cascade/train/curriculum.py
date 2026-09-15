"""Evidence-bound ordered curriculum policy; higher execution is a later phase."""

from dataclasses import dataclass

from silent_cascade.train.state import TrainingError


@dataclass(frozen=True, slots=True)
class CurriculumEvidence:
    requested_stage: str
    oracle_bootstrap: bool = False
    component_episodes: int = 0
    recall_accuracy: float = 0.0
    composition_accuracy: float = 0.0
    chain_accuracy: float = 0.0
    autonomous_episodes: int = 0
    autonomous_success: bool = False
    dynamics_failures: int | None = None
    primary_complete: bool = False


class CurriculumController:
    def __init__(self):
        self.stage = "bootstrap"
        self.history: tuple[CurriculumEvidence, ...] = ()

    def next_stage(self, evidence: CurriculumEvidence) -> str:
        if not isinstance(evidence, CurriculumEvidence):
            raise TrainingError("Curriculum requires explicit evidence")
        stages = ("bootstrap", "one_hop", "two_hop", "primary", "robustness")
        index = stages.index(self.stage)
        if index == 4 or evidence.requested_stage != stages[index + 1]:
            raise TrainingError("Unsupported or out-of-order curriculum promotion")
        justified = (
            evidence.oracle_bootstrap
            if index == 0
            else evidence.component_episodes == 10_000
            and all(
                0.99 < value <= 1.0
                for value in (
                    evidence.recall_accuracy,
                    evidence.composition_accuracy,
                    evidence.chain_accuracy,
                )
            )
            if index == 1
            else evidence.autonomous_episodes == 10_000
            and evidence.autonomous_success
            and evidence.dynamics_failures == 0
            if index == 2
            else evidence.primary_complete
        )
        if not justified:
            raise TrainingError("Curriculum promotion lacks required gate evidence")
        self.stage = evidence.requested_stage
        self.history += (evidence,)
        return self.stage
