import pytest

from silent_cascade.train.state import TrainingError


def test_curriculum_requires_ordered_evidence():
    from silent_cascade.train.curriculum import CurriculumController, CurriculumEvidence

    controller = CurriculumController()
    with pytest.raises(TrainingError):
        controller.next_stage(CurriculumEvidence(requested_stage="two_hop"))
    assert (
        controller.next_stage(CurriculumEvidence(requested_stage="one_hop", oracle_bootstrap=True))
        == "one_hop"
    )
    with pytest.raises(TrainingError):
        controller.next_stage(
            CurriculumEvidence(
                requested_stage="two_hop",
                component_episodes=10000,
                recall_accuracy=0.99,
                composition_accuracy=1.0,
                chain_accuracy=1.0,
            )
        )
    assert (
        controller.next_stage(
            CurriculumEvidence(
                requested_stage="two_hop",
                component_episodes=10000,
                recall_accuracy=1.0,
                composition_accuracy=1.0,
                chain_accuracy=1.0,
            )
        )
        == "two_hop"
    )
    with pytest.raises(TrainingError):
        controller.next_stage(
            CurriculumEvidence(
                requested_stage="primary",
                autonomous_episodes=9999,
                autonomous_success=True,
                dynamics_failures=0,
            )
        )
    assert (
        controller.next_stage(
            CurriculumEvidence(
                requested_stage="primary",
                autonomous_episodes=10000,
                autonomous_success=True,
                dynamics_failures=0,
            )
        )
        == "primary"
    )
