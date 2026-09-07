use super::generation_replacement_is_quiescent;

#[test]
fn generation_replacement_requires_submitted_execution_and_readback_observation_to_be_quiescent() {
    assert!(generation_replacement_is_quiescent(0, 0));
    assert!(!generation_replacement_is_quiescent(1, 0));
    assert!(!generation_replacement_is_quiescent(0, 1));
    assert!(!generation_replacement_is_quiescent(1, 1));
}
