//! Tests for deployment profiles

use super::*;

#[test]
fn test_deployment_profile_default() {
    let profile = DeploymentProfile::default();
    assert_eq!(profile, DeploymentProfile::Balanced);
}

#[test]
fn test_deployment_profile_variants() {
    // All 4 variants are distinct
    let variants = [
        DeploymentProfile::Embedded,
        DeploymentProfile::Balanced,
        DeploymentProfile::Scientific,
        DeploymentProfile::Custom,
    ];

    for i in 0..variants.len() {
        for j in (i + 1)..variants.len() {
            assert_ne!(variants[i], variants[j]);
        }
    }
}

#[test]
fn test_deployment_profile_clone_copy() {
    let profile = DeploymentProfile::Scientific;
    let cloned = profile.clone();
    let copied = profile;
    assert_eq!(profile, cloned);
    assert_eq!(profile, copied);
}
