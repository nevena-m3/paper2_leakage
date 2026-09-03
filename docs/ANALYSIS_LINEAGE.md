# Paper 2 analysis lineage

## Purpose

This file distinguishes authoritative manuscript-facing analyses from retained development history. Historical notebooks are intentionally preserved; they should not be interpreted as equally authoritative.

The immutable original scientific freeze is `paper2-analysis-freeze-v1.0.0` at commit `55cba2083327aa71be5ea7771a3185daefa0c979`. Post-freeze changes described here are provenance/reproducibility corrections only.

## Goal 1

The final primary source notebook is `10_goal1_COMPLETE_REBUILT_v1_1_final2000BS.ipynb`. The frozen Goal-1 completion manifest records its SHA-256 as:

`e1bf319c7088b09456d53a89fd0825563bc6540d0069ca2045c888d0e2d872ee`

The diagnosis permutation correction is implemented in `goal1_diagnosis_permutation_corrected_v1_1.py`: diagnosis stratification is regenerated after each participant-level label permutation. The historical fixed-label-stratified diagnosis null is not used in the final inference.

Final downstream notebooks are `10B_goal1_FINALIZE_sensitivities_figures_v1_0.ipynb`, `10C_goal1_PUBLICATION_figures_FINAL_v1_2.ipynb`, and `10D_goal1_FINAL_FREEZE_v1_0.ipynb`. Goal-1 final uncertainty uses 2,000 participant bootstrap replicates; formal permutation inference uses 1,000 permutations per task and reports the minimum attainable empirical p-value of 1/1001 when no null replicate is as or more extreme.

## Goal 2

The final A-extraction/freeze path is `18_goal2_BAMBOO_acoustic_extraction_v2_1.ipynb` followed by `19_goal2_BAMBOO_acoustic_representation_freeze_v2_1.ipynb`.

The primary inference path is `20_goal2_inference_consequence_PRIMARY_v1_1_with2000BS.ipynb`, followed by `21_goal2_COMPLETION_sensitivities_and_seal_v1_0_1.ipynb`. The completion layer contains the authoritative correction requiring fold-local tuning of the A-on-Q residualizer inside the outer-training data. Earlier fixed-residualizer outputs are historical and are not manuscript authority.

`22_goal2_final_figures.ipynb` creates the final figure package and final DONE seal from the authoritative completion outputs. The final frozen models are `M_A`, `M_A+Q`, and `M_A-resQ`.

## Goal 3

Stage A: `30_goal3_natural_QA_localization_and_experiment_freeze_v1_1.ipynb`.

Stage B: `31_goal3_signal_only_perturbation_calibration_PREP_v1_0.ipynb`, `32_goal3_signal_only_perturbation_calibration_EXECUTE_v1_1.ipynb`, then the outcome-blind targeted revisions `goal3_stageB_v1_2_targeted_revision.py` and `goal3_stageB_v1_3_targeted_revision.py`. The final Stage-B perturbation manifest is immutable before clinical prediction inspection.

Stage C: `33_goal3_controlled_QA_measurement_PREFLIGHT_v1_0_2.ipynb`. This separates the frozen-segmentation A-implementation reproduction gate from the fresh-resegmentation audit used by the controlled experiment.

Stage D: `34_goal3_Goal2_model_bundle_bridge_FINAL_v1_0_1.ipynb`. Five fold-specific frozen Goal-2 model bundles are sealed only after exact repeat-1 OOF reproduction.

Stage E: the heavy controlled run was executed with `35_goal3_controlled_perturbation_PRIMARY_FINAL_v1_1_0.ipynb`, producing deterministic checkpoints for all 14,792 source/variant rows. A later targeted completion patch, `35_goal3_controlled_perturbation_PRIMARY_FINAL_v1_1_1.ipynb`, reused those checkpoints and made two non-estimand changes: it corrected a support/failure-summary reduction from a Series to a scalar count, and it added explicit baseline-gated prediction-unavailability audit tables. No waveform measurement, perturbation dose, frozen model state, prediction rule, or inferential estimand changed. v1.1.1 is therefore the authoritative Stage-E notebook for release provenance.

Stage F/completion: `36_goal3_FINALIZE_robustness_and_completion_v1_0.ipynb`, then `37_goal3_PUBLICATION_figures_v1_0.ipynb` and manual visual approval of the versioned review figures. The approved versions were:

- Figure 5: `Figure5_Goal3_QA_localization_V2_REVIEW`
- Figure 6: `Figure6_Goal3_controlled_pipeline_V2_REVIEW`
- Figure S7: `FigureS_G3_07_feature_and_offtarget_V5_REVIEW`
- Figure S8: `FigureS_G3_08_robustness_V7_REVIEW`

Those approved files were copied to canonical `*_FINAL` names and hashed into the final figure manifest. The final Goal-3 seal records that no waveform measurement, model fitting, GEE analysis, or bootstrap analysis was rerun during figure finalization.

## Interpretation hierarchy

Goal 1 establishes information availability in Q. Goal 2 establishes that acoustic inference is not invariant to Q within the observed cohort. Goal 3 localizes Q–A coupling and tests direct same-source sensitivity to imposed digital transformations. None of these analyses alone identifies a unique physical cause for natural clinical associations.
