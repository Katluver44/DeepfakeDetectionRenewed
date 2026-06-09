#!/usr/bin/env python3
"""Build final_outputs/{mandatory,optional}/<subnarrative>/ and copy the relevant
figures + result/summary files into each. Reproducible; skips missing with a warning.
"""
import shutil
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
FIN  = BASE / "final_outputs"
FIG  = BASE / "outputs" / "figures"
OUT  = BASE / "outputs"
RES  = BASE / "experiments" / "results"
ML   = RES / "mlaad"

# tier -> subnarrative folder -> list of source files (relative to BASE)
MANDATORY = {
 "00_architecture": [
    FIG/"fig1_architecture_ct_pipeline.svg", FIG/"fig1_architecture_ct_pipeline.png",
    BASE/"experiments"/"research.md",
 ],
 "01_objective_hardness_seed_ranking": [
    FIG/"fig_hardness_seed_agreement.png",
    ML/"seed_locked_reproducibility"/"seed_stability_summary.json",
    ML/"seed_locked_reproducibility"/"per_seed_results.json",
    ML/"e6_ranking_lock"/"per_seed_per_attack_eer.csv",
 ],
 "02_geometry_C_and_T": [
    FIG/"fig_C_hardness_crosscorpus.png", FIG/"orthog_partial_correlations.png",
    FIG/"layerwise_r2_comparison.png", FIG/"e1_representation_invariance.png",
    OUT/"final_report.md", OUT/"final_orthogonalization_report.md",
    OUT/"representation_invariance_summary.md", OUT/"sig_layer_features.csv",
 ],
 "03_causality_E9": [
    FIG/"fig_e9_causal_C_main.png", OUT/"ct_causal_intervention"/"e9_causal_C_robust_GOAT.png",
    OUT/"ct_causal_intervention"/"e9_summary.md", OUT/"ct_causal_intervention"/"e9_causal_C.csv",
 ],
 "04_actionability_P1_P5": [
    FIG/"fig_actionability_hardsystems.png",
    ML/"p1_ct_calibration"/"figures"/"eer_before_after.png",
    ML/"p1_ct_calibration"/"summary.md", ML/"p1_ct_calibration"/"per_system_eer.csv",
    ML/"p5_hardness_reweight"/"summary.md",
    ML/"p5_hardness_reweight"/"per_system_metrics_seed42.csv",
 ],
 "05_in_the_wild_C": [
    FIG/"e6_itw_speaker_ct_eer.png",
    OUT/"e6_itw_speaker_summary.md", OUT/"e5_itw_summary.md", OUT/"e6_itw_speaker_ct.csv",
 ],
 "06_mechanism_story1_composite": [
    FIG/"fig_story1_mechanism.png",
    RES/"multiclass_probe"/"multiclass_summary.csv",
    RES/"act_patching"/"act_patching_class_stats.csv",
    RES/"linear_probe"/"probe_metrics_summary.csv",
 ],
}

OPTIONAL = {
 "temporal_smoothing": [
    FIG/"fig_smoothing_dose_dependent.png",
    ML/"smoothing_augmentation"/"smoothing_aug_scatter.png",
    ML/"temporal_intervention"/"hard_vs_easy_sensitivity.png",
    ML/"temporal_intervention"/"variance_vs_eer.png",
    ML/"p4_smoothing_aug"/"summary.md",
    ML/"temporal_intervention"/"causal_interpretation.md",
 ],
 "geometry_supplement": [
    FIG/"orthog_pca.png", FIG/"orthog_variance_partition.png", FIG/"orthog_factor_correlations.png",
    FIG/"orthog_loo_shapley.png", FIG/"layerwise_spearman_profiles.png", FIG/"interaction_panels.png",
    FIG/"e2_graph_independence.png", FIG/"e3_intervention_lite.png",
    OUT/"orthogonalization_summary.md", OUT/"interaction_modeling_summary.md",
    OUT/"graph_independence_summary.md", OUT/"intervention_lite_summary.md",
 ],
 "causality_replication_and_T": [
    OUT/"ct_causal_intervention"/"e9_causal_C_GOAT.png",
    OUT/"ct_causal_intervention"/"e9_causal_T_robust_GOAT.png",
    OUT/"ct_causal_intervention"/"e9_causal_T_GOAT.png",
    OUT/"ct_causal_intervention"/"e9_causal_T.csv",
 ],
 "in_the_wild_supplement": [
    FIG/"e5_itw_ct_manifold.png", FIG/"e5_itw_ct_distributions.png", FIG/"e5_itw_score_vs_ct.png",
    FIG/"e6_itw_within_between.png", FIG/"e6_itw_speaker_quartile_eer.png",
    FIG/"e5_itw_manifold_pca_2d.png", FIG/"e5_itw_manifold_umap_2d.png",
 ],
 "actionability_supplement": [
    ML/"p1_ct_calibration"/"figures"/"calibration_gain_vs_eer.png",
    ML/"p1_ct_calibration"/"figures"/"ct_bonafide_vs_spoof.png",
    ML/"p1_ct_calibration"/"figures"/"auc_before_after.png",
    ML/"p1_ct_calibration"/"figures"/"bal_acc_before_after.png",
    ML/"p3_ct_window11"/"summary.md", ML/"p6_diversity_lambda0.01"/"summary.md",
    ML/"p7_crank_lambda0.1"/"summary.md", ML/"p8_cross_encoder_ensemble"/"summary.md",
 ],
 "robustness_mechanism_E7": [
    OUT/"robustness_mechanism_analysis"/"deltaeer_vs_c.png",
    OUT/"robustness_mechanism_analysis"/"eer_vs_c.png",
    OUT/"robustness_mechanism_analysis"/"quartile_c.png",
    OUT/"robustness_mechanism_analysis"/"auc_vs_c.png",
    OUT/"robustness_mechanism_analysis"/"robustness_summary.md",
 ],
 "manifold_hardness_landscape": [
    ML/"manifold_geometry"/"manifold_geometry_panels.png",
    ML/"extreme_system_analysis"/"wavlm_vs_eer.png",
    ML/"extreme_system_analysis"/"hard_vs_easy_boxplots.png",
    ML/"extreme_system_analysis"/"phoneme_kl_vs_eer.png",
    ML/"residual_hardness"/"residual_hardness_panels.png",
    ML/"residual_hardness_2"/"hypothesis_panels.png",
    ML/"extreme_system_analysis"/"analysis_report.md",
 ],
 "story1_full_mechanism": [
    RES/"multiclass_probe"/"multiclass_ranking.png",
    RES/"act_patching"/"act_patching_violin.png",
    RES/"act_patching"/"act_patching_by_system.png",
    RES/"linear_probe"/"probe_ranking.png",
    RES/"act_patching_probe"/"act_probe_shift_matrix.png",
    RES/"gat_l0_attention"/"kl_divergence.png",
    RES/"gat_l0_attention"/"top_kl_delta_heatmap.png",
    RES/"phoneme_pc1_violin.png", RES/"phoneme_pc1_mean_delta.png",
    RES/"gat_l0_attention"/"report.md",
 ],
 "e4_asvspoof_inconclusive": [
    FIG/"e4_asvspoof_generalization.png",
    OUT/"system_generalization_asvspoof_a01_a06_summary.md",
 ],
}

def build(tier_name, mapping):
    root = FIN / tier_name
    n_ok=n_miss=0
    for sub, files in mapping.items():
        d = root / sub; d.mkdir(parents=True, exist_ok=True)
        for src in files:
            src = Path(src)
            if src.exists():
                shutil.copy2(src, d / src.name); n_ok+=1
            else:
                print(f"  [MISS] {src.relative_to(BASE)}"); n_miss+=1
    print(f"[{tier_name}] copied {n_ok} files into {len(mapping)} subfolders ({n_miss} missing)")
    return n_ok, n_miss

if __name__ == "__main__":
    if FIN.exists(): shutil.rmtree(FIN)
    print(f"Building {FIN.relative_to(BASE)} ...")
    build("mandatory", MANDATORY)
    build("optional",  OPTIONAL)
    # tree summary
    print("\n=== final_outputs tree ===")
    for tier in ["mandatory","optional"]:
        print(tier+"/")
        for sub in sorted((FIN/tier).iterdir()):
            files=sorted(p.name for p in sub.iterdir())
            print(f"  {sub.name}/  ({len(files)} files)")
            for f in files: print(f"      {f}")
