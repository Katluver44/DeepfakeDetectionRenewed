# Higher-Order Mechanism Analysis — Summary

Hard systems (8): FireRedTTS-2.0, Index-TTS-1.5, Spark-TTS-0.5B, VoxCPM-1.5, Higgs-Audio-V2, ZipVoice, OuteTTS, griffin_lim
Easy systems (10): orpheus-tts-0.1-finetune, Kitten-TTS-Nano-0.1, Veena, Supertonic, kokoro, Kitten-TTS-Nano-0.2, Ringg Squirrel TTS v1.0, DeepGram, facebook_mms-tts-eng, Indri-TTS-0.1

## Correlation with EER (n=63)

  phoneme_var_mean: ρ=-0.484 p=0.000 | r=-0.423 p=0.001 *
  phoneme_var_median: ρ=-0.483 p=0.000 | r=-0.453 p=0.000 *
  frame_mean_dist_mean: ρ=-0.336 p=0.007 | r=-0.357 p=0.004 *
  n_segs_mean: ρ=-0.196 p=0.123 | r=-0.196 p=0.123 
  n_nodes_mean: ρ=-0.194 p=0.127 | r=-0.193 p=0.129 
  n_edges_mean: ρ=-0.194 p=0.127 | r=-0.194 p=0.128 
  max_entropy_mean: ρ=-0.193 p=0.129 | r=-0.030 p=0.813 
  edge_density_mean: ρ=+0.160 p=0.209 | r=+0.135 p=0.292 
  median_entropy_mean: ρ=-0.133 p=0.298 | r=-0.142 p=0.267 
  var_entropy_mean: ρ=-0.124 p=0.335 | r=-0.139 p=0.277 

Significant predictors (p<0.05): 3
  phoneme_var_mean: ρ=-0.484
  phoneme_var_median: ρ=-0.483
  frame_mean_dist_mean: ρ=-0.336

## Hard vs Easy comparison

  frame_mean_dist_mean: Δ=-1.4328  d=-1.357  perm-p=0.011 *
  phoneme_var_median: Δ=-0.0323  d=-0.947  perm-p=0.069 
  phoneme_var_mean: Δ=-0.0300  d=-0.902  perm-p=0.082 
  edge_density_mean: Δ=+0.0147  d=+0.752  perm-p=0.139 
  avg_degree_mean: Δ=-0.1448  d=-0.751  perm-p=0.123 
  n_segs_mean: Δ=-2.7634  d=-0.708  perm-p=0.169 
  n_nodes_mean: Δ=-2.7634  d=-0.708  perm-p=0.152 
  n_edges_mean: Δ=-27.5717  d=-0.707  perm-p=0.160 
  max_entropy_mean: Δ=-0.0097  d=-0.703  perm-p=0.085 
  var_entropy_mean: Δ=-0.0110  d=-0.686  perm-p=0.185 

## Multivariate (LOO-CV R²)

  Linear regression: 0.241
  Lasso:             0.278  (α=0.0013)
  Random Forest:     0.184

Lasso selected features: ['phoneme_var_median', 'frame_mean_dist_mean', 'frame_var_dist_mean', 'mean_entropy_mean', 'median_entropy_mean', 'var_entropy_mean', 'mean_gini_mean', 'mean_top1_mass_mean', 'n_nodes_mean', 'n_edges_mean']

Top-5 RF importances:
  phoneme_var_median: 0.2727
  phoneme_var_mean: 0.1686
  frame_var_dist_mean: 0.0928
  var_entropy_mean: 0.0865
  mean_gini_mean: 0.0650