# Outlier Report — Higher-Order Mechanism Analysis

## Spotlight systems

### griffin_lim  (group=hard, mean_EER=0.401)
  phoneme_var_mean: 0.3559
  phoneme_var_median: 0.3493
  frame_mean_dist_mean: 16.5163
  n_segs_mean: 45.8667
  n_nodes_mean: 45.8667

### OuteTTS  (group=hard, mean_EER=0.405)
  phoneme_var_mean: 0.4199
  phoneme_var_median: 0.4224
  frame_mean_dist_mean: 15.8020
  n_segs_mean: 45.0909
  n_nodes_mean: 45.0909

### orpheus-tts-0.1-finetune  (group=easy, mean_EER=0.068)
  phoneme_var_mean: 0.4444
  phoneme_var_median: 0.4397
  frame_mean_dist_mean: 15.9507
  n_segs_mean: 42.0000
  n_nodes_mean: 42.0000

## Correlation context

Top 5 features by |ρ| with EER:

  phoneme_var_mean: ρ=-0.484 p=0.000 | r=-0.423 p=0.001
  phoneme_var_median: ρ=-0.483 p=0.000 | r=-0.453 p=0.000
  frame_mean_dist_mean: ρ=-0.336 p=0.007 | r=-0.357 p=0.004
  n_segs_mean: ρ=-0.196 p=0.123 | r=-0.196 p=0.123
  n_nodes_mean: ρ=-0.194 p=0.127 | r=-0.193 p=0.129