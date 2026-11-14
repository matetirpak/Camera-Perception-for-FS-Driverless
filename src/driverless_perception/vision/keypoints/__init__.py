"""7-keypoint regression on 80x80 cone crops (Dhall et al. 2019): apex, two base corners, and the four points where
the stripe meets the cone body. Loss = keypoint MSE + gamma * cross-ratio terms (gamma = 1e-4). Phase 4."""
