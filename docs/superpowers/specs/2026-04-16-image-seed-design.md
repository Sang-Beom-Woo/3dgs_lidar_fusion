# Image-based Seeding via Temporal Feature Matching

**Date:** 2026-04-16
**Status:** Rolled back — ORB triangulation 결과 품질 부족. 향후 재시도 시 LightGlue/SuperPoint 또는 mono depth 우선 검토.

## Context

Current pipeline ([run_chapter2.py](../../../run_chapter2.py)) seeds 3DGS only from LiDAR points. Seeds are sparse in LiDAR's blind spots (ceiling, floor, above horizon, behind robot) and quantity-limited in textured regions where finer gaussians could improve visual fidelity.

User priorities: **A** — fill LiDAR blind spots · **B** — enrich texture detail in already-covered areas. External depth models (Depth Anything) explicitly out of scope for this iteration.

## Decision

Add image-based seeds via classical **temporal ORB feature matching + triangulation** across consecutive keyframes per camera. Merge with existing LiDAR seeds via simple union + voxel downsample.

**Inherent limitation accepted:** feature matching does not solve **A** (blind spots usually lack texture). Primary benefit is **B**. Follow-up work (mono depth) may address **A** later.

## Architecture

### New module — `image_seeder.py`

```python
def generate_image_seeds(
    ds: BaseDataDataset,
    window: int = 2,
    n_features: int = 3000,
    reproj_thr: float = 2.0,
    min_depth: float = 0.3,
    max_depth: float = 30.0,
    stride: int = 1,
    max_kf: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Temporal ORB matching + triangulation per camera → (pts_world[M,3], colors[M,3])."""
```

Internal helpers:
- `_match_pair(img1, img2, K, orb, bf) -> (pts1, pts2)` — ORB + BF + essential-matrix RANSAC filter
- `_triangulate(pts1, pts2, T_wc1, T_wc2, K, min_depth, max_depth, reproj_thr) -> (Xw[M,3], mask[N])`

### Processing flow

Per camera (front / left / right), independently:

1. Iterate keyframes in order (respecting `stride` / `max_kf`), collecting `(T_world_cam, undistorted_image, new_K)` triples.
2. Maintain a sliding window buffer of the last `window+1` frames.
3. For each new frame, match against each of the previous frames in the buffer.
4. For each pair:
   - ORB detect + describe (`n_features` features)
   - BFMatcher with `crossCheck=True`
   - `cv2.findEssentialMat` + RANSAC → inlier mask
   - `cv2.triangulatePoints` using `P_i = K · [R_i | t_i]` from known poses
   - Filter: depth in both views ∈ [`min_depth`, `max_depth`], reprojection error < `reproj_thr` px
   - Color = mean(img1[v1,u1], img2[v2,u2]) in RGB, 0~1
5. Accumulate all passing triplets → return concatenated arrays.

### Integration — [run_chapter2.py](../../../run_chapter2.py)

`accumulate_seed()` gains a `use_image_seed` branch:

```
LiDAR seed ─┐
            ├→ concat → voxel downsample → gaussians
image seed ─┘
```

New CLI flags:
- `--image-seed` (enable; default off)
- `--orb-features` (default 3000)
- `--match-window` (default 2)
- `--reproj-thr` (default 2.0)
- `--image-seed-min-depth` (default 0.3)
- `--image-seed-max-depth` (default 30.0)

Existing flags (`--stride`, `--max-kf`, `--init-voxel`) apply consistently to both seed sources.

## Data flow

```
BaseDataDataset ── LiDAR path ──→ colorize_keyframe ──→ pts_lidar (world)
                ╰─ image path ──→ generate_image_seeds ──→ pts_img (world)
                                                            │
                                   concatenate + voxel_down_sample
                                                            │
                                                      LidarVisualGS.__init__
```

## Error handling

- Keyframe missing this camera → skip frame entirely for that camera (not a pair candidate).
- Matches < 8 or `findEssentialMat` returns `None` → skip pair.
- All triangulated points failing filters → pair contributes zero seeds, pipeline continues.
- Degenerate motion (pure rotation) → essential-matrix RANSAC degenerate; reprojection-error and depth filters catch residual garbage.

## Testing

- **Unit smoke (per function):** run `generate_image_seeds` on 20 kf, assert `pts.shape[0] > 0` and finite values.
- **Integration:** run `run_chapter2.py --image-seed --max-kf 30 --iters 0 --out seed_only.ply`, confirm PLY contains more points than LiDAR-only baseline under same voxel.
- **Visual:** open produced `map_splat.ply` in `splat_viewer.py`, compare textured-wall detail against LiDAR-only output.

## Out of scope

- Non-OpenCV feature/matching backends (SuperPoint, LightGlue, DISK)
- Cross-camera stereo at same keyframe (3-cam overlap negligible per calibration geometry)
- Bundle adjustment / pose refinement
- Mono depth priors (Depth Anything / Metric3D)
- Online densification during training

## Follow-ups

- Measure fill rate in A-category regions (ceiling/floor mask); if insufficient, plan mono depth integration as separate spec.
- Consider upgrading to LightGlue if ORB match density is too sparse.
