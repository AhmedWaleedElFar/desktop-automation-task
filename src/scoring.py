"""
Scoring Phase: Use Gaussian-based centrality scoring to rank candidate regions.
Inspired by ScreenSeekeR's voting mechanism.

Scores regions based on how well voting boxes cluster in their centers.
"""

import math
from typing import List, Tuple, Dict, Optional


def gaussian_score(
    voting_boxes: List[Dict],
    region: Tuple[int, int, int, int],
    sigma: float = 0.3
) -> float:
    """
    Score a region based on how well voting boxes cluster in its center.
    
    Uses Gaussian formula: exp(-2σ²((x'-0.5)² + (y'-0.5)²))
    - Boxes at center of region = high score
    - Boxes at edges of region = low score
    - Multiple boxes clustering together = compounded high score
    
    Args:
        voting_boxes: List of {"box": [x1,y1,x2,y2], "confidence": 0.0-1.0}
        region: [x1, y1, x2, y2] region to score
        sigma: Gaussian variance (default 0.3 from ScreenSeekeR)
    
    Returns:
        Float score (0.0 to ~1.0+, unbounded)
    """
    x1, y1, x2, y2 = region
    region_w = max(1, x2 - x1)
    region_h = max(1, y2 - y1)
    
    total_score = 0.0
    
    for box_data in voting_boxes:
        box = box_data["box"]
        confidence = box_data.get("confidence", 1.0)
        
        bx1, by1, bx2, by2 = box
        
        # Box center in absolute coordinates
        box_center_x = (bx1 + bx2) / 2.0
        box_center_y = (by1 + by2) / 2.0
        
        # Normalize to [0, 1] within region
        x_norm = (box_center_x - x1) / region_w
        y_norm = (box_center_y - y1) / region_h
        
        # Clamp to region bounds
        x_norm = max(0.0, min(1.0, x_norm))
        y_norm = max(0.0, min(1.0, y_norm))
        
        # Gaussian: peak at (0.5, 0.5) center
        # exp(-2σ²(x'² + y'²)) where (x',y') are distances from center
        dist_sq = (x_norm - 0.5)**2 + (y_norm - 0.5)**2
        gaussian = math.exp(-2 * sigma**2 * dist_sq)
        
        # Weight by confidence of this voting box
        total_score += gaussian * confidence
    
    return total_score


def score_regions(
    voting_boxes: List[Dict],
    candidate_regions: List[List[int]],
    sigma: float = 0.3
) -> List[Tuple[int, float]]:
    """
    Score all candidate regions based on voting boxes.
    
    Args:
        voting_boxes: List of {"box": [x1,y1,x2,y2], "confidence": 0.0-1.0}
        candidate_regions: List of [x1, y1, x2, y2] regions
        sigma: Gaussian variance
    
    Returns:
        List of (region_index, score) tuples, sorted by score descending
    """
    scores = []
    
    for i, region in enumerate(candidate_regions):
        score = gaussian_score(voting_boxes, tuple(region), sigma)
        scores.append((i, score))
    
    # Sort by score descending
    scores.sort(key=lambda x: x[1], reverse=True)
    
    return scores


def iou(box1: List[int], box2: List[int]) -> float:
    """
    Calculate Intersection over Union (IoU) between two boxes.
    
    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]
    
    Returns:
        IoU score (0.0 to 1.0)
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Intersection
    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)
    
    if inter_xmax < inter_xmin or inter_ymax < inter_ymin:
        return 0.0
    
    inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
    
    # Union
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area


def nms(
    regions: List[List[int]],
    scores: List[float],
    iou_threshold: float = 0.5
) -> List[Tuple[int, float]]:
    """
    Non-Maximum Suppression: Remove overlapping regions, keep highest-scoring ones.
    
    Args:
        regions: List of [x1, y1, x2, y2] regions
        scores: List of scores corresponding to regions
        iou_threshold: Overlapping regions with IoU > threshold are suppressed
    
    Returns:
        List of (region_index, score) tuples for kept regions, sorted by score
    """
    if len(regions) == 0:
        return []
    
    # Create list of (index, score) and sort by score descending
    indexed_scores = [(i, scores[i]) for i in range(len(regions))]
    indexed_scores.sort(key=lambda x: x[1], reverse=True)
    
    keep = []
    
    for idx, score in indexed_scores:
        # Check overlap with already-kept regions
        suppressed = False
        
        for kept_idx, kept_score in keep:
            overlap = iou(regions[idx], regions[kept_idx])
            if overlap > iou_threshold:
                # This region overlaps significantly with a kept region
                suppressed = True
                break
        
        if not suppressed:
            keep.append((idx, score))
    
    return keep


def get_best_region(
    voting_boxes: List[Dict],
    candidate_regions: List[List[int]],
    sigma: float = 0.3,
    use_nms: bool = True,
    iou_threshold: float = 0.5
) -> Tuple[int, float, List[int]]:
    """
    Get the best region based on Gaussian scoring and optional NMS.
    
    Args:
        voting_boxes: List of {"box": [x1,y1,x2,y2], "confidence": 0.0-1.0}
        candidate_regions: List of [x1, y1, x2, y2] regions
        sigma: Gaussian variance
        use_nms: Apply Non-Maximum Suppression
        iou_threshold: NMS overlap threshold
    
    Returns:
        Tuple of (best_region_index, best_score, best_region_box)
    """
    # Score all regions
    scored = score_regions(voting_boxes, candidate_regions, sigma)
    
    if not scored:
        return 0, 0.0, candidate_regions[0]
    
    # Apply NMS if requested
    if use_nms and len(scored) > 1:
        scores_only = [s for _, s in scored]
        kept = nms(candidate_regions, scores_only, iou_threshold)
        
        if kept:
            best_idx, best_score = kept[0]
        else:
            best_idx, best_score = scored[0]
    else:
        best_idx, best_score = scored[0]
    
    best_region = candidate_regions[best_idx]
    
    return best_idx, best_score, best_region


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("SCORING TEST - Gaussian Centrality + NMS")
    print("=" * 70)
    
    # Test 1: Gaussian scoring
    print("\n[TEST 1] Gaussian Scoring")
    print("-" * 70)
    
    # Create mock regions and voting boxes
    regions = [
        [0, 0, 100, 100],        # Region 0: top-left (Center is 50, 50)
        [50, 50, 150, 150],      # Region 1: center (Center is 100, 100)
        [200, 200, 300, 300]     # Region 2: bottom-right (Center is 250, 250)
    ]
    
    # FIX: Shift coordinates to cluster around Region 1's center (100, 100)
    voting_boxes = [
        {"box": [95, 95, 105, 105], "confidence": 0.95},      # Very close to region 1 center (100, 100)
        {"box": [98, 98, 102, 102], "confidence": 0.92},      # Very close to region 1 center (100, 100)
        {"box": [50, 50, 55, 55], "confidence": 0.5},         # At region 1 edge / Region 0 center
        {"box": [245, 245, 255, 255], "confidence": 0.3},    # Near region 2 center
    ]
    
    print(f"Regions: {regions}")
    print(f"Voting boxes: {len(voting_boxes)} predictions")
    
    scored = score_regions(voting_boxes, regions)
    
    for region_idx, score in scored:
        print(f"  Region {region_idx} {regions[region_idx]}: score={score:.3f}")
    
    # Region 1 should have highest score (boxes cluster in center)
    assert scored[0][0] == 1, "Region 1 should rank highest"
    print("✓ Gaussian scoring correct: region with centered votes ranks highest")
    
    # Test 2: IoU
    print("\n[TEST 2] Intersection over Union (IoU)")
    print("-" * 70)
    
    box_a = [0, 0, 100, 100]
    box_b = [50, 50, 150, 150]
    box_c = [200, 200, 300, 300]
    
    iou_ab = iou(box_a, box_b)
    iou_ac = iou(box_a, box_c)
    
    print(f"Box A: {box_a}")
    print(f"Box B: {box_b}")
    print(f"Box C: {box_c}")
    print(f"IoU(A, B): {iou_ab:.3f}")
    print(f"IoU(A, C): {iou_ac:.3f}")
    
    assert 0 < iou_ab < 1, "Overlapping boxes should have 0 < IoU < 1"
    assert iou_ac == 0, "Non-overlapping boxes should have IoU = 0"
    print("✓ IoU calculation correct")
    
    # Test 3: NMS
    print("\n[TEST 3] Non-Maximum Suppression (NMS)")
    print("-" * 70)
    
    regions_for_nms = [
        [0, 0, 100, 100],          # 0: high score but overlaps with 1
        [50, 50, 150, 150],        # 1: highest score
        [60, 60, 160, 160],        # 2: overlaps heavily with 1, suppress
        [300, 300, 400, 400]       # 3: isolated, keep
    ]
    
    scores_for_nms = [0.7, 0.95, 0.8, 0.6]  # 1 > 2 > 0 > 3 by score
    
    print(f"Before NMS:")
    for i, (region, score) in enumerate(zip(regions_for_nms, scores_for_nms)):
        print(f"  Region {i}: score={score:.2f}")
    
    kept = nms(regions_for_nms, scores_for_nms, iou_threshold=0.3)
    
    print(f"After NMS (threshold=0.3):")
    for idx, score in kept:
        print(f"  Region {idx}: score={score:.2f}")
    
    # Should keep regions 1 and 3 (highest scoring non-overlapping)
    kept_indices = [idx for idx, _ in kept]
    assert 1 in kept_indices, "Region 1 (highest score) should be kept"
    assert 3 in kept_indices, "Region 3 (isolated) should be kept"
    print("✓ NMS correctly removes overlapping regions")
    
    # Test 4: Full pipeline
    print("\n[TEST 4] Full Scoring Pipeline")
    print("-" * 70)
    
    best_idx, best_score, best_region = get_best_region(
        voting_boxes, regions, sigma=0.3, use_nms=False
    )
    
    print(f"Best region index: {best_idx}")
    print(f"Best region: {best_region}")
    print(f"Best score: {best_score:.3f}")
    
    assert best_idx == 1, "Best region should be index 1"
    assert best_score > 0, "Best score should be positive"
    print("✓ Full pipeline working correctly")
    
    print("\n" + "=" * 70)
    print("✅ All scoring tests passed!")
    print("=" * 70)
