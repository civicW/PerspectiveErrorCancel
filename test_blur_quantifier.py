#!/usr/bin/env python3
"""
test_blur_quantifier.py
=======================
Unit tests for blur quantification module.

Usage:
    python test_blur_quantifier.py
"""

import numpy as np
import cv2
from blur_quantifier import (
    quantify_blur_fft,
    quantify_blur_spatial,
    BlurReportFFT,
    BlurReportSpatial,
)


def create_test_image(size=(480, 640), blur_sigma=0):
    """Create a test image with optional Gaussian blur."""
    # Create checkerboard pattern
    img = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    square_size = 40
    for i in range(0, size[0], square_size):
        for j in range(0, size[1], square_size):
            if ((i // square_size) + (j // square_size)) % 2 == 0:
                img[i:i+square_size, j:j+square_size] = 255

    # Apply blur if requested
    if blur_sigma > 0:
        ksize = int(blur_sigma * 6) | 1  # Ensure odd kernel size
        img = cv2.GaussianBlur(img, (ksize, ksize), blur_sigma)

    return img


def test_fft_blur():
    """Test FFT-based blur quantification."""
    print("\n=== Testing FFT Blur Quantification ===")

    # Sharp image
    sharp = create_test_image(blur_sigma=0)
    report_sharp = quantify_blur_fft(sharp)
    print(f"\nSharp image:")
    report_sharp.print_summary()

    # Blurred image
    blurred = create_test_image(blur_sigma=5.0)
    report_blurred = quantify_blur_fft(blurred)
    print(f"\nBlurred image (sigma=5.0):")
    report_blurred.print_summary()

    # Verify sharp image has higher sharpness score
    assert report_sharp.sharpness_score > report_blurred.sharpness_score, \
        "Sharp image should have higher sharpness score"
    assert report_sharp.hf_ratio > report_blurred.hf_ratio, \
        "Sharp image should have higher HF ratio"

    print("\n✓ FFT blur quantification test passed")


def test_spatial_blur():
    """Test spatially-varying blur quantification."""
    print("\n=== Testing Spatial Blur Quantification ===")

    # Sharp image
    sharp = create_test_image(blur_sigma=0)
    report_sharp = quantify_blur_spatial(
        sharp,
        window_size=32,
        stride=16,
        query_coords=(320, 240)
    )
    print(f"\nSharp image:")
    report_sharp.print_summary()

    # Blurred image
    blurred = create_test_image(blur_sigma=5.0)
    report_blurred = quantify_blur_spatial(
        blurred,
        window_size=32,
        stride=16,
        query_coords=(320, 240)
    )
    print(f"\nBlurred image (sigma=5.0):")
    report_blurred.print_summary()

    # Verify blurred image has higher sigma
    assert report_blurred.mean_sigma > report_sharp.mean_sigma, \
        "Blurred image should have higher mean sigma"
    assert report_blurred.query_sigma > report_sharp.query_sigma, \
        "Blurred image should have higher query sigma"

    print("\n✓ Spatial blur quantification test passed")


def test_spatially_varying_blur():
    """Test detection of spatially-varying blur (sharp center, blurred edges)."""
    print("\n=== Testing Spatially-Varying Blur Detection ===")

    # Create image with sharp center and blurred edges
    img = create_test_image(blur_sigma=0)
    h, w = img.shape[:2]

    # Blur edges
    edge_width = 100
    for y in range(h):
        for x in range(w):
            # Distance from center
            dx = abs(x - w // 2)
            dy = abs(y - h // 2)
            dist = max(dx / (w // 2), dy / (h // 2))

            if dist > 0.6:  # Outer 40%
                # Apply increasing blur toward edges
                sigma = (dist - 0.6) * 15.0
                ksize = int(sigma * 2) | 1
                if ksize >= 3:
                    patch = img[max(0, y-ksize):min(h, y+ksize+1),
                               max(0, x-ksize):min(w, x+ksize+1)]
                    if patch.size > 0:
                        blurred_patch = cv2.GaussianBlur(patch, (ksize, ksize), sigma)
                        img[max(0, y-ksize):min(h, y+ksize+1),
                            max(0, x-ksize):min(w, x+ksize+1)] = blurred_patch

    # Quantify
    report = quantify_blur_spatial(img, window_size=32, stride=16)
    print(f"\nSpatially-varying blur:")
    report.print_summary()

    # Verify edge/centre ratio > 1
    assert report.edge_centre_ratio > 1.0, \
        "Edge should be more blurred than center"

    print("\n✓ Spatially-varying blur detection test passed")


def test_comparison():
    """Test before/after comparison scenario."""
    print("\n=== Testing Before/After Comparison ===")

    # Simulate "before correction" (blurred edges)
    before = create_test_image(blur_sigma=0)
    h, w = before.shape[:2]

    # Add edge blur
    edge_mask = np.zeros((h, w), dtype=np.float32)
    for y in range(h):
        for x in range(w):
            dx = abs(x - w // 2) / (w // 2)
            dy = abs(y - h // 2) / (h // 2)
            edge_mask[y, x] = max(dx, dy)

    # Apply spatially-varying blur
    for sigma_level in range(1, 6):
        mask = (edge_mask > sigma_level * 0.15) & (edge_mask <= (sigma_level + 1) * 0.15)
        if mask.any():
            ksize = sigma_level * 4 + 1
            blurred = cv2.GaussianBlur(before, (ksize, ksize), sigma_level)
            before[mask] = blurred[mask]

    # Simulate "after correction" (less blur)
    after = create_test_image(blur_sigma=1.0)

    # Quantify both
    query_coords = (w - 100, h - 100)  # Edge position

    blur_before = quantify_blur_spatial(before, query_coords=query_coords)
    blur_after = quantify_blur_spatial(after, query_coords=query_coords)

    print(f"\nBefore correction:")
    blur_before.print_summary()

    print(f"\nAfter correction:")
    blur_after.print_summary()

    # Calculate reduction
    if blur_before.query_sigma and blur_after.query_sigma:
        reduction = (1 - blur_after.query_sigma / blur_before.query_sigma) * 100
        print(f"\n{'='*52}")
        print(f"  Blur at {query_coords}:")
        print(f"    Before    : {blur_before.query_sigma:.1f} px")
        print(f"    After     : {blur_after.query_sigma:.1f} px")
        print(f"    Reduction : {reduction:.1f}%")
        print(f"{'='*52}")

        assert reduction > 0, "Should show blur reduction"

    print("\n✓ Before/after comparison test passed")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Blur Quantifier Unit Tests")
    print("=" * 60)

    try:
        test_fft_blur()
        test_spatial_blur()
        test_spatially_varying_blur()
        test_comparison()

        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)

    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
