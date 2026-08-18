"""SURF/FLANN/RANSAC 全局单应性配准。

将扫描件对齐到模板坐标系，消除由扫描仪引起的平移、旋转、缩放和透视扭曲。

核心流程：
1. SURF 特征提取  — 扫描件和模板各提取尺度不变特征点
2. FLANN 匹配     — KD-Tree 近似最近邻搜索，k=2
3. Lowe 比率测试  — 保留 d1/d2 < 0.70 的可靠匹配，剔除模棱两可的配对
4. RANSAC 单应性  — 从含有错误匹配的集合中鲁棒估计 3×3 透视变换矩阵
5. warpPerspective — 将扫描件变换到模板尺寸，边界填白
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import cv2
import numpy as np

from pixel_diff.exceptions import AlignmentError
from pixel_diff.models import PixelDiffConfig


@dataclass(frozen=True)
class AlignmentResult:
    """配准结果：对齐后的扫描件 BGR 图像及诊断指标。"""

    aligned_bgr: np.ndarray
    """对齐后的扫描件图像，尺寸与模板一致。"""

    good_matches: int
    """经过 Lowe 比率测试保留的优质匹配点数量。"""

    inlier_ratio: float
    """RANSAC 内点比例 = inlier 数 / good_matches 数，越接近 1 越好。"""

    homography: np.ndarray
    """3×3 透视变换矩阵（单应性），将扫描件坐标映射到模板坐标。"""

    detector: str
    """实际使用的特征检测器名称。"""

    detector_fallback: bool
    """是否从首选检测器回退到备用检测器。"""

    feature_downsampled: bool = False
    feature_scale: float = 1.0
    feature_downsample_fallback: bool = False
    blank_page_alignment: bool = False
    blank_page_pair: bool = False


@dataclass(frozen=True)
class FeatureDetector:
    detector: Any
    name: str
    fallback: bool


@dataclass(frozen=True)
class HomographyEstimate:
    homography: np.ndarray
    good_matches: int
    inlier_ratio: float
    detector: str
    detector_fallback: bool


def _restore_homography_scale(
    homography: np.ndarray,
    *,
    scan_scale: float,
    template_scale: float,
) -> np.ndarray:
    """Convert a reduced-coordinate homography to original image coordinates."""
    scan_scaling = np.diag([scan_scale, scan_scale, 1.0])
    template_scaling = np.diag([template_scale, template_scale, 1.0])
    return np.asarray(
        np.linalg.inv(template_scaling) @ homography @ scan_scaling,
        dtype=np.float64,
    )


def _find_homography_deterministic(
    scan_points: np.ndarray,
    template_points: np.ndarray,
    *,
    reprojection_threshold: float,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Run RANSAC from a fixed OpenCV RNG state for reproducible fallback."""
    cv2.setRNGSeed(12345)
    homography, inlier_mask = cv2.findHomography(
        scan_points,
        template_points,
        cv2.RANSAC,
        reprojection_threshold,
    )
    return homography, inlier_mask


def align_scan_to_template_bgr(
    scan_bgr: np.ndarray,
    template_bgr: np.ndarray,
    config: PixelDiffConfig,
) -> AlignmentResult:
    """使用全局单应性将扫描件 BGR 图像对齐到模板 BGR 图像。

    整个流程对灰度图操作（SURF/SIFT 不依赖颜色），最后对原彩色图做 warp。

    Raises:
        AlignmentError: 特征点为空、优质匹配不足或单应性计算失败时抛出。
    """
    scan_gray = cv2.cvtColor(scan_bgr, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)
    if config.blank_page_alignment_enabled:
        scan_blank = _is_effectively_blank(scan_gray, config)
        template_blank = _is_effectively_blank(template_gray, config)
        if scan_blank or template_blank:
            target_height, target_width = template_gray.shape
            aligned = (
                scan_bgr.copy()
                if scan_bgr.shape[:2] == template_bgr.shape[:2]
                else cv2.resize(scan_bgr, (target_width, target_height))
            )
            return AlignmentResult(
                aligned_bgr=aligned,
                good_matches=0,
                inlier_ratio=1.0 if scan_blank and template_blank else 0.0,
                homography=np.eye(3, dtype=np.float64),
                detector="blank_identity",
                detector_fallback=False,
                blank_page_alignment=True,
                blank_page_pair=scan_blank and template_blank,
            )
    feature_downsampled = False
    feature_downsample_fallback = False
    feature_scale = 1.0

    if (
        config.alignment_feature_downsample_enabled
        and config.alignment_feature_scale < 1.0
    ):
        feature_scale = config.alignment_feature_scale
        scan_small = cv2.resize(
            scan_gray,
            None,
            fx=feature_scale,
            fy=feature_scale,
            interpolation=cv2.INTER_AREA,
        )
        template_small = cv2.resize(
            template_gray,
            None,
            fx=feature_scale,
            fy=feature_scale,
            interpolation=cv2.INTER_AREA,
        )
        try:
            estimate = _estimate_homography(
                scan_small,
                template_small,
                config,
                max(0.5, config.ransac_reprojection_threshold * feature_scale),
            )
            if estimate.inlier_ratio < config.alignment_feature_min_inlier_ratio:
                raise AlignmentError(
                    "alignment: reduced feature inlier ratio below configured threshold"
                )
            homography = _restore_homography_scale(
                estimate.homography,
                scan_scale=feature_scale,
                template_scale=feature_scale,
            )
            feature_downsampled = True
        except AlignmentError:
            if not config.alignment_feature_fallback_enabled:
                raise
            estimate = _estimate_homography(
                scan_gray,
                template_gray,
                config,
                config.ransac_reprojection_threshold,
            )
            homography = estimate.homography
            feature_scale = 1.0
            feature_downsample_fallback = True
    else:
        estimate = _estimate_homography(
            scan_gray,
            template_gray,
            config,
            config.ransac_reprojection_threshold,
        )
        homography = estimate.homography

    template_height, template_width = template_bgr.shape[:2]
    aligned = cv2.warpPerspective(
        scan_bgr,
        homography,
        (template_width, template_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    return AlignmentResult(
        aligned_bgr=aligned,
        good_matches=estimate.good_matches,
        inlier_ratio=estimate.inlier_ratio,
        homography=homography,
        detector=estimate.detector,
        detector_fallback=estimate.detector_fallback,
        feature_downsampled=feature_downsampled,
        feature_scale=feature_scale,
        feature_downsample_fallback=feature_downsample_fallback,
    )


def _is_effectively_blank(gray: np.ndarray, config: PixelDiffConfig) -> bool:
    ink_ratio = float(np.mean(gray < config.blank_page_ink_threshold))
    return ink_ratio <= config.blank_page_max_ink_ratio


def _estimate_homography(
    scan_gray: np.ndarray,
    template_gray: np.ndarray,
    config: PixelDiffConfig,
    reprojection_threshold: float,
) -> HomographyEstimate:
    """Estimate a homography in the coordinate system of the supplied images."""
    cv2.setRNGSeed(12345)
    feature_detector = _create_feature_detector(config)
    detector = feature_detector.detector
    scan_keypoints, scan_descriptors = detector.detectAndCompute(scan_gray, None)
    template_keypoints, template_descriptors = detector.detectAndCompute(template_gray, None)

    # 描述子为空 = 图像几乎没有可检测的结构特征
    if scan_descriptors is None or template_descriptors is None:
        raise AlignmentError(f"alignment: {feature_detector.name.upper()} descriptors are empty")
    if (
        len(scan_keypoints) < config.min_good_matches
        or len(template_keypoints) < config.min_good_matches
    ):
        raise AlignmentError(f"alignment: not enough {feature_detector.name.upper()} keypoints")

    # ── 2) FLANN 特征匹配（KD-Tree，k=2 近邻）──
    index_params: dict[str, bool | int | float | str] = {"algorithm": 1, "trees": 5}
    search_params: dict[str, bool | int | float | str] = {"checks": 50}
    matcher = cv2.FlannBasedMatcher(index_params, search_params)
    raw_matches = matcher.knnMatch(scan_descriptors, template_descriptors, k=2)

    # ── 3) Lowe 比率测试 ──
    # 原理：最佳匹配的距离应显著小于次佳（d1/d2 < 0.70），
    #       否则说明该点存在多个相似候选，匹配不可靠。
    good_matches = []
    for match_pair in raw_matches:
        if len(match_pair) != 2:
            continue
        first, second = match_pair
        if first.distance < config.lowe_ratio * second.distance:
            good_matches.append(first)

    if len(good_matches) < config.min_good_matches:
        raise AlignmentError("alignment: good matches below configured threshold")

    # ── 4) RANSAC 估计单应性矩阵 ──
    # 将匹配点坐标提取为 N×1×2 的 float32 数组
    scan_points = np.array(
        [scan_keypoints[m.queryIdx].pt for m in good_matches],
        dtype=np.float32,
    ).reshape(-1, 1, 2)
    template_points = np.array(
        [template_keypoints[m.trainIdx].pt for m in good_matches],
        dtype=np.float32,
    ).reshape(-1, 1, 2)

    homography, inlier_mask = _find_homography_deterministic(
        scan_points,
        template_points,
        reprojection_threshold=reprojection_threshold,
    )

    if homography is None or not np.isfinite(homography).all() or inlier_mask is None:
        raise AlignmentError("alignment: homography computation failed")

    inlier_ratio = float(inlier_mask.ravel().sum() / len(good_matches))
    return HomographyEstimate(
        homography=homography,
        good_matches=len(good_matches),
        inlier_ratio=inlier_ratio,
        detector=feature_detector.name,
        detector_fallback=feature_detector.fallback,
    )


def _create_feature_detector(config: PixelDiffConfig) -> FeatureDetector:
    requested = config.feature_detector
    try:
        return FeatureDetector(
            detector=_create_detector_by_name(requested, config),
            name=requested,
            fallback=False,
        )
    except AlignmentError:
        fallback = config.feature_detector_fallback
        if not fallback or fallback == requested:
            raise
        return FeatureDetector(
            detector=_create_detector_by_name(fallback, config),
            name=fallback,
            fallback=True,
        )


def _create_detector_by_name(name: str, config: PixelDiffConfig) -> Any:
    if name == "surf":
        return _create_surf_detector(config)
    if name == "sift":
        return _create_sift_detector(config)
    raise AlignmentError(f"alignment: unsupported feature detector {name!r}")


def _create_surf_detector(config: PixelDiffConfig) -> Any:
    """Create an OpenCV SURF detector, failing clearly when nonfree SURF is unavailable."""

    xfeatures2d = getattr(cv2, "xfeatures2d", None)
    surf_create = getattr(xfeatures2d, "SURF_create", None)
    if surf_create is None:
        raise AlignmentError(
            "alignment: SURF is unavailable in this OpenCV build; install an "
            "opencv-contrib-python build with xfeatures2d/SURF enabled"
        )
    try:
        return cast(Any, surf_create)(hessianThreshold=config.surf_hessian_threshold)
    except cv2.error as exc:
        raise AlignmentError(
            "alignment: SURF is unavailable because this OpenCV build was not compiled "
            "with nonfree algorithms enabled"
        ) from exc


def _create_sift_detector(config: PixelDiffConfig) -> Any:
    sift_create = getattr(cv2, "SIFT_create", None)
    if sift_create is None:
        raise AlignmentError("alignment: SIFT is unavailable in this OpenCV build")
    return cast(Any, sift_create)(nfeatures=config.sift_nfeatures)
