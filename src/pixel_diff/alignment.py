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

import math
from dataclasses import dataclass
from typing import Any, cast

import cv2
import numpy as np

from pixel_diff.exceptions import AlignmentError
from pixel_diff.models import PixelDiffConfig


@dataclass(frozen=True)
class HomographyDistortion:
    """把单应矩阵分解出的「几何变形」指标。

    用于判断配准是否把图像「拉歪」：
    - rotation_deg：配准残差旋转（度），对齐后图像应接近 0。
    - scale_x / scale_y：线性部分奇异值（缩放），接近 1 表示无拉伸。
    - shear_deg：剪切角（度），>0 表示被拉成平行四边形（表格线倾斜）。
    - anisotropy：各向异性比 = max(sx,sy)/min(sx,sy)，1 表示均匀缩放。
    """

    rotation_deg: float
    scale_x: float
    scale_y: float
    shear_deg: float
    anisotropy: float


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

    distortion: HomographyDistortion | None = None
    """单应矩阵分解出的几何变形指标；为 None 表示未计算（如空白页兜底）。"""

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
    distortion: HomographyDistortion | None = None


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


_HOMOGRAPHY_METHODS: dict[str, int] = {
    "ransac": cv2.RANSAC,
    "usac_magsac": getattr(cv2, "USAC_MAGSAC", cv2.RANSAC),
    "usac_default": getattr(cv2, "USAC_DEFAULT", cv2.RANSAC),
    "usac_accurate": getattr(cv2, "USAC_ACCURATE", cv2.RANSAC),
    "usac_fast": getattr(cv2, "USAC_FAST", cv2.RANSAC),
}


def _homography_method_code(name: str) -> int:
    """将配置里的方法名映射为 cv2.findHomography 的方法常量。

    未知方法回退到 USAC_MAGSAC（若可用）或 RANSAC。
    """
    return _HOMOGRAPHY_METHODS.get(name, _HOMOGRAPHY_METHODS["usac_magsac"])


def _decompose_homography(homography: np.ndarray) -> HomographyDistortion:
    """把 3×3 单应矩阵的线性部分分解成旋转/缩放/剪切指标。

    取单应矩阵左上 2×2 线性部分 A，用 QR 式分解：
        A = R(α) · [[scale_x, shear'], [0, scale_y]]
    - rotation_deg = α（配准残差旋转）
    - scale_x / scale_y = 列的缩放（含符号，符号仅表示翻转）
    - shear_deg = atan2(shear', scale_y)（剪切角）
    - anisotropy = max(|sx|,|sy|) / min(|sx|,|sy|)（各向异性拉伸比）

    对文档配准（近似仿射单应）这是「平均变形」的可靠近似；透视项 h31/h32
    在文档场景下接近 0，可忽略。
    """
    a = float(homography[0, 0])
    b = float(homography[0, 1])
    c = float(homography[1, 0])
    d = float(homography[1, 1])
    alpha = math.atan2(c, a)
    cos_a = math.cos(alpha)
    sin_a = math.sin(alpha)
    scale_x = a * cos_a + c * sin_a
    shear_prime = b * cos_a + d * sin_a
    scale_y = -b * sin_a + d * cos_a
    if abs(scale_y) > 1e-9:
        shear_deg = math.degrees(math.atan2(shear_prime, scale_y))
    else:
        shear_deg = 0.0
    sx = abs(scale_x)
    sy = abs(scale_y)
    anisotropy = max(sx, sy) / max(1e-9, min(sx, sy))
    return HomographyDistortion(
        rotation_deg=math.degrees(alpha),
        scale_x=scale_x,
        scale_y=scale_y,
        shear_deg=shear_deg,
        anisotropy=anisotropy,
    )


def _is_homography_distorted(
    distortion: HomographyDistortion | None,
    inlier_ratio: float,
    config: PixelDiffConfig,
) -> bool:
    """判定单应矩阵是否把图像「拉歪」（变形超标）或配准失败。

    触发条件（任一满足即视为「差异过大 / 疑似非同一文档」）：
    1. 内点率过低 → 特征匹配失败、配准本质失败；
    2. 剪切角超阈值 → 表格/文字被拉成平行四边形；
    3. 各向异性缩放超阈值 → 图像被横向/纵向拉伸；
    4. 残差旋转超阈值 → 图像整体被拉斜（配准未纠正对齐）。

    纯内容差异（改字、加行）但配准干净时，以上指标均在阈值内 → 返回 False。
    """
    if distortion is None:
        # 无法评估（如空白页兜底对齐），退化到仅看内点率，避免误杀
        return inlier_ratio < config.alignment_min_valid_inlier_ratio
    if inlier_ratio < config.alignment_min_valid_inlier_ratio:
        return True
    if abs(distortion.shear_deg) > config.alignment_max_shear_deg:
        return True
    if distortion.anisotropy > config.alignment_max_anisotropy:
        return True
    if abs(distortion.rotation_deg) > config.alignment_max_rotation_deg:
        return True
    return False



_ECC_MOTIONS: dict[str, int] = {
    "translation": cv2.MOTION_TRANSLATION,
    "euclidean": cv2.MOTION_EUCLIDEAN,
    "affine": cv2.MOTION_AFFINE,
    "homography": cv2.MOTION_HOMOGRAPHY,
}


def _refine_with_ecc(
    aligned_bgr: np.ndarray,
    template_bgr: np.ndarray,
    config: PixelDiffConfig,
) -> np.ndarray:
    """用 ECC（增强相关系数）对配准结果做亚像素级精修。

    在全局单应性配准之后，用 ECC 在模板与已对齐扫描件之间估计一个
    小幅度运动模型（默认平移+旋转），修正残留的亚像素偏移，再对
    已对齐图做二次 warp。ECC 失败时静默回退到原配准结果。

    Args:
        aligned_bgr: 已通过 warpPerspective 对齐到模板尺寸的扫描件。
        template_bgr: 模板原图（参照）。

    Returns:
        亚像素精修后的扫描件，尺寸与模板一致。
    """
    motion = _ECC_MOTIONS.get(config.ecc_motion_type, cv2.MOTION_EUCLIDEAN)
    aligned_gray = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    template_gray = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        config.ecc_max_iterations,
        config.ecc_epsilon,
    )
    try:
        if motion == cv2.MOTION_HOMOGRAPHY:
            warp_matrix: np.ndarray = np.eye(3, dtype=np.float32)
            _, warp_matrix = cv2.findTransformECC(
                template_gray, aligned_gray, warp_matrix, motion, criteria
            )
            refined = cv2.warpPerspective(
                aligned_bgr,
                warp_matrix,
                (template_bgr.shape[1], template_bgr.shape[0]),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(255, 255, 255),
            )
        else:
            warp_matrix = np.eye(2, 3, dtype=np.float32)
            _, warp_matrix = cv2.findTransformECC(
                template_gray, aligned_gray, warp_matrix, motion, criteria
            )
            refined = cv2.warpAffine(
                aligned_bgr,
                warp_matrix,
                (template_bgr.shape[1], template_bgr.shape[0]),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(255, 255, 255),
            )
        return refined
    except cv2.error:
        return aligned_bgr


def _detect_scan_border(gray: np.ndarray, config: PixelDiffConfig) -> tuple[int, int, int, int]:
    """检测扫描件外围的扫描仪边框条，返回可用区域 (x0, y0, x1, y1)。

    判定依据：扫描边框条（纸缘阴影、压板边缘）是灰度落在中灰区间
    ``[scan_border_gray_low, scan_border_gray_high]`` 的近连续长条；而正文
    文本行里墨迹是黑的（< gray_low）、纸面是白的（> gray_high），中灰占比
    很低。因此「某行/列中灰像素占比 > scan_border_line_ratio」即判为边框条。

    只在四边各自的外围 ``scan_border_max_ratio`` 范围内向内逐行/逐列搜索，
    一旦遇到非边框行/列立即停止，避免误剥页面内部的表格线或页眉分隔线。

    Returns:
        剥离边框后的有效区域 (x0, y0, x1, y1)，半开区间。无边框时返回全图。
    """
    height, width = gray.shape
    band = (gray >= config.scan_border_gray_low) & (gray <= config.scan_border_gray_high)
    row_ratio = band.mean(axis=1)
    col_ratio = band.mean(axis=0)
    max_y = int(height * config.scan_border_max_ratio)
    max_x = int(width * config.scan_border_max_ratio)
    threshold = config.scan_border_line_ratio

    y0 = 0
    while y0 < max_y and row_ratio[y0] > threshold:
        y0 += 1
    y1 = height
    while height - y1 < max_y and row_ratio[y1 - 1] > threshold:
        y1 -= 1
    x0 = 0
    while x0 < max_x and col_ratio[x0] > threshold:
        x0 += 1
    x1 = width
    while width - x1 < max_x and col_ratio[x1 - 1] > threshold:
        x1 -= 1

    # 退化保护：剥离后区域过小说明判定失控，放弃剥离
    if x1 - x0 < width // 2 or y1 - y0 < height // 2:
        return 0, 0, width, height
    return x0, y0, x1, y1


def _estimate_homography_border_stripped(
    scan_gray: np.ndarray,
    template_gray: np.ndarray,
    config: PixelDiffConfig,
) -> HomographyEstimate | None:
    """剥掉扫描件边框条后估计单应性，并把裁剪平移合成回原始扫描坐标。

    单应性最终仍把「原始扫描件坐标」映射到「模板原始坐标」，所以下游
    warp/差异框坐标系完全不变。无边框可剥或估计失败时返回 None，由调用方
    回退到原有逻辑。
    """
    x0, y0, x1, y1 = _detect_scan_border(scan_gray, config)
    if (x0, y0) == (0, 0) and (x1, y1) == (scan_gray.shape[1], scan_gray.shape[0]):
        return None

    scan_crop = scan_gray[y0:y1, x0:x1]
    try:
        estimate = _estimate_transform(
            scan_crop,
            template_gray,
            config,
            config.ransac_reprojection_threshold,
        )
    except AlignmentError:
        return None

    # 仿射回退：尺寸明显不同的扫描件与电子版之间通常是「缩放+旋转+平移」的仿射关系、
    # 几乎无透视。此时 8DOF 单应虽能拟合更多内点，但会引入不必要的透视项，使远离
    # 内点集中的区域（如页面底部）产生额外错位。单应透视项 h31/h32 接近 0（无透视）
    # 时改用仿射（6DOF），让整页对齐更均匀。仅在 border_stripped 路径（尺寸不同）生效，
    # 且要求两图尺寸差异明显（>= alignment_affine_min_size_ratio），否则两图并非缩放
    # 关系（如 e1f84156 尺寸比 1.0004），仿射因内点率低反而配不准。
    homography_crop = np.asarray(estimate.homography, dtype=np.float64)
    scan_h, scan_w = scan_gray.shape[:2]
    tmpl_h, tmpl_w = template_gray.shape[:2]
    size_ratio = max(scan_h / tmpl_h, tmpl_h / scan_h, scan_w / tmpl_w, tmpl_w / scan_w)
    if (
        config.alignment_prefer_affine
        and size_ratio >= config.alignment_affine_min_size_ratio
        and abs(float(homography_crop[2, 0])) < config.alignment_affine_perspective_threshold
        and abs(float(homography_crop[2, 1])) < config.alignment_affine_perspective_threshold
    ):
        try:
            affine_points, affine_template_points, affine_good, affine_det, affine_fb = (
                _match_features(scan_crop, template_gray, config)
            )
            affine, affine_mask = _estimate_affine_deterministic(
                affine_points,
                affine_template_points,
                reprojection_threshold=config.ransac_reprojection_threshold,
            )
        except AlignmentError:
            affine, affine_mask = None, None
        if affine is not None and np.isfinite(affine).all() and affine_mask is not None:
            affine_h = np.eye(3, dtype=np.float64)
            affine_h[:2] = affine
            estimate = HomographyEstimate(
                homography=affine_h,
                good_matches=affine_good,
                inlier_ratio=float(affine_mask.ravel().sum() / max(1, affine_good)),
                detector=affine_det,
                detector_fallback=affine_fb,
                distortion=_decompose_homography(affine_h),
            )

    # H_full = H_crop @ T_crop，其中 T_crop 把原始扫描坐标平移到裁剪坐标
    crop_translation = np.array(
        [[1.0, 0.0, -float(x0)], [0.0, 1.0, -float(y0)], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return HomographyEstimate(
        homography=np.asarray(estimate.homography @ crop_translation, dtype=np.float64),
        good_matches=estimate.good_matches,
        inlier_ratio=estimate.inlier_ratio,
        detector=estimate.detector,
        detector_fallback=estimate.detector_fallback,
        distortion=estimate.distortion,
    )


def _find_homography_deterministic(
    scan_points: np.ndarray,
    template_points: np.ndarray,
    *,
    reprojection_threshold: float,
    method: int = cv2.RANSAC,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Run homography estimation from a fixed OpenCV RNG state for reproducibility."""
    cv2.setRNGSeed(12345)
    homography, inlier_mask = cv2.findHomography(
        scan_points,
        template_points,
        method,
        reprojection_threshold,
    )
    return homography, inlier_mask


def _estimate_similarity_deterministic(
    scan_points: np.ndarray,
    template_points: np.ndarray,
    *,
    reprojection_threshold: float,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Robust similarity (uniform-scale + rotation + translation) estimation.

    Uses cv2.estimateAffinePartial2D with RANSAC. The returned 2×3 matrix is
    constrained to a similarity transform (equal x/y scale, no shear, no
    perspective), so warping never distorts tables or text geometry.
    """
    cv2.setRNGSeed(12345)
    matrix, inlier_mask = cv2.estimateAffinePartial2D(
        scan_points,
        template_points,
        method=cv2.RANSAC,
        ransacReprojThreshold=reprojection_threshold,
    )
    return matrix, inlier_mask


def _estimate_affine_deterministic(
    scan_points: np.ndarray,
    template_points: np.ndarray,
    *,
    reprojection_threshold: float,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Robust affine (scale + rotation + shear + translation) estimation.

    Uses cv2.estimateAffine2D with RANSAC. The returned 2×3 matrix allows
    independent x/y scale and shear (6 DOF) but no perspective. Suitable for
    scanned documents whose geometry is a near-pure affine mapping (scaling +
    rotation + translation without lens perspective).
    """
    cv2.setRNGSeed(12345)
    matrix, inlier_mask = cv2.estimateAffine2D(
        scan_points,
        template_points,
        method=cv2.RANSAC,
        ransacReprojThreshold=reprojection_threshold,
    )
    return matrix, inlier_mask


def _erase_table_lines(gray: np.ndarray, config: PixelDiffConfig) -> np.ndarray:
    """把表格横竖线从灰度图擦除（inpaint 成背景），供配准特征提取使用。

    表格横竖线是「长而重复」的结构，SIFT 易在不同位置之间产生歧义匹配；
    文字字形区分度高。本函数把长横线/长竖线 inpaint 成背景，使 SIFT 特征
    集中到文字上。仅在 ``alignment_ignore_table_lines=True`` 时生效。
    """
    if not config.alignment_ignore_table_lines:
        return gray
    min_len = int(config.alignment_table_line_min_length)
    if min_len <= 0:
        return gray

    ink = (gray < config.alignment_table_line_ink_threshold).astype(np.uint8)
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (min_len, 1))
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, min_len))
    lines_h = cv2.morphologyEx(ink, cv2.MORPH_OPEN, horiz_kernel)
    lines_v = cv2.morphologyEx(ink, cv2.MORPH_OPEN, vert_kernel)
    lines = cv2.bitwise_or(lines_h, lines_v)
    if cv2.countNonZero(lines) == 0:
        return gray

    # 外扩 2px 覆盖抗锯齿/描边，再 inpaint 成背景，避免残留硬边产生伪特征
    lines = cv2.dilate(lines, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    inpaint_mask = (lines * 255).astype(np.uint8)
    return cv2.inpaint(gray, inpaint_mask, 3, cv2.INPAINT_TELEA)


def _match_features(
    scan_gray: np.ndarray,
    template_gray: np.ndarray,
    config: PixelDiffConfig,
) -> tuple[np.ndarray, np.ndarray, int, str, bool]:
    """Detect features and run Lowe-ratio matching.

    Returns ``(scan_points, template_points, good_count, detector_name,
    detector_fallback)`` where point arrays are N×1×2 float32.
    """
    cv2.setRNGSeed(12345)
    feature_detector = _create_feature_detector(config)
    detector = feature_detector.detector
    scan_gray = _erase_table_lines(scan_gray, config)
    template_gray = _erase_table_lines(template_gray, config)
    scan_keypoints, scan_descriptors = detector.detectAndCompute(scan_gray, None)
    template_keypoints, template_descriptors = detector.detectAndCompute(template_gray, None)

    if scan_descriptors is None or template_descriptors is None:
        raise AlignmentError(f"alignment: {feature_detector.name.upper()} descriptors are empty")
    if (
        len(scan_keypoints) < config.min_good_matches
        or len(template_keypoints) < config.min_good_matches
    ):
        raise AlignmentError(f"alignment: not enough {feature_detector.name.upper()} keypoints")

    index_params: dict[str, bool | int | float | str] = {"algorithm": 1, "trees": 5}
    search_params: dict[str, bool | int | float | str] = {"checks": 50}
    matcher = cv2.FlannBasedMatcher(index_params, search_params)
    raw_matches = matcher.knnMatch(scan_descriptors, template_descriptors, k=2)

    good_matches = []
    for match_pair in raw_matches:
        if len(match_pair) != 2:
            continue
        first, second = match_pair
        if first.distance < config.lowe_ratio * second.distance:
            good_matches.append(first)

    if len(good_matches) < config.min_good_matches:
        raise AlignmentError("alignment: good matches below configured threshold")

    scan_points = np.array(
        [scan_keypoints[m.queryIdx].pt for m in good_matches],
        dtype=np.float32,
    ).reshape(-1, 1, 2)
    template_points = np.array(
        [template_keypoints[m.trainIdx].pt for m in good_matches],
        dtype=np.float32,
    ).reshape(-1, 1, 2)
    return scan_points, template_points, len(good_matches), feature_detector.name, feature_detector.fallback


def _similarity_to_homography(matrix: np.ndarray) -> np.ndarray:
    """Convert a 2×3 similarity matrix to a 3×3 homography for downstream warp."""
    homography = np.eye(3, dtype=np.float64)
    homography[:2] = matrix
    return homography


def _estimate_similarity(
    scan_gray: np.ndarray,
    template_gray: np.ndarray,
    config: PixelDiffConfig,
    reprojection_threshold: float,
) -> HomographyEstimate:
    """Estimate a similarity transform (no distortion) in the supplied coordinate system."""
    scan_points, template_points, good, detector, fallback = _match_features(
        scan_gray, template_gray, config
    )
    matrix, inlier_mask = _estimate_similarity_deterministic(
        scan_points,
        template_points,
        reprojection_threshold=reprojection_threshold,
    )
    if matrix is None or not np.isfinite(matrix).all() or inlier_mask is None:
        raise AlignmentError("alignment: similarity estimation failed")
    # 拒绝反射（det<0）：文档配准不应出现镜像
    if np.linalg.det(matrix[:, :2]) <= 0:
        raise AlignmentError("alignment: similarity estimate is a reflection")
    inlier_ratio = float(inlier_mask.ravel().sum() / good)
    return HomographyEstimate(
        homography=_similarity_to_homography(matrix),
        good_matches=good,
        inlier_ratio=inlier_ratio,
        detector=detector,
        detector_fallback=fallback,
    )


def _estimate_transform(
    scan_gray: np.ndarray,
    template_gray: np.ndarray,
    config: PixelDiffConfig,
    reprojection_threshold: float,
) -> HomographyEstimate:
    """配准矩阵分派：similarity-only 模式走相似变换（不变形），否则走自由单应矩阵。"""
    if config.alignment_similarity_only:
        return _estimate_similarity(scan_gray, template_gray, config, reprojection_threshold)
    return _estimate_homography(scan_gray, template_gray, config, reprojection_threshold)


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

    # 尺寸判断：两图尺寸完全相同时，内容位置/尺度基本一致，跳过边框剥离与
    # 归一化，直接走下方 SIFT 配准（避免预处理误剥内容、引入配准误差，导致
    # 真实差异漏报或误报）。只有尺寸不同（扫描件多了阴影/边距或缩放）时才做
    # 归一化/边框剥离，消除扫描阴影、边距与缩放差异。
    same_size = scan_bgr.shape[:2] == template_bgr.shape[:2]
    estimate = None
    if not same_size:
        # 边框剥离配准（优先）：扫描件外围的扫描仪边框条会吸引大量 SIFT 特征点，
        # 把单应性往错误方向拽。先剥掉边框条再估单应性，裁剪平移合成回原始坐标。
        estimate = (
            _estimate_homography_border_stripped(scan_gray, template_gray, config)
            if config.scan_border_strip_enabled
            else None
        )
        # 归一化配准：内容框裁剪 + 统一尺寸，消除边距/尺寸/宽高比差异后估单应性。
        # 归一化只用于估算，最终 warp 仍在原始坐标进行。归一化不可用时回退原逻辑。
        if estimate is None and config.alignment_normalize_enabled:
            estimate = _estimate_homography_normalized(scan_gray, template_gray, config)
    if estimate is not None:
        homography = estimate.homography
    elif (
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
            estimate = _estimate_transform(
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
            estimate = _estimate_transform(
                scan_gray,
                template_gray,
                config,
                config.ransac_reprojection_threshold,
            )
            homography = estimate.homography
            feature_scale = 1.0
            feature_downsample_fallback = True
    else:
        estimate = _estimate_transform(
            scan_gray,
            template_gray,
            config,
            config.ransac_reprojection_threshold,
        )
        homography = estimate.homography

    # 最终 warp 始终输出到模板原始尺寸 (template_width, template_height)，
    # 保证所有下游差异框坐标都在模板坐标系中。
    template_height, template_width = template_bgr.shape[:2]
    aligned = cv2.warpPerspective(
        scan_bgr,
        homography,
        (template_width, template_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    # 可选的 ECC 亚像素精修（默认关闭，实测对扫描件反而变差，详见配置注释）
    if config.ecc_refinement_enabled:
        aligned = _refine_with_ecc(aligned, template_bgr, config)
    # 分区配准（可选）：文字区/表格区各自独立估单应，互不干扰。
    # 全局单应常被特征点更多的表格区主导，导致文字区（上半）整体错位。
    # 分区配准成功时直接返回其对齐结果；不适用（无表格线/某区特征不足）则回退全局单应。
    if config.alignment_split_regions_enabled:
        split_result = _align_split_regions_bgr(
            scan_bgr, template_bgr, scan_gray, template_gray, config
        )
        if split_result is not None:
            return split_result
    return AlignmentResult(
        aligned_bgr=aligned,
        good_matches=estimate.good_matches,
        inlier_ratio=estimate.inlier_ratio,
        homography=homography,
        detector=estimate.detector,
        detector_fallback=estimate.detector_fallback,
        distortion=estimate.distortion,
        feature_downsampled=feature_downsampled,
        feature_scale=feature_scale,
        feature_downsample_fallback=feature_downsample_fallback,
    )


def _align_split_regions_bgr(
    scan_bgr: np.ndarray,
    template_bgr: np.ndarray,
    scan_gray: np.ndarray,
    template_gray: np.ndarray,
    config: PixelDiffConfig,
) -> AlignmentResult | None:
    """文字区 / 表格区分区独立配准；不适用时返回 ``None``（回退全局单应）。

    思路：扫描件的正文文字与下方表格常有不同的非线性形变，单一全局单应只能
    「折中」拟合，且常被特征点更多的表格区主导，导致文字区整体错位。本函数
    检测最上方一条长表格横线作为分界，把特征点分成文字区（上方）/表格区（下方），
    各自用本区特征点估单应，分别 warp 后按分界羽化融合，使两部分各自对齐、互不干扰。

    不适用场景（返回 None）：无长表格线、分界太靠边、某区特征点不足、单应估计失败。
    """
    min_len = int(config.alignment_split_table_line_min_length)
    if min_len <= 0:
        return None

    # 1. 检测 scan 的长横线和长竖线，取最上方一条横线作为文字区/表格区分界。
    #    要求横线、竖线**同时存在**（真正的表格 = 有横竖框），避免把正文里的
    #    下划线/分隔线（只有横线、无竖线）误当成分界，导致分区切错引入接缝假差异。
    ink = (scan_gray < config.alignment_table_line_ink_threshold).astype(np.uint8)
    horiz = cv2.morphologyEx(
        ink,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (min_len, 1)),
    )
    vert = cv2.morphologyEx(
        ink,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, min_len)),
    )
    line_rows = np.where(horiz.sum(axis=1) > 0)[0]
    line_cols = np.where(vert.sum(axis=0) > 0)[0]
    if len(line_rows) == 0 or len(line_cols) == 0:
        return None  # 无横线或无竖线 → 不是真正的表格，不分区
    y_split = int(line_rows.min())
    scan_h = scan_gray.shape[0]
    if y_split < scan_h * 0.15 or y_split > scan_h * 0.85:
        return None

    # 2. 特征点分区
    scan_points, template_points, good, detector, fallback = _match_features(
        scan_gray, template_gray, config
    )
    pts = scan_points.reshape(-1, 2)
    text_idx = pts[:, 1] < y_split
    table_idx = ~text_idx
    min_pts = max(config.min_good_matches, 8)
    if int(text_idx.sum()) < min_pts or int(table_idx.sum()) < min_pts:
        return None

    # 3. 各自估单应
    try:
        text_h, text_mask = _find_homography_deterministic(
            scan_points[text_idx],
            template_points[text_idx],
            reprojection_threshold=config.ransac_reprojection_threshold,
            method=_homography_method_code(config.homography_method),
        )
        table_h, table_mask = _find_homography_deterministic(
            scan_points[table_idx],
            template_points[table_idx],
            reprojection_threshold=config.ransac_reprojection_threshold,
            method=_homography_method_code(config.homography_method),
        )
        global_h, global_mask = _find_homography_deterministic(
            scan_points,
            template_points,
            reprojection_threshold=config.ransac_reprojection_threshold,
            method=_homography_method_code(config.homography_method),
        )
    except cv2.error:
        return None
    if text_h is None or table_h is None:
        return None

    # 3b. 分区价值判断：分区配准只在「文字区被全局单应明显牺牲」时才有价值。
    #     若文字区用全局单应本就能对齐（分区仅能把文字区内点率提升不足阈值），
    #     说明分区无必要 → 回退全局单应，避免表格区单应因特征点不足（内点率低）
    #     把表格线配出模糊/重影（实测 e1f84156：文字区全局内点率 0.731，分区后
    #     仅 0.727 无提升，但表格区单应内点率仅 0.477，导致表格线重影）。
    text_inlier = (
        float(text_mask.ravel().mean()) if text_mask is not None else 0.0
    )
    global_text_inlier = (
        float(global_mask.ravel()[text_idx].mean())
        if global_mask is not None and int(text_idx.sum()) > 0
        else 0.0
    )
    if text_inlier - global_text_inlier < config.alignment_split_min_text_inlier_gain:
        return None

    # 4. 分界映射到 template 系（用表格区单应映射 scan 的 (center_x, y_split)）
    center_x = scan_gray.shape[1] / 2.0
    src = np.array([[center_x, float(y_split), 1.0]], dtype=np.float64)
    mapped = table_h @ src.reshape(3, 1)
    split_t = float(mapped[1, 0] / max(mapped[2, 0], 1e-9))

    # 5. 分别 warp + 羽化融合
    th, tw = template_gray.shape
    warp_text = cv2.warpPerspective(
        scan_bgr,
        text_h,
        (tw, th),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    warp_table = cv2.warpPerspective(
        scan_bgr,
        table_h,
        (tw, th),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    band = float(config.alignment_split_blend_band)
    ys = np.arange(th, dtype=np.float32)
    if band <= 0:
        w = (ys >= split_t).astype(np.float32)
    else:
        w = np.clip((ys - (split_t - band)) / (2.0 * band), 0.0, 1.0)
    w = w.reshape(-1, 1, 1)
    fused = (
        warp_text.astype(np.float32) * (1.0 - w)
        + warp_table.astype(np.float32) * w
    ).astype(np.uint8)

    # 6. 指标：两区合并内点率；distortion 用表格区单应（特征更多、更稳）
    total_inlier = (
        int(text_mask.ravel().sum()) + int(table_mask.ravel().sum())
        if text_mask is not None and table_mask is not None
        else 0
    )
    inlier_ratio = total_inlier / good if good else 0.0

    return AlignmentResult(
        aligned_bgr=fused,
        good_matches=good,
        inlier_ratio=inlier_ratio,
        homography=table_h,
        detector=detector,
        detector_fallback=fallback,
        distortion=_decompose_homography(table_h),
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
    scan_points, template_points, good, detector, fallback = _match_features(
        scan_gray, template_gray, config
    )
    homography, inlier_mask = _find_homography_deterministic(
        scan_points,
        template_points,
        reprojection_threshold=reprojection_threshold,
        method=_homography_method_code(config.homography_method),
    )

    if homography is None or not np.isfinite(homography).all() or inlier_mask is None:
        raise AlignmentError("alignment: homography computation failed")

    inlier_ratio = float(inlier_mask.ravel().sum() / good)
    return HomographyEstimate(
        homography=homography,
        good_matches=good,
        inlier_ratio=inlier_ratio,
        detector=detector,
        detector_fallback=fallback,
        distortion=_decompose_homography(homography),
    )


def _content_bbox(
    gray: np.ndarray,
    threshold: int,
) -> tuple[int, int, int, int] | None:
    """检测灰度页的内容边界框 (x0, y0, x1, y1)，用于归一化裁剪白边。

    灰度低于 threshold 的像素视为内容（墨迹）。返回 None 表示页面无内容。
    """
    mask = gray < threshold
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _estimate_homography_normalized(
    scan_gray: np.ndarray,
    template_gray: np.ndarray,
    config: PixelDiffConfig,
) -> HomographyEstimate | None:
    """在内容归一化后的图上估计单应性，再映射回原始坐标系。

    步骤：
    1. 检测两图内容边界框，裁剪白边。
    2. 各向异性缩放到统一目标尺寸（消除边距/尺寸/宽高比差异）。
    3. 在归一化图上做 SIFT/FLANN/RANSAC 配准。
    4. 把单应性矩阵映射回原始像素坐标，使下游 warp 仍在模板原始
       坐标系进行（差异坐标不受影响）。

    返回 None 表示归一化不适用（无内容、宽高比差异过大或匹配失败），
    调用方回退到原图配准。
    """
    target_w, target_h = config.alignment_normalize_target_size
    if target_w <= 0 or target_h <= 0:
        return None

    scan_bbox = _content_bbox(scan_gray, config.alignment_normalize_ink_threshold)
    template_bbox = _content_bbox(template_gray, config.alignment_normalize_ink_threshold)
    if scan_bbox is None or template_bbox is None:
        return None

    sx0, sy0, sx1, sy1 = scan_bbox
    tx0, ty0, tx1, ty1 = template_bbox
    sw = sx1 - sx0
    sh = sy1 - sy0
    tw = tx1 - tx0
    th = ty1 - ty0
    if sw <= 0 or sh <= 0 or tw <= 0 or th <= 0:
        return None

    # 尺寸差异过小 → SIFT 尺度不变性已能处理，归一化反而引入误差，跳过
    size_ratio = max(sw / tw, tw / sw, sh / th, th / sh)
    if size_ratio < config.alignment_normalize_min_size_ratio:
        return None

    uniform = config.alignment_similarity_only
    if uniform:
        # 等比缩放（保长宽比）：两图内容裁剪后都缩放到同一最长边，
        # 不引入各向异性拉伸，故表格/文字几何不会变形。
        longest = float(max(target_w, target_h))
        scan_scale = longest / max(sw, sh)
        tmpl_scale = longest / max(tw, th)
        scan_norm = cv2.resize(
            scan_gray[sy0:sy1, sx0:sx1],
            (max(1, int(round(sw * scan_scale))), max(1, int(round(sh * scan_scale)))),
            interpolation=cv2.INTER_AREA,
        )
        template_norm = cv2.resize(
            template_gray[ty0:ty1, tx0:tx1],
            (max(1, int(round(tw * tmpl_scale))), max(1, int(round(th * tmpl_scale)))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        # 旧行为：各向异性缩放到统一目标尺寸（可能拉伸表格，仅非 similarity 模式使用）
        scan_aspect = sw / sh
        template_aspect = tw / th
        aspect_delta = max(scan_aspect, template_aspect) / min(scan_aspect, template_aspect)
        if aspect_delta > config.alignment_normalize_max_aspect_delta:
            return None
        scan_norm = cv2.resize(
            scan_gray[sy0:sy1, sx0:sx1],
            (target_w, target_h),
            interpolation=cv2.INTER_AREA,
        )
        template_norm = cv2.resize(
            template_gray[ty0:ty1, tx0:tx1],
            (target_w, target_h),
            interpolation=cv2.INTER_AREA,
        )

    # RANSAC 重投影阈值按归一化尺度放大，保持与原始像素尺度一致
    norm_scale = (scan_norm.shape[1] / sw + scan_norm.shape[0] / sh) / 2.0
    threshold_norm = config.ransac_reprojection_threshold * max(1.0, norm_scale)

    try:
        estimate = _estimate_transform(scan_norm, template_norm, config, threshold_norm)
    except AlignmentError:
        return None

    # 映射回原始坐标：H_full = T_template_inv @ H_norm @ T_scan
    # 等比模式下 scale_x == scale_y，保证最终变换仍是相似变换（不变形）。
    if uniform:
        sx = scan_scale
        sy = scan_scale
        stx = 1.0 / tmpl_scale
        sty = 1.0 / tmpl_scale
    else:
        sx = target_w / sw
        sy = target_h / sh
        stx = tw / target_w
        sty = th / target_h
    t_scan = np.array(
        [
            [sx, 0.0, -sx0 * sx],
            [0.0, sy, -sy0 * sy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    t_template_inv = np.array(
        [
            [stx, 0.0, tx0],
            [0.0, sty, ty0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    homography_full = t_template_inv @ estimate.homography @ t_scan

    return HomographyEstimate(
        homography=homography_full,
        good_matches=estimate.good_matches,
        inlier_ratio=estimate.inlier_ratio,
        detector=estimate.detector,
        detector_fallback=estimate.detector_fallback,
        distortion=_decompose_homography(homography_full),
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
