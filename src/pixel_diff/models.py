"""类型化数据模型与配置验证。

定义：
- PixelDiffConfig   — 算法全量参数，支持 YAML 加载和运行时校验
- DifferenceRegion  — 单个疑似差异区域（id + 外接矩形 + 面积）
- PixelDiffResult   — 单页比对结果（状态 + 差异列表 + 诊断指标）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeAlias

import yaml

from pixel_diff.exceptions import ConfigurationError

# 类型别名
KernelSize: TypeAlias = tuple[int, int]
"""形态学核尺寸 (宽, 高)。"""

HSVRange: TypeAlias = tuple[tuple[int, int, int], tuple[int, int, int]]
"""HSV 范围 ((h_low, s_low, v_low), (h_high, s_high, v_high))。"""


@dataclass(frozen=True)
class PixelDiffConfig:
    """算法全量可配置参数，使用 frozen dataclass 确保不可变性。

    所有参数均有默认值（300 DPI 下的经验调优值），
    支持通过 YAML 文件覆盖部分参数。
    """

    # ─── 渲染 ───
    dpi: int = 300
    """PDF 渲染精度（Dots Per Inch），影响输出图像分辨率。"""

    # ─── 颜色过滤（去除红章蓝签）───
    filter_colored_marks: bool = True
    """是否启用 HSV 色彩过滤去除红色公章和蓝色手写签名。"""

    red_hsv_ranges: tuple[HSVRange, ...] = (
        ((0, 40, 40), (10, 255, 255)),
        ((170, 40, 40), (180, 255, 255)),
    )
    """红色 HSV 范围：色环两端，H∈[0,10]∪[170,180]，S≥40，V≥40。"""

    blue_hsv_ranges: tuple[HSVRange, ...] = (((100, 40, 40), (124, 255, 255)),)
    """蓝色 HSV 范围：H∈[100,124]，S≥40，V≥40。"""

    # ─── SIFT 配准 ───
    sift_nfeatures: int = 10000
    """SIFT 最大特征点数量。扫描件设较大值以保证噪声环境下的匹配量。"""

    lowe_ratio: float = 0.70
    """Lowe 比率测试阈值：d1/d2 < 0.70 才保留匹配。"""

    min_good_matches: int = 15
    """通过 Lowe 测试的最少匹配点数，不足则抛出 AlignmentError。"""

    ransac_reprojection_threshold: float = 3.0
    """RANSAC 重投影误差阈值（像素），判定内点/外点的距离上限。"""

    # ─── 差异检测 ───
    crop_margin: int = 40
    """边缘裁剪宽度（像素），抑制扫描仪边框伪影。"""

    min_diff_area: float = 200.0
    """差异区域最小面积（像素），面积不足的不计入结果。"""

    # ─── 局部相似性过滤 ───
    local_similarity_filter: bool = True
    """是否启用局部相似性多级过滤（主开关）。"""

    local_similarity_iou_threshold: float = 0.62
    """通用 IoU 阈值：局部最佳 IoU ≥ 0.62 的视为配准残余，过滤为背景。"""

    local_similarity_padding: int = 8
    """局部 IoU 搜索时向外扩展的 padding 像素数。"""

    local_similarity_search_radius: int = 4
    """局部 IoU 平移搜索半径（±4px，共 81 次尝试）。"""

    # ─── 局部相似性：水平残差过滤 ───
    horizontal_residual_min_aspect: float = 12.0
    """长水平残差过滤：最小宽高比。"""

    horizontal_residual_max_height: int = 20
    """长水平残差过滤：最大高度（像素）。"""

    short_horizontal_residual_min_aspect: float = 2.5
    """短水平残差过滤：最小宽高比。"""

    short_horizontal_residual_max_height: int = 20
    """短水平残差过滤：最大高度（像素）。"""

    short_horizontal_residual_min_iou: float = 0.55
    """短水平残差过滤：局部 IoU 最低阈值。"""

    wide_text_residual_min_area: float = 5000.0
    """宽文本残差过滤：最小面积（像素）。"""

    wide_text_residual_min_aspect: float = 3.0
    """宽文本残差过滤：最小宽高比。"""

    wide_text_residual_min_iou: float = 0.30
    """宽文本残差过滤：局部 IoU 最低阈值。"""

    # ─── 局部相似性：稀疏残差过滤 ───
    sparse_residual_max_area: float = 400.0
    """稀疏残差过滤 A：最大面积。"""

    sparse_residual_max_density: float = 0.04
    """稀疏残差过滤 A：最高局部前景密度。"""

    small_residual_max_area: float = 220.0
    """稀疏残差过滤 B（小残差）：最大面积。"""

    small_residual_max_density: float = 0.12
    """稀疏残差过滤 B（小残差）：最高局部前景密度。"""

    residual_filter_min_area: float = 200.0
    """稀疏残差过滤的通用最小面积门槛。"""

    residual_density_padding: int = 40
    """局部前景密度计算时的外扩 padding 像素数。"""

    # ─── 二值化 ───
    adaptive_block_size: int = 21
    """自适应阈值邻域大小（须为 >1 的奇数）。"""

    adaptive_c: int = 10
    """自适应阈值偏移量 C，从加权均值中减去的常数。"""

    median_blur_kernel: int = 3
    """中值滤波核大小（须为奇数，0=跳过）。"""

    bilateral_diameter: int = 9
    """双边滤波邻域直径。"""

    bilateral_sigma_color: float = 75.0
    """双边滤波灰度域标准差。越大则更多灰度差异被平滑。"""

    bilateral_sigma_space: float = 75.0
    """双边滤波空间域标准差。越大则更远像素参与平滑。"""

    # ─── 形态学 ───
    min_noise_component_area: float = 12.0
    """小连通域删除阈值（像素面积），低于此值的组件视为噪声。"""

    open_kernel: KernelSize = (3, 3)
    close_kernel: KernelSize = (3, 3)
    dilate_kernel: KernelSize = (15, 10)
    """膨胀核尺寸（宽,高），横向 > 纵向适配横排中文。
    较大的核将碎片合并为块，便于后续过滤器命中。"""

    morph_iterations_open: int = 1
    morph_iterations_close: int = 1
    morph_iterations_dilate: int = 1
    """形态学操作的迭代次数。"""

    @classmethod
    def from_yaml(cls, path: str | Path) -> PixelDiffConfig:
        """从 YAML 文件加载配置并校验。

        自动将列表类型的核尺寸转为 tuple。
        """
        with Path(path).open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}

        # YAML 加载时 kernel 是列表，需要转 tuple 以匹配 KernelSize
        for key in ("open_kernel", "close_kernel", "dilate_kernel"):
            if key in raw:
                raw[key] = tuple(raw[key])

        config = cls(**raw)
        config.validate()
        return config

    def validate(self) -> None:
        """校验所有不依赖具体图像尺寸的参数。

        Raises:
            ConfigurationError: 任何参数值不合法时抛出。
        """
        if self.dpi <= 0:
            raise ConfigurationError("configuration: dpi must be positive")
        if not 0 < self.lowe_ratio < 1:
            raise ConfigurationError("configuration: lowe_ratio must be in (0, 1)")
        if self.min_good_matches < 4:
            raise ConfigurationError("configuration: min_good_matches must be at least 4")
        if self.ransac_reprojection_threshold <= 0:
            raise ConfigurationError(
                "configuration: ransac_reprojection_threshold must be positive"
            )
        if self.crop_margin < 0:
            raise ConfigurationError("configuration: crop_margin must be non-negative")
        if self.min_diff_area < 0:
            raise ConfigurationError("configuration: min_diff_area must be non-negative")
        if not 0 <= self.local_similarity_iou_threshold <= 1:
            raise ConfigurationError(
                "configuration: local_similarity_iou_threshold must be in [0, 1]"
            )
        if self.local_similarity_padding < 0:
            raise ConfigurationError(
                "configuration: local_similarity_padding must be non-negative"
            )
        if self.local_similarity_search_radius < 0:
            raise ConfigurationError(
                "configuration: local_similarity_search_radius must be non-negative"
            )
        if self.horizontal_residual_min_aspect < 0:
            raise ConfigurationError(
                "configuration: horizontal_residual_min_aspect must be non-negative"
            )
        if self.horizontal_residual_max_height < 0:
            raise ConfigurationError(
                "configuration: horizontal_residual_max_height must be non-negative"
            )
        if self.short_horizontal_residual_min_aspect < 0:
            raise ConfigurationError(
                "configuration: short_horizontal_residual_min_aspect must be non-negative"
            )
        if self.short_horizontal_residual_max_height < 0:
            raise ConfigurationError(
                "configuration: short_horizontal_residual_max_height must be non-negative"
            )
        if not 0 <= self.short_horizontal_residual_min_iou <= 1:
            raise ConfigurationError(
                "configuration: short_horizontal_residual_min_iou must be in [0, 1]"
            )
        if self.wide_text_residual_min_area < 0:
            raise ConfigurationError(
                "configuration: wide_text_residual_min_area must be non-negative"
            )
        if self.wide_text_residual_min_aspect < 0:
            raise ConfigurationError(
                "configuration: wide_text_residual_min_aspect must be non-negative"
            )
        if not 0 <= self.wide_text_residual_min_iou <= 1:
            raise ConfigurationError(
                "configuration: wide_text_residual_min_iou must be in [0, 1]"
            )
        if self.sparse_residual_max_area < 0:
            raise ConfigurationError(
                "configuration: sparse_residual_max_area must be non-negative"
            )
        if self.sparse_residual_max_density < 0:
            raise ConfigurationError(
                "configuration: sparse_residual_max_density must be non-negative"
            )
        if self.small_residual_max_area < 0:
            raise ConfigurationError(
                "configuration: small_residual_max_area must be non-negative"
            )
        if self.small_residual_max_density < 0:
            raise ConfigurationError(
                "configuration: small_residual_max_density must be non-negative"
            )
        if self.residual_filter_min_area < 0:
            raise ConfigurationError(
                "configuration: residual_filter_min_area must be non-negative"
            )
        if self.residual_density_padding < 0:
            raise ConfigurationError(
                "configuration: residual_density_padding must be non-negative"
            )
        if self.adaptive_block_size <= 1 or self.adaptive_block_size % 2 == 0:
            raise ConfigurationError(
                "configuration: adaptive_block_size must be an odd integer greater than 1"
            )
        if self.median_blur_kernel != 0 and (
            self.median_blur_kernel <= 1 or self.median_blur_kernel % 2 == 0
        ):
            raise ConfigurationError(
                "configuration: median_blur_kernel must be 0 or an odd integer greater than 1"
            )
        if self.min_noise_component_area < 0:
            raise ConfigurationError(
                "configuration: min_noise_component_area must be non-negative"
            )
        for name, kernel in (
            ("open_kernel", self.open_kernel),
            ("close_kernel", self.close_kernel),
            ("dilate_kernel", self.dilate_kernel),
        ):
            _validate_kernel(name, kernel)
        for name, value in (
            ("morph_iterations_open", self.morph_iterations_open),
            ("morph_iterations_close", self.morph_iterations_close),
            ("morph_iterations_dilate", self.morph_iterations_dilate),
        ):
            if value < 0:
                raise ConfigurationError(f"configuration: {name} must be non-negative")

    def validate_for_image(self, width: int, height: int) -> None:
        """校验依赖图像尺寸的参数（如 crop_margin 不能超过半幅图像）。

        应在加载图像后、开始比对前调用。
        """
        self.validate()
        if width <= 0 or height <= 0:
            raise ConfigurationError("configuration: image dimensions must be positive")
        if self.crop_margin >= width / 2 or self.crop_margin >= height / 2:
            raise ConfigurationError(
                "configuration: crop_margin must be smaller than half of image width and height"
            )


@dataclass(frozen=True)
class DifferenceRegion:
    """单个疑似像素级差异区域（模板页面坐标系）。

    坐标 (x, y) 为外接矩形左上角，width/height 为矩形宽高，
    均以模板页面像素为单位。
    """

    id: int
    """差异区域序号（从 1 开始，自上而下、自左而右排列）。"""

    x: int
    y: int
    """外接矩形左上角坐标（模板页面坐标系，像素）。"""

    width: int
    height: int
    """外接矩形宽高（像素）。"""

    area: float
    """差异区域的实际轮廓面积（像素），非外接矩形面积。"""

    def to_dict(self) -> dict[str, int | float]:
        """转为 JSON 可序列化的字典。"""
        return {
            "id": self.id,
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "area": self.area,
        }


@dataclass(frozen=True)
class PixelDiffResult:
    """单页比对完整结果。"""

    status: str
    """比对状态："completed" 或 "partial"。"""

    page: int
    """页面索引（0-based）。"""

    image: dict[str, int]
    """页面图像元信息 {"width": int, "height": int, "dpi": int}。"""

    differences: list[DifferenceRegion]
    """疑似差异区域列表（已排序、编号）。"""

    metrics: dict[str, int | float]
    """诊断指标 {"elapsed_ms", "good_matches", "inlier_ratio"}。"""

    visual_output_path: str | None = None
    """红框标注图输出路径（可选）。"""

    metadata: dict[str, Any] = field(default_factory=dict)
    """附加元数据（如 ghost 图路径、template 图路径等）。"""

    def to_dict(self) -> dict[str, Any]:
        """转为 JSON 可序列化的字典。"""
        return {
            "status": self.status,
            "page": self.page,
            "image": self.image,
            "differences": [region.to_dict() for region in self.differences],
            "metrics": self.metrics,
            "visual_output_path": self.visual_output_path,
            "metadata": self.metadata,
        }


def _validate_kernel(name: str, kernel: KernelSize) -> None:
    """校验形态学核尺寸：(宽, 高) 均为正整数。"""
    if len(kernel) != 2:
        raise ConfigurationError(f"configuration: {name} must contain width and height")
    width, height = kernel
    if width <= 0 or height <= 0:
        raise ConfigurationError(f"configuration: {name} dimensions must be positive")
