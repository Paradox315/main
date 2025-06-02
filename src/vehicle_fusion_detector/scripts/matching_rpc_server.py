#!/usr/bin/env python
import os
import sys
from typing import List, Optional, Tuple

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from scipy.optimize import linear_sum_assignment

# 添加当前脚本目录到Python路径
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

try:
    from graph_matching import (
        convert_detections_to_preds,
        graph_matching_assignment,
    )

    GRAPH_MATCHING_AVAILABLE = True
except ImportError as e:
    print(f"Graph matching module not available: {e}")
    GRAPH_MATCHING_AVAILABLE = False

app = FastAPI(title="Detection Matching RPC Service")


# 数据模型定义 - 与ROS消息结构保持一致
class Point(BaseModel):
    x: float
    y: float
    z: float


class Detection(BaseModel):
    object_id: str  # 这一帧中目标检测目标的序号id
    type: str  # 类型
    confidence: float  # 置信度
    box_2d: List[int]  # 2D边界框 [x_min, y_min, x_max, y_max] - 与ROS消息中int32[4]对应
    position: Point  # 3D坐标 (x, y, z)
    bbox_3d: List[float]  # 3D边界框 - 与ROS消息中float32[6]对应


class MatchingRequest(BaseModel):
    own_detections: List[Detection]
    other_detections: List[Detection]
    other_vehicle_id: str
    use_iou_for_cost: bool = True
    iou_threshold: float = 0.3
    distance_threshold: float = 2.0
    prefilter_max_distance: float = 10.0


class MatchingResult(BaseModel):
    matches: List[Tuple[int, int]]  # [(own_idx, other_idx), ...]
    method_used: str
    success: bool
    error_message: Optional[str] = None


class MatchingRPCService:
    def __init__(self):
        pass

    def _calculate_iou(self, boxA: List[int], boxB: List[int]) -> float:
        """计算两个边界框的IoU - 修改为处理int类型的box_2d"""
        xA = max(boxA[0], boxB[0])  # x_min
        yA = max(boxA[1], boxB[1])  # y_min
        xB = min(boxA[2], boxB[2])  # x_max
        yB = min(boxA[3], boxB[3])  # y_max

        interArea = max(0, xB - xA) * max(0, yB - yA)
        if interArea == 0:
            return 0.0

        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        if boxAArea <= 0 or boxBArea <= 0:
            return 0.0

        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou

    def _calculate_center_distance(self, boxA: List[int], boxB: List[int]) -> float:
        """计算两个边界框中心点的距离 - 修改为处理int类型的box_2d"""
        centerAx = (boxA[0] + boxA[2]) / 2.0
        centerAy = (boxA[1] + boxA[3]) / 2.0
        centerBx = (boxB[0] + boxB[2]) / 2.0
        centerBy = (boxB[1] + boxB[3]) / 2.0
        return np.sqrt((centerAx - centerBx) ** 2 + (centerAy - centerBy) ** 2)

    def _calculate_3d_distance(self, detA: Detection, detB: Detection) -> float:
        """计算两个检测的3D位置距离"""
        dx = detA.position.x - detB.position.x
        dy = detA.position.y - detB.position.y
        dz = detA.position.z - detB.position.z
        return np.sqrt(dx * dx + dy * dy + dz * dz)

    def graph_matching(self, request: MatchingRequest) -> MatchingResult:
        """图匹配方法"""
        try:
            if not GRAPH_MATCHING_AVAILABLE:
                return MatchingResult(
                    matches=[],
                    method_used="graph_matching",
                    success=False,
                    error_message="Graph matching module not available",
                )

            own_dets = request.own_detections
            other_dets = request.other_detections

            if len(own_dets) <= 1 or len(other_dets) <= 1:
                return MatchingResult(
                    matches=[],
                    method_used="graph_matching",
                    success=False,
                    error_message="Need more than 1 detection for graph matching",
                )

            # 转换为图匹配所需的格式
            ego_preds = convert_detections_to_preds(own_dets, detection_id_offset=0)
            cav_preds = convert_detections_to_preds(other_dets, detection_id_offset=0)

            if ego_preds.shape[0] == 0 or cav_preds.shape[0] == 0:
                return MatchingResult(
                    matches=[],
                    method_used="graph_matching",
                    success=False,
                    error_message="Empty predictions after conversion",
                )

            # 执行图匹配
            assignment = graph_matching_assignment(ego_preds, cav_preds)

            return MatchingResult(
                matches=assignment, method_used="graph_matching", success=True
            )

        except Exception as e:
            return MatchingResult(
                matches=[],
                method_used="graph_matching",
                success=False,
                error_message=str(e),
            )

    def traditional_matching(self, request: MatchingRequest) -> MatchingResult:
        """传统匹配方法"""
        try:
            own_dets = request.own_detections
            other_dets = request.other_detections
            num_own = len(own_dets)
            num_other = len(other_dets)

            if num_own == 0 or num_other == 0:
                return MatchingResult(
                    matches=[], method_used="traditional_matching", success=True
                )

            # 预筛选
            valid_own_indices = []
            valid_other_indices = []
            valid_pairs = []

            for i in range(num_own):
                for j in range(num_other):
                    # 类别检查
                    if own_dets[i].type != other_dets[j].type:
                        continue

                    # 距离检查
                    dist = self._calculate_center_distance(
                        own_dets[i].box_2d, other_dets[j].box_2d
                    )
                    if dist > request.prefilter_max_distance:
                        continue

                    valid_pairs.append((i, j))
                    if i not in valid_own_indices:
                        valid_own_indices.append(i)
                    if j not in valid_other_indices:
                        valid_other_indices.append(j)

            if not valid_pairs:
                return MatchingResult(
                    matches=[], method_used="traditional_matching", success=True
                )

            # 构建成本矩阵
            own_idx_map = {
                orig_idx: new_idx for new_idx, orig_idx in enumerate(valid_own_indices)
            }
            other_idx_map = {
                orig_idx: new_idx
                for new_idx, orig_idx in enumerate(valid_other_indices)
            }

            num_prefiltered_own = len(valid_own_indices)
            num_prefiltered_other = len(valid_other_indices)
            cost_matrix = np.full((num_prefiltered_own, num_prefiltered_other), np.inf)

            for orig_own_idx, orig_other_idx in valid_pairs:
                pf_own_idx = own_idx_map[orig_own_idx]
                pf_other_idx = other_idx_map[orig_other_idx]

                if request.use_iou_for_cost:
                    iou = self._calculate_iou(
                        own_dets[orig_own_idx].box_2d, other_dets[orig_other_idx].box_2d
                    )
                    cost_matrix[pf_own_idx, pf_other_idx] = (
                        1.0 - iou if iou > 0 else np.inf
                    )
                else:
                    distance = self._calculate_center_distance(
                        own_dets[orig_own_idx].box_2d, other_dets[orig_other_idx].box_2d
                    )
                    cost_matrix[pf_own_idx, pf_other_idx] = distance

            if np.all(np.isinf(cost_matrix)):
                return MatchingResult(
                    matches=[], method_used="traditional_matching", success=True
                )

            # 执行匈牙利算法
            row_ind_pf, col_ind_pf = linear_sum_assignment(cost_matrix)

            # 过滤有效匹配
            valid_matches = []
            for r_pf_idx, c_pf_idx in zip(row_ind_pf, col_ind_pf):
                cost = cost_matrix[r_pf_idx, c_pf_idx]
                if np.isinf(cost):
                    continue

                original_own_idx = valid_own_indices[r_pf_idx]
                original_other_idx = valid_other_indices[c_pf_idx]

                is_match_valid = False
                if request.use_iou_for_cost:
                    iou_val = 1.0 - cost
                    if iou_val >= request.iou_threshold:
                        is_match_valid = True
                else:
                    if cost <= request.distance_threshold:
                        is_match_valid = True

                if is_match_valid:
                    valid_matches.append((original_own_idx, original_other_idx))

            return MatchingResult(
                matches=valid_matches, method_used="traditional_matching", success=True
            )

        except Exception as e:
            return MatchingResult(
                matches=[],
                method_used="traditional_matching",
                success=False,
                error_message=str(e),
            )


# 全局服务实例
matching_service = MatchingRPCService()


@app.post("/graph_matching", response_model=MatchingResult)
async def graph_matching_endpoint(request: MatchingRequest):
    """图匹配API端点"""
    return matching_service.graph_matching(request)


@app.post("/traditional_matching", response_model=MatchingResult)
async def traditional_matching_endpoint(request: MatchingRequest):
    """传统匹配API端点"""
    return matching_service.traditional_matching(request)


@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {"status": "healthy", "graph_matching_available": GRAPH_MATCHING_AVAILABLE}


if __name__ == "__main__":
    import argparse
    import sys

    # 过滤掉ROS相关的命令行参数
    filtered_args = []
    for arg in sys.argv[1:]:
        if not arg.startswith("__") and not arg.startswith("_"):
            filtered_args.append(arg)

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")

    # 只解析过滤后的参数
    args = parser.parse_args(filtered_args)

    print(f"Starting RPC server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
