Header header
string own_vehicle_id

# 融合结果包含的检测数据
demoros2/DetectionsWithOdom own_detections_with_odom    # 自车检测结果
demoros2/DetectionsWithOdom[] other_detections_list     # 来自不同车辆的检测结果

# 目标关联信息：重命名为MatchResult
demoros2/MatchResult[] match_results

# 融合后的检测结果
demoros2/DetectionsWithOdom fused_detections
