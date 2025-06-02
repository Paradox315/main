#!/usr/bin/env python
import numpy as np
import pygmtools as pygm
from scipy import stats


def build_graph(preds):
    """
    构建图的邻接矩阵
    :param preds: shape:(n,9), 特征为(id,cls,score,x,y,z,l,w,h)
    其中id为检测目标的唯一标识符，cls为类别，score为置信度，x,y,z为检测框中心坐标，l,w,h为检测框的长宽高
    :return: graph: shape(n,n,5): 代理的图，特征为[dist, azimuth, cls_from, cls_to]
    其中 dist 为相对距离，azimuth 为相对角度，cls_from 和 cls_to 为类别
    """
    n = preds.shape[0]
    points = preds[:, 3:6]  # x, y, z
    cls = preds[:, 1]  # 类别

    # 计算距离矩阵
    dist_mat = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)

    # 计算方位角
    norms = np.linalg.norm(points, axis=1)
    dot_product = np.dot(points, points.T)
    norm_product = norms[:, None] * norms[None, :]
    # 避免除零和数值误差
    norm_product = np.maximum(norm_product, 1e-8)
    cos_azimuth = np.clip(dot_product / norm_product, -1.0, 1.0)
    azimuth = np.arccos(cos_azimuth)
    azimuth = np.nan_to_num(azimuth, nan=0.0)

    # 类别矩阵
    cls_from = np.tile(cls[:, None], (1, n))
    cls_to = np.tile(cls[None, :], (n, 1))

    # 堆叠所有特征
    graph = np.stack([dist_mat, azimuth, cls_from, cls_to], axis=-1)
    return graph


def build_conn_edge(graph):
    """
    构建连接矩阵和边矩阵
    :param graph: shape(n,n,features_len): 代理的图
    :return: conn: shape(m,2): 代理的连接矩阵
             edge: shape(m,features_len): 代理的边矩阵
    """
    # 找到非零距离的连接（排除自环）
    from_idx, to_idx = np.where((graph[:, :, 0] != 0) & (graph[:, :, 0] > 1e-6))
    conn = np.stack([from_idx, to_idx], axis=1)
    edge = graph[from_idx, to_idx]
    return conn, edge


def node_affinity_fn(preds1, preds2, lamda1=0.5, lamda2=0.1):
    """
    计算节点亲和力矩阵
    :param preds1: (n1, 9) 第一组预测结果
    :param preds2: (n2, 9) 第二组预测结果
    :param lamda1: 形状亲和力权重
    :param lamda2: 位置亲和力权重
    :return: affinity: shape(n1, n2): 节点亲和力矩阵
    """
    preds1 = preds1[0]
    preds2 = preds2[0]
    cls1, cls2 = preds1[:, 1].astype(bool), preds2[:, 1].astype(bool)
    pos1, pos2 = preds1[:, 3:6], preds2[:, 3:6]  # x, y, z
    shape1, shape2 = preds1[:, 6:9], preds2[:, 6:9]  # l, w, h

    # 检查两个节点是否为同一类别
    affinity1 = cls1[:, None] == cls2[None, :]

    # 计算形状亲和力
    shape_dist = np.linalg.norm(shape1[:, None, :] - shape2[None, :, :], axis=2)
    affinity2 = np.exp(-lamda1 * shape_dist)

    # 计算位置亲和力
    pos_dist = np.linalg.norm(pos1[:, None, :] - pos2[None, :, :], axis=2)
    affinity3 = np.exp(-lamda2 * pos_dist)

    # 计算联合亲和力
    mu1, mu2 = 0.5, 0.5
    affinity = affinity1.astype(float) * (mu1 * affinity2 + mu2 * affinity3)

    return affinity[None, :]


def edge_affinity_fn(edges1, edges2, lamda1=0.5, lamda2=0.1):
    """
    计算边亲和力矩阵
    :param edges1: shape(m1, 5): 第一组边特征 [dist, azimuth, cls_from, cls_to]
    :param edges2: shape(m2, 5): 第二组边特征 [dist, azimuth, cls_from, cls_to]
    :param lamda1: 距离亲和力权重
    :param lamda2: 角度亲和力权重
    :return: affinity: shape(m1, m2): 边亲和力矩阵
    """
    edges1 = edges1[0]
    edges2 = edges2[0]
    if edges1.shape[0] == 0 or edges2.shape[0] == 0:
        return np.zeros((1, edges1.shape[0], edges2.shape[0]))

    cls_edge1, cls_edge2 = edges1[:, 2:].astype(int), edges2[:, 2:].astype(int)
    dist1, dist2 = edges1[:, 0], edges2[:, 0]
    azu1, azu2 = edges1[:, 1], edges2[:, 1]

    # 检查两条边是否为同一类别
    cls_from_match = cls_edge1[:, 0][:, None] == cls_edge2[:, 0][None, :]  # (m1, m2)
    cls_to_match = cls_edge1[:, 1][:, None] == cls_edge2[:, 1][None, :]  # (m1, m2)
    affinity1 = cls_from_match & cls_to_match  # (m1, m2)

    # 计算距离亲和力
    dist_ratio = dist1[:, None] / (dist2[None, :] + 1e-8)
    affinity2 = np.exp(-lamda1 * (dist_ratio + 1e-2) ** 2)

    # 计算角度亲和力
    angle_diff = np.abs(np.sin(azu1[:, None]) - np.sin(azu2[None, :]))
    affinity3 = np.exp(-lamda2 * angle_diff)

    # 联合亲和力
    affinity = affinity1.astype(float) * np.mean(
        np.stack([affinity2, affinity3]), axis=0
    )

    return affinity[None, :]


def build_affinity_matrix(ego_preds, cav_preds):
    """
    构建图匹配的亲和力矩阵
    :param ego_preds: numpy数组，形状为(n1, 9)
    :param cav_preds: numpy数组，形状为(n2, 9)
    :return: K: 亲和力矩阵, n1: ego节点数, n2: cav节点数
    """
    ego_graph = build_graph(ego_preds)
    cav_graph = build_graph(cav_preds)

    n1, n2 = np.array([len(ego_preds)]), np.array([len(cav_preds)])

    conn1, edge1 = build_conn_edge(ego_graph)
    conn2, edge2 = build_conn_edge(cav_graph)

    K = pygm.utils.build_aff_mat(
        ego_preds,
        edge1,
        conn1,
        cav_preds,
        edge2,
        conn2,
        n1,
        None,
        n2,
        None,
        edge_aff_fn=edge_affinity_fn,
        node_aff_fn=node_affinity_fn,
    )
    print(f"Affinity matrix shape: {K.shape}")

    return K, n1, n2


def graph_matching_assignment(ego_preds, cav_preds, mathod="rrwm"):
    """
    基于亲和力矩阵进行图匹配分配
    :param K: 亲和力矩阵
    :param n1: 第一组节点数量
    :param n2: 第二组节点数量
    :param method: 匹配方法、
    :return: assignment: 匹配结果，形状为(min(n1,n2), 2)
    """
    K, n1, n2 = build_affinity_matrix(ego_preds, cav_preds)
    print(f"Affinity matrix shape: {K.shape}, n1: {n1}, n2: {n2}")
    X = pygm.rrwm(K, n1, n2)
    match = pygm.hungarian(X)  # 使用匈牙利算法进行匹配
    ego_ids, cav_ids = np.where(match == 1)
    dist = [
        np.linalg.norm(ego_preds[i][3:6] - cav_preds[j][3:6])
        for i, j in zip(ego_ids, cav_ids)
    ]
    affinities = stats.zscore(dist)
    # Create a boolean mask for values with abs(zscore) <= 2
    mask = np.abs(affinities) <= 2

    return ego_ids[mask], cav_ids[mask]


def convert_detections_to_preds(detections_list, detection_id_offset=0):
    """
    将ROS检测消息转换为预测数组格式
    :param detections_list: DetectionsWithOdom消息中的detections列表
    :param detection_id_offset: ID偏移量
    :return: numpy数组，形状为(n, 9)
    """
    if not detections_list:
        return np.empty((0, 9))

    preds = []
    for i, det in enumerate(detections_list):
        # 构建预测数组: [id, cls,score,x,y,z,l,w,h]
        x = det.position.x
        y = det.position.y
        z = det.position.z
        l = det.bbox_3d[4] - det.bbox_3d[0]  # 假设bbox_3d[0]是x_min, bbox_3d[4]是x_max
        w = det.bbox_3d[5] - det.bbox_3d[1]  # 假设bbox_3d[1]是y_min, bbox_3d[5]是y_max
        h = (
            det.bbox_3d[-1] - det.bbox_3d[2]
        )  # 假设bbox_3d[2]是z_min, bbox_3d[-1]是z_max

        # 类别编码（这里简化处理，可根据实际需求调整）
        cls = 1 if det.type == "person" else 0  # 假设只有两类：person和chair

        # 置信度作为属性概率
        score = det.confidence

        pred = [
            detection_id_offset + i,  # id
            cls,  # 类别
            score,  # 置信度
            x,
            y,
            z,  # 检测框中心坐标
            l,
            w,
            h,  # 检测框的长宽高
        ]
        preds.append(pred)

    return np.array(preds)
