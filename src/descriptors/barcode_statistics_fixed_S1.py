"""Audited barcode statistics; source hash: f387d17a50a8e1b846788b6d69bcaacdfa4913c51150eec470b59ea4765471fa"""
import os
import numpy as np
TRAINING_ELEMENTS=["Li","O","P","S"]
CUTOFF=8.0

def compute_betti_features(barcode_path, element_counts):
    """
    解析 barcode 文件，计算与训练完全一致的 175维 Betti 特征。
    175 = 35 全局 + 35×4 元素 (Li, O, P, S)

    barcode 文件每行格式: pair_label dim birth death
    例如: LiLi 0 0.0 3.085
    """
    Bar0Death, Bar1Birth, Bar1Death, Bar2Birth, Bar2Death = ([], [], [], [], [])
    WBar0Death, WBar1Birth, WBar1Death, WBar2Birth, WBar2Death = ([], [], [], [], [])
    ele_data = {e: [[], [], [], [], []] for e in TRAINING_ELEMENTS}
    if not os.path.exists(barcode_path):
        return np.zeros(175, dtype=float)
    with open(barcode_path, 'r') as f:
        lines = f.read().splitlines()
    for line in lines:
        parts = line.split()
        if len(parts) != 4:
            continue
        pair_label, dim_str, birth_str, death_str = parts
        center_atom = pair_label[0:2] if len(pair_label) >= 2 and pair_label[1].islower() else pair_label[0]
        try:
            dim = int(dim_str)
            birth = float(birth_str)
            death = float(death_str)
        except ValueError:
            continue
        ca_num = element_counts.get(center_atom, 1)
        if dim == 1:
            Bar1Birth.append(birth)
            WBar1Birth.append(birth / ca_num)
        elif dim == 2:
            Bar2Birth.append(birth)
            WBar2Birth.append(birth / ca_num)
        if not np.isinf(death):
            if dim == 0:
                Bar0Death.append(death)
                WBar0Death.append(death / ca_num)
            elif dim == 1:
                Bar1Death.append(death)
                WBar1Death.append(death / ca_num)
            elif dim == 2:
                Bar2Death.append(death)
                WBar2Death.append(death / ca_num)
        if center_atom in ele_data:
            if dim == 1:
                ele_data[center_atom][1].append(birth)
            elif dim == 2:
                ele_data[center_atom][3].append(birth)
            if not np.isinf(death):
                if dim == 0:
                    ele_data[center_atom][0].append(death)
                elif dim == 1:
                    ele_data[center_atom][2].append(death)
                elif dim == 2:
                    ele_data[center_atom][4].append(death)
    Bar0Death = np.array(Bar0Death, float)
    Bar1Birth = np.array(Bar1Birth, float)
    Bar1Death = np.array(Bar1Death, float)
    Bar2Birth = np.array(Bar2Birth, float)
    Bar2Death = np.array(Bar2Death, float)
    WBar0Death = np.array(WBar0Death, float)
    WBar1Birth = np.array(WBar1Birth, float)
    WBar1Death = np.array(WBar1Death, float)
    WBar2Birth = np.array(WBar2Birth, float)
    WBar2Death = np.array(WBar2Death, float)
    global_features = []
    if len(Bar0Death) > 0:
        global_features.extend([np.mean(Bar0Death), np.std(Bar0Death), np.max(Bar0Death), np.min(Bar0Death), np.sum(WBar0Death)])
    else:
        global_features.extend([0.0] * 5)
    if len(Bar1Death) > 0:
        dur1 = Bar1Death - Bar1Birth
        wdur1 = WBar1Death - WBar1Birth
        global_features.extend([np.mean(dur1), np.std(dur1), np.max(dur1), np.min(dur1), np.sum(wdur1), np.mean(Bar1Birth), np.std(Bar1Birth), np.max(Bar1Birth), np.min(Bar1Birth), np.sum(WBar1Birth), np.mean(Bar1Death), np.std(Bar1Death), np.max(Bar1Death), np.min(Bar1Death), np.sum(WBar1Death)])
    else:
        global_features.extend([0.0] * 15)
    if len(Bar2Death) > 0:
        dur2 = Bar2Death - Bar2Birth
        wdur2 = WBar2Death - WBar2Birth
        global_features.extend([np.mean(dur2), np.std(dur2), np.max(dur2), np.min(dur2), np.sum(wdur2), np.mean(Bar2Birth), np.std(Bar2Birth), np.max(Bar2Birth), np.min(Bar2Birth), np.sum(WBar2Birth), np.mean(Bar2Death), np.std(Bar2Death), np.max(Bar2Death), np.min(Bar2Death), np.sum(WBar2Death)])
    else:
        global_features.extend([0.0] * 15)
    elem_features = []
    for ele in TRAINING_ELEMENTS:
        bar = [np.array(b, float) for b in ele_data[ele]]
        ca_num = element_counts.get(ele, 1)
        feats = np.zeros(35, float)
        if len(bar[0]) > 0:
            feats[0] = np.mean(bar[0])
            feats[1] = np.std(bar[0])
            feats[2] = np.max(bar[0])
            feats[3] = np.min(bar[0])
            feats[4] = np.sum(bar[0]) / ca_num
        if len(bar[2]) > 0 and len(bar[1]) > 0:
            dur1 = bar[2] - bar[1]
            feats[5] = np.mean(dur1)
            feats[6] = np.std(dur1)
            feats[7] = np.max(dur1)
            feats[8] = np.min(dur1)
            feats[9] = np.sum(dur1) / ca_num
            feats[10] = np.mean(bar[1])
            feats[11] = np.std(bar[1])
            feats[12] = np.max(bar[1])
            feats[13] = np.min(bar[1])
            feats[14] = np.sum(bar[1]) / ca_num
            feats[15] = np.mean(bar[2])
            feats[16] = np.std(bar[2])
            feats[17] = np.max(bar[2])
            feats[18] = np.min(bar[2])
            feats[19] = np.sum(bar[2]) / ca_num
        if len(bar[4]) > 0 and len(bar[3]) > 0:
            dur2 = bar[4] - bar[3]
            feats[20] = np.mean(dur2)
            feats[21] = np.std(dur2)
            feats[22] = np.max(dur2)
            feats[23] = np.min(dur2)
            feats[24] = np.sum(dur2) / ca_num
            feats[25] = np.mean(bar[3])
            feats[26] = np.std(bar[3])
            feats[27] = np.max(bar[3])
            feats[28] = np.min(bar[3])
            feats[29] = np.sum(bar[3]) / ca_num
            feats[30] = np.mean(bar[4])
            feats[31] = np.std(bar[4])
            feats[32] = np.max(bar[4])
            feats[33] = np.min(bar[4])
            feats[34] = np.sum(bar[4]) / ca_num
        elem_features.append(feats)
    all_features = np.concatenate([np.array(global_features)] + elem_features)
    return all_features
