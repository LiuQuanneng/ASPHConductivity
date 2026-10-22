import numpy as np
import os
import pandas as pd
import math
import re
from tqdm import tqdm
from multiprocessing import Pool
from ripser import ripser  # 拓扑计算核心库

# ============================== 1. 配置参数（替代原config.py） ==============================
# 数据根目录（请根据实际路径修改！）
data_dir = os.environ.get("ASPH_DATA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "structure_inputs"))
# 截断距离（拓扑计算的邻域截断）
cut = 8.0
# 是否启用多进程
USE_MULTIPROCESS = True
# 目标特征文件夹名称（最终输出组合特征）
fname = "feature_topo_compo"
# 原子向量数据类型定义（替代原structure.py中的dt）
dt = np.dtype([
    ('typ', 'U2'),  # 元素类型（字符串，最多2个字符）
    ('pos', 'f8', 3)  # 原子位置（3维浮点数）
])


# ============================== 2. 辅助函数：基础工具 ==============================
def get_id_list(data_dir):
    """从properties.txt读取所有物质的ID列表"""
    prop_path = os.path.join(data_dir, "properties.txt")
    print(f"查找的文件路径：{prop_path}")

    if not os.path.exists(prop_path):
        raise FileNotFoundError(f"properties.txt不存在：{prop_path}")

    with open(prop_path, 'r') as f:
        lines = f.read().splitlines()

    id_list = []
    for line in lines[0:]:  # 保留所有行（含表头）
        if not line.strip():
            continue
        id = line.split()[0]
        id_list.append(id)

    # 移除表头ID（如果properties.txt第一行是表头）
    if id_list and id_list[0].lower() in ["id", "icsd", "material_id"]:
        id_list = id_list[1:]

    return id_list


def split_list(all_id_list):
    """将ID列表拆分为10个子列表（适配多进程）"""
    id_list_splited = []
    step = math.ceil(len(all_id_list) / 10.0)
    for i in range(0, len(all_id_list), step):
        end = min(i + step, len(all_id_list))
        id_list_splited.append(all_id_list[i:end])
    return id_list_splited


def split_element(data_dir):
    """提取物质-元素映射（从结构文件第一行）"""
    pattern = re.compile("[A-Z]{1}[a-z]{0,1}")
    sub_dict = {}
    id_list = get_id_list(data_dir)

    for sub in id_list:
        struct_path = os.path.join(data_dir, "structure", f"{sub}.txt")
        if not os.path.exists(struct_path):
            print(f"警告：结构文件不存在 {struct_path}，跳过")
            continue
        with open(struct_path, 'r') as tmp:
            try:
                ls = tmp.read().splitlines()[0]
            except IndexError:
                print(f"警告：结构文件为空 {struct_path}，跳过")
                continue
        eles = set(pattern.findall(ls))
        sub_dict[sub] = eles
    return sub_dict


def atoms_frequency(data_dir):
    """统计数据集元素出现频率"""
    sub_dict = split_element(data_dir)
    atom_single = {}
    for sub in sub_dict.keys():
        for ele in sub_dict[sub]:
            atom_single[ele] = atom_single.get(ele, 0.0) + 1.0
    return atom_single


# ============================== 3. 结构处理函数（增加文件存在性检查） ==============================
def get_prim_structure_info(data_dir, id):
    """解析POSCAR格式结构文件，提取晶格向量和原子信息（跳过已有文件）"""
    save_path = os.path.join(data_dir, "atoms", f"{id}_original.npz")
    # 检查：文件存在且非空则跳过
    if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
        print(f"✅ 物质 {id} 的原始原子数据已存在，跳过解析")
        return

    struct_path = os.path.join(data_dir, "structure", f"{id}.txt")
    if not os.path.exists(struct_path):
        raise FileNotFoundError(f"结构文件不存在：{struct_path}")

    with open(struct_path, 'r') as f:
        lines = f.read().splitlines()

    # 构建元素-原子数量映射
    atom_map = []
    index_up = 0
    try:
        for atom, num in zip(lines[5].split(), lines[6].split()):
            index_up += int(num)
            atom_map.append((atom, index_up))
    except IndexError:
        raise ValueError(f"结构文件格式错误（元素/原子数行缺失）：{struct_path}")

    # 提取晶格向量
    lattice_vec = []
    try:
        for line in lines[2:5]:
            x, y, z = line.split()
            lattice_vec.append([float(x), float(y), float(z)])
    except (IndexError, ValueError):
        raise ValueError(f"晶格向量解析失败：{struct_path}")
    lattice_vec = np.array(lattice_vec)

    # 提取原子位置和类型
    try:
        atom_nums = len(lines[8:])
        atom_vec = np.zeros([atom_nums], dtype=dt)
        index_atom = 0

        for i in range(atom_nums):
            line = lines[8 + i]
            x, y, z = line.split()
            # 分配元素类型
            if i < atom_map[index_atom][1]:
                atom_vec[i]['typ'] = atom_map[index_atom][0]
            else:
                index_atom += 1
                atom_vec[i]['typ'] = atom_map[index_atom][0]
            # 存储分数坐标
            atom_vec[i]['pos'][:] = np.array([float(x), float(y), float(z)])
    except (IndexError, ValueError):
        raise ValueError(f"原子坐标解析失败：{struct_path}")

    # 保存原始原子数据
    output_dir = os.path.join(data_dir, "atoms")
    os.makedirs(output_dir, exist_ok=True)
    with open(save_path, 'wb') as out_file:
        np.savez(out_file, lattice_vec=lattice_vec, atom_vec=atom_vec)


def enlarge_cell(data_dir, id):
    """放大晶胞（确保截断距离内原子完整）（跳过已有文件）"""
    save_path = os.path.join(data_dir, "atoms", f"{id}_enlarge.npz")
    # 检查：文件存在且非空则跳过
    if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
        print(f"✅ 物质 {id} 的放大晶胞数据已存在，跳过放大")
        return

    original_path = os.path.join(data_dir, "atoms", f"{id}_original.npz")
    if not os.path.exists(original_path):
        raise FileNotFoundError(f"原始原子数据不存在：{original_path}")

    with open(original_path, 'rb') as structfile:
        data = np.load(structfile)
        lattice_vec = data['lattice_vec']
        atom_vec = data['atom_vec']

    # 计算放大倍数
    min_lattice = min([np.linalg.norm(i) for i in lattice_vec])
    mul_time = int(np.ceil(cut / min_lattice))

    # 处理中心原子（笛卡尔坐标）
    center_atom_vec = atom_vec.copy()
    center_atom_vec['pos'][:] += mul_time
    center_atom_vec['pos'][:] = np.matmul(center_atom_vec['pos'][:], lattice_vec)

    # 生成放大后的原子
    enlarge_dict = {}
    atom_nums = 0
    for atom in atom_vec:
        typ = atom['typ']
        if typ not in enlarge_dict:
            enlarge_dict[typ] = []
        # 三维循环生成相邻原子
        for i in range(mul_time * 2 + 2):
            for j in range(mul_time * 2 + 2):
                for k in range(mul_time * 2 + 2):
                    point = [atom['pos'][0] + i, atom['pos'][1] + j, atom['pos'][2] + k]
                    if point not in enlarge_dict[typ]:
                        enlarge_dict[typ].append(point)
                        atom_nums += 1

    # 构建放大后的原子向量
    enlarge_vec = np.zeros([atom_nums], dtype=dt)
    cart_enlarge_vec = np.zeros([atom_nums], dtype=dt)
    atom_index = 0

    for typ, vec_list in enlarge_dict.items():
        for v in vec_list:
            enlarge_vec[atom_index]['typ'] = typ
            enlarge_vec[atom_index]['pos'][:] = np.array(v)
            cart_enlarge_vec[atom_index]['typ'] = typ
            cart_enlarge_vec[atom_index]['pos'][:] = np.matmul(np.array(v), lattice_vec)
            atom_index += 1

    # 保存放大后的数据
    os.makedirs(os.path.join(data_dir, "atoms"), exist_ok=True)
    with open(save_path, 'wb') as out_file:
        np.savez(out_file, CAV=center_atom_vec, CEV=cart_enlarge_vec)


# ============================== 4. 拓扑计算（Betti数）（增加文件存在性检查） ==============================
def get_betti_num(data_dir, id):
    """计算Betti数（Li-Li + 阴离子对），返回Li-Li距离列表（跳过已有文件）"""
    betti_save_path = os.path.join(data_dir, "betti_num", id)
    li_li_distances = []
    anions = ['S', 'O', 'P', 'N']

    # 检查：文件存在且非空则跳过
    if os.path.exists(betti_save_path) and os.path.getsize(betti_save_path) > 0:
        print(f"✅ 物质 {id} 的Betti数文件已存在，跳过计算")
        return li_li_distances

    # 读取放大原子数据
    enlarge_path = os.path.join(data_dir, "atoms", f"{id}_enlarge.npz")
    if not os.path.exists(enlarge_path):
        print(f"⚠️ 物质 {id} 放大原子数据不存在，跳过Betti数计算")
        return li_li_distances

    with open(enlarge_path, 'rb') as structfile:
        data = np.load(structfile)
        center_atom_vec = data['CAV']
        cart_enlarge_vec = data['CEV']

    # 创建输出目录
    os.makedirs(os.path.join(data_dir, "betti_num"), exist_ok=True)

    # 处理每个中心原子
    with open(betti_save_path, 'w') as out_File, tqdm(
            total=len(center_atom_vec),
            desc=f"处理{id}拓扑",
            unit="原子",
            colour="cyan",
            leave=False
    ) as pbar:

        for cav in center_atom_vec:
            center_atom_type = cav['typ']
            target_neighbors = []

            # 确定目标邻居类型
            if center_atom_type == 'Li':
                target_neighbors = ['Li']
            elif center_atom_type in anions:
                target_neighbors = anions
            else:
                pbar.update(1)
                continue

            # 遍历目标邻居类型
            for neighbor_type in target_neighbors:
                pair_index = []
                current_pair_distances = []

                # 筛选邻居原子
                for i, vec in enumerate(cart_enlarge_vec):
                    vec_typ = vec['typ']
                    if vec_typ != neighbor_type:
                        continue
                    dist = np.linalg.norm(vec['pos'][:] - cav['pos'][:])
                    if dist <= cut and not np.allclose(vec['pos'][:], cav['pos'][:]):
                        pair_index.append(i)
                        current_pair_distances.append(dist)

                if not pair_index:
                    continue

                # 记录Li-Li距离
                if center_atom_type == 'Li' and neighbor_type == 'Li':
                    li_li_distances.extend(current_pair_distances)

                # 计算Rips复形
                pair_pos = np.zeros((len(pair_index) + 1, 3))
                pair_pos[0][:] = cav['pos'][:]
                for idx, i in enumerate(pair_index, 1):
                    pair_pos[idx][:] = cart_enlarge_vec[i]['pos'][:]

                try:
                    dgms = ripser(pair_pos, maxdim=2, thresh=cut)['dgms']
                except Exception as e:
                    print(f"⚠️ {id} 拓扑计算失败：{e}")
                    continue

                # 保存条形码数据
                pair_label = f"{center_atom_type}{neighbor_type}"
                for dim, dgm in enumerate(dgms):
                    for p in dgm:
                        out_File.write(f"{pair_label} {dim} {p[0]} {p[1]}\n")

            pbar.update(1)

    return li_li_distances


def global_li_li_statistics(data_dir, all_ids):
    """全局Li-Li距离统计"""
    print("\n====== 全局Li-Li距离横向统计 ======")
    stats = []
    global_distances = []

    for id in tqdm(all_ids, desc="全局统计", unit="化合物", colour="green"):
        distances = get_betti_num(data_dir, id)
        global_distances.extend(distances)
        stats.append({
            "id": id,
            "count": len(distances),
            "mean": np.mean(distances) if distances else np.nan,
            "min": np.min(distances) if distances else np.nan,
            "max": np.max(distances) if distances else np.nan,
            "std": np.std(distances) if distances else np.nan,
            "distances": distances
        })

    # 保存统计结果
    stats_df = pd.DataFrame(stats)
    output_dir = os.path.join(data_dir, "statistics")
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "li_li_global_statistics.csv")
    stats_df.to_csv(csv_path, index=False, na_rep="NaN")

    # 打印汇总
    if global_distances:
        print("\n====== 全局Li-Li距离统计汇总 ======")
        print(f"总Li-Li原子对数量：{len(global_distances)}")
        print(f"平均距离：{np.mean(global_distances):.4f} Å")
        print(f"最小距离：{np.min(global_distances):.4f} Å")
        print(f"最大距离：{np.max(global_distances):.4f} Å")
        print(f"标准差：{np.std(global_distances):.4f} Å")
    else:
        print("\n⚠️ 未统计到Li-Li距离数据")
    print(f"\n详细统计已保存至：{csv_path}")

    return stats_df


# ============================== 5. 特征计算函数（增加文件存在性检查） ==============================
def get_feature_composition(data_dir, id):
    """计算成分特征（元素属性统计）（跳过已有文件）"""
    save_path = os.path.join(data_dir, "feature_composition", f"{id}_feature.npy")
    # 检查：文件存在且非空则跳过
    if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
        print(f"✅ 物质 {id} 的成分特征已存在，跳过计算")
        return

    enlarge_path = os.path.join(data_dir, "atoms", f"{id}_enlarge.npz")
    if not os.path.exists(enlarge_path):
        raise FileNotFoundError(f"放大原子数据不存在：{enlarge_path}")

    with open(enlarge_path, 'rb') as structfile:
        data = np.load(structfile)
        center_atom_vec = data['CAV']

    # 统计元素出现次数
    typ_dict = {}
    for vec in center_atom_vec:
        typ = vec['typ']
        typ_dict[typ] = typ_dict.get(typ, 0) + 1

    # 读取元素属性表
    prop_path = os.path.join(data_dir, "element_properties.csv")
    if not os.path.exists(prop_path):
        raise FileNotFoundError(f"元素属性表不存在：{prop_path}")
    element_properties = pd.read_csv(prop_path)
    element_properties.set_index('Abbr', inplace=True)

    # 构建属性数组
    tmp_array = []
    for ele, n in typ_dict.items():
        if ele not in element_properties.index:
            raise ValueError(f"元素 {ele} 不在属性表中")
        ele_list = list(element_properties.loc[ele])
        tmp_array.extend([ele_list for _ in range(n)])

    # 计算统计特征
    Feature = [
        np.mean(tmp_array, axis=0),
        np.std(tmp_array, axis=0),
        np.sum(tmp_array, axis=0),
        np.max(tmp_array, axis=0),
        np.min(tmp_array, axis=0)
    ]
    Feature_1 = np.concatenate(Feature, axis=0).astype(float)

    # 保存特征
    output_dir = os.path.join(data_dir, "feature_composition")
    os.makedirs(output_dir, exist_ok=True)
    with open(save_path, 'wb') as outfile:
        np.save(outfile, Feature_1)


def get_feature_with_s_nobin(data_dir, id):
    """计算拓扑特征（无分箱）（跳过已有文件）"""
    save_path = os.path.join(data_dir, "feature_add_s_nobin", f"{id}_feature.npy")
    # 检查：文件存在且非空则跳过
    if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
        print(f"✅ 物质 {id} 的拓扑特征已存在，跳过计算")
        return

    # 元素频率排序
    atom_single = atoms_frequency(data_dir)
    common_pair = {k: i for i, k in enumerate(sorted(atom_single, key=atom_single.get, reverse=True))}
    com_len = len(common_pair)

    # 读取原子数据
    enlarge_path = os.path.join(data_dir, "atoms", f"{id}_enlarge.npz")
    with open(enlarge_path, 'rb') as structfile:
        data = np.load(structfile)
        center_atom_vec = data['CAV']

    # 统计元素分布
    typ_dict = {}
    ele_Barcode = {}
    for vec in center_atom_vec:
        typ = vec['typ']
        typ_dict[typ] = typ_dict.get(typ, 0) + 1
        if typ in common_pair and typ not in ele_Barcode:
            ele_Barcode[typ] = [[], [], [], [], []]  # Bar0Death, Bar1Birth, Bar1Death, Bar2Birth, Bar2Death

    # 读取拓扑数据
    betti_path = os.path.join(data_dir, "betti_num", id)
    if not os.path.exists(betti_path):
        get_betti_num(data_dir, id)

    with open(betti_path, 'r') as phfile:
        lines = phfile.read().splitlines()

    # 解析条形码数据
    Bar0Death, Bar1Birth, Bar1Death, Bar2Birth, Bar2Death = [], [], [], [], []
    WBar0Death, WBar1Birth, WBar1Death, WBar2Birth, WBar2Death = [], [], [], [], []

    for line in lines:
        parts = line.split()
        if len(parts) != 4:
            continue
        typ, dim, birth, death = parts
        # 提取中心原子
        center_atom = typ[0:2] if len(typ) >= 2 and typ[1].islower() else typ[0]
        if center_atom not in typ_dict:
            continue
        ca_num = typ_dict[center_atom]

        try:
            dim = int(dim)
            birth = float(birth)
            death = float(death)
        except ValueError:
            continue

        # 出生时间
        if dim == 1:
            Bar1Birth.append(birth)
            WBar1Birth.append(birth / ca_num)
        elif dim == 2:
            Bar2Birth.append(birth)
            WBar2Birth.append(birth / ca_num)

        # 死亡时间（忽略inf）
        if np.isinf(death):
            continue
        if dim == 0:
            Bar0Death.append(death)
            WBar0Death.append(death / ca_num)
        elif dim == 1:
            Bar1Death.append(death)
            WBar1Death.append(death / ca_num)
        elif dim == 2:
            Bar2Death.append(death)
            WBar2Death.append(death / ca_num)

    # 转换为数组
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

    # 计算全局拓扑特征（Feature_2）
    Feature_2 = []
    # Betti0
    if len(Bar0Death) > 0:
        Feature_2.extend([
            np.mean(Bar0Death), np.std(Bar0Death), np.max(Bar0Death),
            np.min(Bar0Death), np.sum(WBar0Death)
        ])
    else:
        Feature_2.extend([0.] * 5)

    # Betti1
    if len(Bar1Death) > 0:
        dur1 = Bar1Death - Bar1Birth
        wdur1 = WBar1Death - WBar1Birth
        Feature_2.extend([
            # 持续时间
            np.mean(dur1), np.std(dur1), np.max(dur1), np.min(dur1), np.sum(wdur1),
            # 出生时间
            np.mean(Bar1Birth), np.std(Bar1Birth), np.max(Bar1Birth), np.min(Bar1Birth), np.sum(WBar1Birth),
            # 死亡时间
            np.mean(Bar1Death), np.std(Bar1Death), np.max(Bar1Death), np.min(Bar1Death), np.sum(WBar1Death)
        ])
    else:
        Feature_2.extend([0.] * 15)

    # Betti2
    if len(Bar2Death) > 0:
        dur2 = Bar2Death - Bar2Birth
        wdur2 = WBar2Death - WBar2Birth
        Feature_2.extend([
            # 持续时间
            np.mean(dur2), np.std(dur2), np.max(dur2), np.min(dur2), np.sum(wdur2),
            # 出生时间
            np.mean(Bar2Birth), np.std(Bar2Birth), np.max(Bar2Birth), np.min(Bar2Birth), np.sum(WBar2Birth),
            # 死亡时间
            np.mean(Bar2Death), np.std(Bar2Death), np.max(Bar2Death), np.min(Bar2Death), np.sum(WBar2Death)
        ])
    else:
        Feature_2.extend([0.] * 15)

    Feature_2 = np.array(Feature_2, float)

    # 计算按元素分组的拓扑特征（Feature_3）
    Feature_3 = np.zeros([com_len, 35], float)
    # 重新解析条形码到元素分组
    with open(betti_path, 'r') as phfile:
        lines = phfile.read().splitlines()

    for line in lines:
        parts = line.split()
        if len(parts) != 4:
            continue
        typ, dim, birth, death = parts
        center_atom = typ[0:2] if len(typ) >= 2 and typ[1].islower() else typ[0]
        if center_atom not in common_pair:
            continue
        ca_num = typ_dict.get(center_atom, 1)

        try:
            dim = int(dim)
            birth = float(birth)
            death = float(death)
        except ValueError:
            continue

        # 存储出生时间
        if dim == 1:
            ele_Barcode[center_atom][1].append(birth)
        elif dim == 2:
            ele_Barcode[center_atom][3].append(birth)
        # 存储死亡时间
        if np.isinf(death):
            continue
        if dim == 0:
            ele_Barcode[center_atom][0].append(death)
        elif dim == 1:
            ele_Barcode[center_atom][2].append(death)
        elif dim == 2:
            ele_Barcode[center_atom][4].append(death)

    # 计算元素拓扑特征
    for ele, bar in ele_Barcode.items():
        bar = [np.array(b, float) for b in bar]
        ca_num = typ_dict.get(ele, 1)
        ele_idx = common_pair[ele]

        # Bar0Death (0-4)
        if len(bar[0]) > 0:
            Feature_3[ele_idx, 0] = np.mean(bar[0])
            Feature_3[ele_idx, 1] = np.std(bar[0])
            Feature_3[ele_idx, 2] = np.max(bar[0])
            Feature_3[ele_idx, 3] = np.min(bar[0])
            Feature_3[ele_idx, 4] = np.sum(bar[0]) / ca_num

        # Bar1 持续时间 (5-9)、出生(10-14)、死亡(15-19)
        if len(bar[2]) > 0 and len(bar[1]) > 0:
            dur1 = bar[2] - bar[1]
            Feature_3[ele_idx, 5] = np.mean(dur1)
            Feature_3[ele_idx, 6] = np.std(dur1)
            Feature_3[ele_idx, 7] = np.max(dur1)
            Feature_3[ele_idx, 8] = np.min(dur1)
            Feature_3[ele_idx, 9] = np.sum(dur1) / ca_num

            Feature_3[ele_idx, 10] = np.mean(bar[1])
            Feature_3[ele_idx, 11] = np.std(bar[1])
            Feature_3[ele_idx, 12] = np.max(bar[1])
            Feature_3[ele_idx, 13] = np.min(bar[1])
            Feature_3[ele_idx, 14] = np.sum(bar[1]) / ca_num

            Feature_3[ele_idx, 15] = np.mean(bar[2])
            Feature_3[ele_idx, 16] = np.std(bar[2])
            Feature_3[ele_idx, 17] = np.max(bar[2])
            Feature_3[ele_idx, 18] = np.min(bar[2])
            Feature_3[ele_idx, 19] = np.sum(bar[2]) / ca_num

        # Bar2 持续时间 (20-24)、出生(25-29)、死亡(30-34)
        if len(bar[4]) > 0 and len(bar[3]) > 0:
            dur2 = bar[4] - bar[3]
            Feature_3[ele_idx, 20] = np.mean(dur2)
            Feature_3[ele_idx, 21] = np.std(dur2)
            Feature_3[ele_idx, 22] = np.max(dur2)
            Feature_3[ele_idx, 23] = np.min(dur2)
            Feature_3[ele_idx, 24] = np.sum(dur2) / ca_num

            Feature_3[ele_idx, 25] = np.mean(bar[3])
            Feature_3[ele_idx, 26] = np.std(bar[3])
            Feature_3[ele_idx, 27] = np.max(bar[3])
            Feature_3[ele_idx, 28] = np.min(bar[3])
            Feature_3[ele_idx, 29] = np.sum(bar[3]) / ca_num

            Feature_3[ele_idx, 30] = np.mean(bar[4])
            Feature_3[ele_idx, 31] = np.std(bar[4])
            Feature_3[ele_idx, 32] = np.max(bar[4])
            Feature_3[ele_idx, 33] = np.min(bar[4])
            Feature_3[ele_idx, 34] = np.sum(bar[4]) / ca_num

    # 合并特征
    Feature_3 = Feature_3.flatten()
    Feature = np.concatenate((Feature_2, Feature_3), axis=0)

    # 保存特征
    output_dir = os.path.join(data_dir, "feature_add_s_nobin")
    os.makedirs(output_dir, exist_ok=True)
    with open(save_path, 'wb') as outfile:
        np.save(outfile, Feature)


def get_feature_with_s_nobin_Bar0(data_dir, id):
    """简化拓扑特征（仅Bar0，占位实现）"""
    # 如需实现，参考get_feature_with_s_nobin，仅保留Betti0相关逻辑
    get_feature_with_s_nobin(data_dir, id)  # 临时复用完整拓扑特征


def get_feature_topo_compo(data_dir, id):
    """组合成分+拓扑特征（跳过已有文件）"""
    save_path = os.path.join(data_dir, "feature_topo_compo", f"{id}_feature.npy")
    # 检查：文件存在且非空则跳过
    if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
        print(f"✅ 物质 {id} 的组合特征已存在，跳过计算")
        return

    # 加载拓扑特征
    topo_path = os.path.join(data_dir, "feature_add_s_nobin", f"{id}_feature.npy")
    if not os.path.exists(topo_path):
        get_feature_with_s_nobin(data_dir, id)
    with open(topo_path, "rb") as f:
        feature_topo = np.load(f)

    # 加载成分特征
    compo_path = os.path.join(data_dir, "feature_composition", f"{id}_feature.npy")
    if not os.path.exists(compo_path):
        get_feature_composition(data_dir, id)
    with open(compo_path, "rb") as f:
        feature_compo = np.load(f)

    # 组合特征
    Feature = np.concatenate((feature_compo, feature_topo), axis=0)

    # 保存组合特征
    output_dir = os.path.join(data_dir, "feature_topo_compo")
    os.makedirs(output_dir, exist_ok=True)
    with open(save_path, 'wb') as outfile:
        np.save(outfile, Feature)


# ============================== 6. 数据加载（带特征命名） ==============================
def get_data_X_y(data_dir, fname):
    """加载特征和标签（容错版）：无标签列时仅生成ID+特征的CSV"""
    # 读取properties.txt
    prop_path = os.path.join(data_dir, "properties.txt")
    if not os.path.exists(prop_path):
        raise FileNotFoundError(f"properties.txt不存在：{prop_path}")

    with open(prop_path, 'r') as f:
        lines = f.read().splitlines()

    # 处理空文件情况
    if not lines:
        raise ValueError("properties.txt为空，无法读取ID列表")

    # ========== 修改1：弱化表头检查，自动检测是否有标签列 ==========
    header = lines[0].split()
    has_label_col = len(header) >= 2  # 判断是否有标签列
    target_col = 1 if has_label_col else None
    label_col_name = header[1] if has_label_col else "label"  # 默认标签列名

    X = []
    ids = []  # 移除y列表，改为按需添加
    id_list = get_id_list(data_dir)

    # 加载每个样本的特征（仅保留ID和特征，标签按需处理）
    for id in id_list:
        feature_path = os.path.join(data_dir, fname, f"{id}_feature.npy")
        if not os.path.exists(feature_path):
            print(f"警告：特征文件不存在 {feature_path}，跳过该样本")
            continue

        # 读取特征
        try:
            with open(feature_path, 'rb') as fe:
                feature = np.load(fe)
        except Exception as e:
            print(f"警告：读取特征文件 {feature_path} 失败：{e}，跳过该样本")
            continue

        # ========== 修改2：无标签列时直接保留ID和特征 ==========
        if not has_label_col:
            X.append(feature)
            ids.append(id)
            continue

        # 有标签列时，尝试读取标签（失败则仅保留ID和特征）
        target_value = None
        for line in lines:
            line_parts = line.strip().split()
            if len(line_parts) < 2:
                continue
            if line_parts[0] == id:  # 精准匹配ID
                try:
                    target_value = float(line_parts[1])
                except ValueError:
                    print(f"警告：ID {id} 的标签不是数值类型：{line_parts[1]}，仅保留特征")
                break

        # 无论是否有标签值，都保留ID和特征
        X.append(feature)
        ids.append(id)

    # 处理无有效数据情况
    if not X:
        raise ValueError("无有效特征数据，请检查特征文件和properties.txt")

    # 检查特征长度一致性（原有逻辑保留）
    feature_lengths = [len(x) for x in X]
    if len(set(feature_lengths)) > 1:
        print("警告：特征长度不一致！以下样本长度异常：")
        for id, length in zip(ids, feature_lengths):
            print(f"样本 {id}: 长度 {length}")
        # 取最长长度，补全短特征
        max_len = max(feature_lengths)
        X = [np.pad(x, (0, max_len - len(x)), mode='constant') if len(x) < max_len else x[:max_len] for x in X]

    X = np.array(X)

    # 生成有含义的特征名称（原有逻辑保留）
    prop_path = os.path.join(data_dir, "element_properties.csv")
    if not os.path.exists(prop_path):
        raise FileNotFoundError(f"元素属性表不存在：{prop_path}，请确认文件路径")

    elem_props = pd.read_csv(prop_path)
    if 'Abbr' not in elem_props.columns:
        raise ValueError("element_properties.csv必须包含'Abbr'列（元素缩写）")

    prop_names = elem_props.columns.drop('Abbr').tolist()
    compo_stats = ['mean', 'std', 'sum', 'max', 'min']
    compo_names = [f"{prop}_{stat}" for prop in prop_names for stat in compo_stats]

    global_topo_names = [
        "betti0_death_mean", "betti0_death_std", "betti0_death_max",
        "betti0_death_min", "betti0_death_sum_weighted",
        "betti1_duration_mean", "betti1_duration_std", "betti1_duration_max",
        "betti1_duration_min", "betti1_duration_sum_weighted",
        "betti1_birth_mean", "betti1_birth_std", "betti1_birth_max",
        "betti1_birth_min", "betti1_birth_sum_weighted",
        "betti1_death_mean", "betti1_death_std", "betti1_death_max",
        "betti1_death_min", "betti1_death_sum_weighted",
        "betti2_duration_mean", "betti2_duration_std", "betti2_duration_max",
        "betti2_duration_min", "betti2_duration_sum_weighted",
        "betti2_birth_mean", "betti2_birth_std", "betti2_birth_max",
        "betti2_birth_min", "betti2_birth_sum_weighted",
        "betti2_death_mean", "betti2_death_std", "betti2_death_max",
        "betti2_death_min", "betti2_death_sum_weighted"
    ]

    atom_freq = atoms_frequency(data_dir)
    sorted_elements = sorted(atom_freq.keys(), key=lambda x: atom_freq[x], reverse=True) if atom_freq else []
    element_topo_names = []
    for ele in sorted_elements:
        element_topo_names += [f"{ele}_{name}" for name in global_topo_names]

    all_feature_names = compo_names + global_topo_names + element_topo_names
    feature_total_len = X.shape[1]

    # 调整名称列表长度
    if len(all_feature_names) > feature_total_len:
        all_feature_names = all_feature_names[:feature_total_len]
    elif len(all_feature_names) < feature_total_len:
        all_feature_names += [f"unknown_feat_{i}" for i in range(feature_total_len - len(all_feature_names))]

    # ========== 修改3：构造DataFrame（按需添加标签列） ==========
    df = pd.DataFrame(X, columns=all_feature_names)
    df.insert(0, 'ID', ids)

    # 有标签列时尝试添加标签，无则跳过
    if has_label_col:
        # 重新读取标签值（仅为了赋值，不影响特征保留）
        y = []
        for id in ids:
            target_value = None
            for line in lines:
                line_parts = line.strip().split()
                if len(line_parts) >= 2 and line_parts[0] == id:
                    try:
                        target_value = float(line_parts[1])
                    except ValueError:
                        target_value = np.nan
                    break
            y.append(target_value if target_value is not None else np.nan)
        df[label_col_name] = y  # 添加标签列（缺失值为NaN）
        print(f"提示：已检测到标签列 {label_col_name}，缺失标签的样本将填充为NaN")
    else:
        print("提示：未检测到标签列，仅生成ID+特征的CSV文件")

    # 保存CSV
    csv_filename = f"{fname}_combined_features_named_Select.csv" if has_label_col else f"{fname}_features_only.csv"
    csv_path = os.path.join(data_dir, csv_filename)
    try:
        df.to_csv(csv_path, index=False, encoding='utf-8-sig', na_rep="NaN")
    except Exception as e:
        df.to_csv(csv_path, index=False, encoding='gbk', na_rep="NaN")
        print(f"提示：UTF-8编码保存失败，已改用GBK编码：{e}")

    print(f"✅ 特征文件已保存至：{csv_path}")
    print(f"✅ 有效样本数：{len(ids)}，特征维度：{feature_total_len}")

    # ========== 修改4：返回值适配（无标签时y为None） ==========
    y = df[label_col_name].values if has_label_col else None
    return X, y


# ============================== 7. 批量处理函数 ==============================
def process_all_features(data_dir, ids):
    """批量计算特征（带进度）"""
    for id in tqdm(ids, desc="批量计算特征", unit="物质", colour="green"):
        try:
            get_feature_topo_compo(data_dir, id)
        except Exception as e:
            print(f"\n处理物质 {id} 出错: {e}")
    print("\n🎉 特征计算完成！")


def batch_process(data_dir, all_ids):
    """批量处理结构+晶胞+拓扑"""
    # 1. 结构解析和晶胞放大
    print("====== 结构解析 & 晶胞放大 ======")
    for id in tqdm(all_ids, desc="结构处理", unit="化合物", colour="blue"):
        try:
            get_prim_structure_info(data_dir, id)
            enlarge_cell(data_dir, id)
        except Exception as e:
            print(f"\n处理 {id} 结构失败: {e}")

    # 2. 全局Li-Li统计（取消注释以启用）
    # global_li_li_statistics(data_dir, all_ids)


def batch_handle(id_list):
    """单进程批量处理子列表"""
    for id in id_list:
        try:
            # 处理结构和晶胞（已有文件会自动跳过）
            get_prim_structure_info(data_dir, id)
            enlarge_cell(data_dir, id)

            # 处理拓扑计算（已有文件会自动跳过）
            get_betti_num(data_dir, id)

            # 处理特征计算（已有文件会自动跳过）
            get_feature_topo_compo(data_dir, id)
        except Exception as e:
            print(f"\n处理 {id} 失败: {e}")


# ============================== 8. 特征函数映射 ==============================
func_map = {
    "feature_topo_compo": get_feature_topo_compo,
    "feature_add_s_nobin": get_feature_with_s_nobin,
    "feature_add_s_nobin_Bar0": get_feature_with_s_nobin_Bar0,
    "feature_composition": get_feature_composition
}


# ============================== 9. 主函数 ==============================
def main():
    print("====== 开始材料特征提取流程 ======")
    # 1. 获取ID列表
    id_list = get_id_list(data_dir)
    print(f"✅ 加载到 {len(id_list)} 个物质ID")

    # 2. 批量预处理（结构+晶胞+拓扑）
    batch_process(data_dir, id_list)

    # 3. 特征计算（单/多进程）
    print("\n====== 特征计算 ======")
    if USE_MULTIPROCESS:
        print("⚡ 多进程模式（10进程）")
        pool = Pool(10)
        splited_lists = split_list(id_list)
        pool.map(batch_handle, splited_lists)
        pool.close()
        pool.join()
    else:
        print("🔄 单进程模式")
        batch_handle(id_list)

    # 4. 生成带命名的特征CSV
    try:
        get_data_X_y(data_dir, fname)
    except Exception as e:
        print(f"\n❌ 生成特征CSV失败：{e}")
        raise

    print("\n====== 所有流程执行完成！======")


if __name__ == '__main__':
    # 检查必要文件夹
    required_dirs = [
        os.path.join(data_dir, "structure"),
        os.path.join(data_dir, "atoms"),
        os.path.join(data_dir, "betti_num"),
        os.path.join(data_dir, "feature_composition"),
        os.path.join(data_dir, "feature_add_s_nobin"),
        os.path.join(data_dir, "feature_topo_compo")
    ]
    for dir_path in required_dirs:
        os.makedirs(dir_path, exist_ok=True)

    # 执行主流程（增加全局异常捕获）
    try:
        main()
    except Exception as e:
        print(f"\n❌ 程序执行失败：{e}")
        exit(1)