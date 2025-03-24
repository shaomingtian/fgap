import numpy as np
import re
from typing import List, Dict, Tuple
from collections import defaultdict

import fgap_config

# --------------------------
# 单位转换模块
# --------------------------
UNIT_MAP = {
    'b': 1,
    'k': 1024,
    'kb': 1024,
    'm': 1024**2,
    'mb': 1024**2,
    'g': 1024**3,
    'gb': 1024**3,
    't': 1024**4,
    'tb': 1024**4
}

FSDIR_IDX_MAP = {}

def create_fs_tag_file(config_file, output_file):
    dirs = []

    i = 0
    # 读取配置文件
    with open(config_file, 'r') as f:
        for line in f:
            # 跳过以 '#' 开头的注释行
            if line.startswith('#'):
                continue

            # 分割行，并提取第二列
            parts = line.split()
            if len(parts) > 1:  # 确保第二列存在
                dir_path = parts[1]  # 第二列内容
                FSDIR_IDX_MAP[dir_path] = i
                i += 1
                # 如果路径不是以 '/' 结尾，则添加 '/'
                if not dir_path.endswith('/'):
                    dir_path += '/'
                dirs.append(dir_path)

    # 将结果写入输出文件
    with open(output_file, 'w') as f:
        for index, dir_path in enumerate(dirs):

            f.write(f"{index} {dir_path}\n")



def parse_size(size_str: str) -> float:
    """带单位转换的核心函数"""
    try:
        size_str = size_str.strip().lower()
        match = re.match(r"^([\d.]+)\s*([a-z]*)$", size_str)
        if not match:
            raise ValueError()

        value, unit = match.groups()
        numeric = float(value)

        if not unit:  # 无单位默认为字节
            return int(numeric)

        # 统一单位缩写
        unit = unit[0] if unit in ['k', 'm', 'g', 't', 'b'] else unit
        if unit not in UNIT_MAP:
            raise ValueError()

        return int(numeric * UNIT_MAP[unit])
    except:
        raise ValueError(f"无效的尺寸格式: {size_str}")

# --------------------------
# 配置文件解析模块
# --------------------------
def parse_fs_config(config_path: str) -> Tuple[Dict, Dict]:
    """解析文件系统配置"""
    fs_info = {
        'm': 0, 'BF': [], 'SF': [],
        'L': [], 'BO': [], 'SO': [], 'names': []
    }
    fs_map = {}

    with open(config_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split()
            if len(parts) < 6:
                continue

            try:
                fs_type, fs_dir = parts[0], parts[1]
                bw = parse_size(parts[-2])
                cap = parse_size(parts[-1])

                if (fs_type, fs_dir) in fs_map:
                    continue

                idx = len(fs_info['BF'])
                fs_map[(fs_type, fs_dir)] = idx
                fs_info['BF'].append(bw)
                fs_info['SF'].append(cap)
                fs_info['names'].append([fs_type, fs_dir])
            except Exception as e:
                print(f"Error parsing fs.config line {line_num}: {str(e)}")
                continue

    # 初始化状态字段
    m = len(fs_info['BF'])
    fs_info.update({
        'm': m,
        'L': [0]*m,
        'BO': [0]*m,
        'SO': [0]*m
    })
    return fs_info, fs_map

def parse_pred_file(pred_path: str, fs_map: Dict) -> Tuple[Dict, Dict]:
    """解析预测文件"""
    file_data = defaultdict(lambda: {
        'S': None, 'RS': None, 'WS': None,
        'TS': None, 'TE': None,
        'Bw': [0]*len(fs_map),
        'Br': [0]*len(fs_map)
    })

    with open(pred_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split()
            if len(parts) < 12:
                continue

            try:
                filename = parts[0]
                fs_type, fs_dir = parts[1], parts[2]
                fs_key = (fs_type, fs_dir)

                if fs_key not in fs_map:
                    continue
                fs_idx = fs_map[fs_key]

                # 解析所有数值字段
                pred_data = {
                    'Bw': parse_size(parts[5]),
                    'Br': parse_size(parts[6]),
                    'S': parse_size(parts[7]),
                    'RS': parse_size(parts[8]),
                    'WS': parse_size(parts[9]),
                    'TS': float(parts[10]),
                    'TE': float(parts[11])
                }

                # 更新基础信息
                if file_data[filename]['S'] is None:
                    for k in ['S', 'RS', 'WS', 'TS', 'TE']:
                        file_data[filename][k] = pred_data[k]

                # 更新性能数据
                file_data[filename]['Bw'][fs_idx] = pred_data['Bw']
                file_data[filename]['Br'][fs_idx] = pred_data['Br']

            except Exception as e:
                print(f"Error parsing pred.txt line {line_num}: {str(e)}")
                continue

    # 转换为标准格式
    files_info = {'n':0, 'S':[], 'RS':[], 'WS':[], 'TS':[], 'TE':[], 'filenames':[]}
    pred_perf = {'Bw':[], 'Br':[], 'B':[]}

    for filename, data in file_data.items():
        if any(v is None for v in [data['S'], data['RS'], data['WS'], data['TS'], data['TE']]):
            continue

        files_info['filenames'].append(filename)
        files_info['S'].append(data['S'])
        files_info['RS'].append(data['RS'])
        files_info['WS'].append(data['WS'])
        files_info['TS'].append(data['TS'])
        files_info['TE'].append(data['TE'])

        pred_perf['Bw'].append(data['Bw'])
        pred_perf['Br'].append(data['Br'])
        pred_perf['B'].append([0]*len(fs_map))

    files_info['n'] = len(files_info['filenames'])
    return files_info, pred_perf

# --------------------------
# ADO 算法核心实现
# --------------------------
class AdaptiveDataOrchestration:
    def __init__(self, files_info: Dict, fs_info: Dict, pred_perf: Dict):
        self.files = files_info
        self.fs = fs_info
        self.pred = pred_perf
        self.D = [0] * files_info['n']
        self.U = np.zeros((files_info['n'], fs_info['m']), dtype=int)

        # 验证数据一致性
        assert len(pred_perf['B']) == files_info['n']
        for i in range(files_info['n']):
            assert len(pred_perf['B'][i]) == fs_info['m']

    def _calc_weights(self, file_idx: int) -> Tuple[float, float]:
        total = self.files['WS'][file_idx] + self.files['RS'][file_idx]
        if total == 0:
            return (0.5, 0.5)
        return (
            self.files['WS'][file_idx] / total,
            self.files['RS'][file_idx] / total
        )

    def _integrate_perf(self, file_idx: int, fs_idx: int) -> float:
        w_w, w_r = self._calc_weights(file_idx)
        return self.pred['Bw'][file_idx][fs_idx] * w_w + self.pred['Br'][file_idx][fs_idx] * w_r

    def _update_fs_state(self, fs_idx: int, file_idx: int):
        self.fs['SO'][fs_idx] += self.files['S'][file_idx]
        self.fs['BO'][fs_idx] += self.pred['B'][file_idx][fs_idx]
        self.fs['L'][fs_idx] += 1
        self.files['TS'][file_idx] = float('inf')
        self.D[file_idx] = fs_idx

    def _check_constraints(self, file_idx: int, fs_idx: int) -> bool:
        if fs_idx >= len(self.fs['BF']) or fs_idx >= len(self.pred['B'][file_idx]):
            return False
        bw_ok = (self.fs['BO'][fs_idx] + self.pred['B'][file_idx][fs_idx] <= self.fs['BF'][fs_idx]) or (self.U[file_idx][fs_idx] == 1)
        cap_ok = (self.fs['SO'][fs_idx] + self.files['S'][file_idx] <= self.fs['SF'][fs_idx])
        return bw_ok and cap_ok

    def _adjust_prediction(self, file_idx: int, fs_idx: int):
        if fs_idx >= len(self.fs['SF']) or fs_idx >= len(self.pred['B'][file_idx]):
            return
        if self.fs['SO'][fs_idx] + self.files['S'][file_idx] > self.fs['SF'][fs_idx]:
            self.pred['B'][file_idx][fs_idx] = 0
        else:
            self.pred['B'][file_idx][fs_idx] = self.fs['BF'][fs_idx] / (self.fs['L'][fs_idx] + 1)
            self.U[file_idx][fs_idx] = 1

    def run(self) -> List[int]:
        sorted_files = sorted(range(self.files['n']), key=lambda x: self.files['TS'][x])
        for file_idx in sorted_files:
            if self.files['TS'][file_idx] == float('inf'):
                continue

            best_fs, max_perf = -1, -np.inf
            for fs_idx in range(self.fs['m']):
                if fs_idx >= len(self.pred['B'][file_idx]):
                    continue

                current_perf = self._integrate_perf(file_idx, fs_idx)

                if self._check_constraints(file_idx, fs_idx):
                    if current_perf > max_perf:
                        max_perf = current_perf
                        best_fs = fs_idx
                else:
                    self._adjust_prediction(file_idx, fs_idx)

            if best_fs != -1 and max_perf > 0:
                self._update_fs_state(best_fs, file_idx)
        return self.D


def ado(pred_file, file_fs_tag_file):
    create_fs_tag_file(fgap_config.FS_CONFIG_PATH, fgap_config.FGAP_FS)
    try:
        fs_info, fs_map = parse_fs_config(fgap_config.FS_CONFIG_PATH)
        files_info, pred_perf = parse_pred_file(pred_file, fs_map)

        # 计算综合性能（修复语法错误）
        for i in range(files_info['n']):
            for j in range(fs_info['m']):
                total = files_info['WS'][i] + files_info['RS'][i]
                if total != 0:
                    w_w = files_info['WS'][i] / total
                    w_r = files_info['RS'][i] / total
                else:
                    w_w, w_r = 0.5, 0.5
                pred_perf['B'][i][j] = pred_perf['Bw'][i][j] * w_w + pred_perf['Br'][i][j] * w_r

        # 运行算法
        ado = AdaptiveDataOrchestration(files_info, fs_info, pred_perf)
        allocation = ado.run()

        # 打印结果
        with open(file_fs_tag_file, 'w') as f:
            print("\n文件分配结果：")
            for idx, fs_idx in enumerate(allocation):
                fs_name = fs_info['names'][fs_idx]
                file_size = files_info['S'][idx] // 1024**2  # 转换为MB
                used = fs_info['SO'][fs_idx] // 1024**2
                total = fs_info['SF'][fs_idx] // 1024**2
                print(f"文件 {files_info['filenames'][idx]} ({file_size}MB) -> {fs_name} (已用: {used}MB/总量: {total}MB)")
                f.write(f"{files_info['filenames'][idx]} {FSDIR_IDX_MAP[fs_name[1]]}\n")

    except Exception as e:
        print(f"程序运行错误: {str(e)}")


# --------------------------
# 主程序
# --------------------------
if __name__ == "__main__":
    ado("pred_test.txt", "aaaaa")
