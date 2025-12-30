# Tests

单元测试覆盖了 `fusang.py` 中的核心算法函数。

## 运行测试

本项目使用 `uv` 管理依赖，pytest 已配置为开发依赖。

### 使用 uv 运行测试（推荐）

```bash
# 确保开发依赖已安装
uv sync --with dev

# 运行所有测试
uv run pytest tests/

# 运行特定测试文件
uv run pytest tests/test_math_utils.py

# 显示详细输出
uv run pytest tests/ -v

# 显示覆盖率（需要先添加 pytest-cov: uv add --dev pytest-cov）
uv run pytest tests/ --cov=fusang --cov-report=html
```

### 使用传统方式运行测试

```bash
# 安装 pytest（如果尚未安装）
pip install pytest

# 运行所有测试
pytest tests/

# 运行特定测试文件
pytest tests/test_math_utils.py

# 显示详细输出
pytest tests/ -v
```

## 测试文件结构

- `test_math_utils.py` - 数学工具函数（组合数计算、最大索引查找）
- `test_quartet.py` - 四元组相关函数（ID生成、拓扑识别、树创建）
- `test_tree_operations.py` - 树操作函数（修改树、字符串转换）
- `test_data_processing.py` - 数据处理函数（对齐解析、四元组初始化）
- `test_evaluation.py` - 评估函数（RF距离计算）
- `test_mask_operations.py` - 掩码操作函数（节点对选择、边掩码）

## 测试覆盖的核心函数

- `comb_math()` - 组合数计算
- `nlargest_indices()` - 最大n个值的索引
- `get_quartet_ID()` - 四元组ID生成
- `get_topology_ID()` - 拓扑ID生成
- `get_current_topology_id()` - 当前拓扑ID识别
- `tree_from_quartet()` - 从四元组创建树
- `get_modify_tree()` - 修改树结构
- `transform_str()` - 字符串转换
- `cmp()` - 数值比较函数
- `_process_alignment()` - 对齐文件处理
- `get_numpy()` - 转换为numpy数组
- `initialize_quartet_data()` - 四元组数据初始化
- `calculate_rf_distance()` - RF距离计算
- `select_mask_node_pair()` - 掩码节点对选择
- `mask_edge()` - 边掩码操作

