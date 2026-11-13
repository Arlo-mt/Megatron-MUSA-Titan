# RMSNorm 精度测试

使用 pytest 验证本仓库两个 Triton RMSNorm 的前向接口及 fused autograd 支持。

## 范围与组织

- 使用 pytest，只在 MUSA 上执行被测 Triton 算子。
- 仅覆盖 `torch.float32` 和 `torch.bfloat16`，输入、权重和残差使用相同 dtype。
- 覆盖普通 RMSNorm 和 fused add + RMSNorm 的前向精度、重复执行确定性、batch 不变性及接口约定。
- fused autograd 用例检查输入、残差和权重的一阶梯度可计算、有限且非零。
- 不包含 benchmark、性能门槛、独立的梯度数值正确性比较或端到端训练验收。

本目录包含以下测试文件：

```text
musa_titan/rl_align_kernels/tests/ops/rmsnorm/
├── README.md
├── conftest.py                  # MUSA fixture、固定种子、参考计算和比较函数
├── test_rmsnorm.py              # 两个算子共用的精度、不变性和权重接口用例
└── test_fused_add_rmsnorm.py     # fused 残差语义、舍入、异常及 autograd 检查
```

测试直接导入仓库 `musa_titan/rl_align_kernels/kernels/rmsnorm/` 中的实现，并核对实际导入路径。

## 测试方法

### 独立参考值

使用固定种子在 CPU 生成数据，先转换到被测 dtype，再将同一份数据传到 MUSA。
CPU 只负责生成数据和计算参考值，不作为被测算子的执行后端。
参考计算不调用被测 kernel，也不复制其 Triton reduction tree。

普通 RMSNorm 使用独立 PyTorch FP32 公式：

```text
x32 = FP32(x)
w32 = FP32(weight)
rstd_ref = 1 / sqrt(mean(x32 * x32, dim=-1) + eps)
y_ref = x32 * rstd_ref[..., None] * w32
```

用 CPU `torch.nn.functional.rms_norm` 交叉检查公式。参考值必须基于已经转换到
被测 dtype 的输入，避免将输入量化误差算入 kernel 误差。MUSA 输出转 CPU FP32
后与 FP32 参考值比较，比较包含最终输出转换带来的舍入误差。

fused add + RMSNorm 使用以下计算约定：

```text
s32 = FP32(x) + FP32(residual)
rstd_ref = 1 / sqrt(mean(s32 * s32, dim=-1) + eps)
y_ref = s32 * rstd_ref[..., None] * FP32(weight)
residual_ref = cast(s32, input.dtype)
```

归一化使用尚未转换回输入 dtype 的 `s32`。不能以 BF16 的
`rms_norm(x + residual)` 作为 fused 参考值，因为该写法会提前舍入残差和。

### 数值精度

使用 `torch.testing.assert_close`，固定门槛如下。

| 被测结果 | atol | rtol |
| --- | ---: | ---: |
| FP32 归一化输出 | 2e-5 | 2e-5 |
| BF16 归一化输出，转 FP32 后比较 | 2e-2 | 2e-2 |
| fused 输出的 FP32 `rstd`，两种输入 dtype 均适用 | 2e-5 | 2e-5 |

所有有限输入样例都检查输出有限性。失败时先定位计算差异，不自动放宽门槛。
不同实现的 reduction 顺序可能不同，因此独立参考精度通过不等于逐位一致。

### 逐位一致性

先检查 shape 和 dtype，再将张量复制到 CPU、转为连续存储并展平，按原始字节比较。
不只使用 `max_abs_diff == 0` 或浮点数值相等，避免遗漏正负零等位模式差异。

- 重复执行：同一输入执行 10 次，全部与第一次结果逐位一致。
- batch 不变性：固定目标行和权重，改变 batch 大小、目标行位置及相邻行数据。
- 行布局不变性：插入无关 padding 行、打乱行顺序后恢复，以及完整 batch 与分片执行。
- 张量布局不变性：同一逻辑输入以不同前导维度、连续和非连续布局执行，恢复相同 shape 后比较。

普通 RMSNorm 检查归一化输出；fused 检查四个输出，包括 FP32 `rstd` 和
`residual_sum_fp32`。同一后端内的不变性检查全部要求零字节差异。

### Fused 四输出与接口约定

| 输出 | shape / dtype | 检查方法 |
| --- | --- | --- |
| `normalized` | 与输入相同 | 与 `y_ref` 按输入 dtype 对应容差比较 |
| `residual_sum` | 与输入相同 | 与 `residual_ref` 逐位比较 |
| `rstd` | `input.shape[:-1]` / FP32 | 与 `rstd_ref` 按 FP32 容差比较 |
| `residual_sum_fp32` | 与输入相同 / FP32 | 与 `s32` 逐位比较 |

另行检查输出 device、输入和权重未被修改，以及 fused 四个非空输出使用独立存储。
一维输入对应的 `rstd` 应为标量。

异常检查对齐 Titan 接口：普通 RMSNorm 的权重维度错误，以及 fused 的 shape、dtype、eps
错误均触发 `ValueError`。

fused 支持一阶 autograd：分别让输入、残差或权重需要梯度，以归一化输出和残差输出的和
构造 loss，检查对应梯度的 shape、dtype、有限性及非零性，并确认 FP32 `rstd` 和
`residual_sum_fp32` 不参与求导。该用例验证 backward 接入，不提供独立的梯度数值精度
或 backward batch 不变性验收。

## 测试矩阵

两种算子均覆盖 FP32、BF16。除专门的 epsilon 测试外，默认 `eps=1e-6`。
按下表分组参数化，组与组之间不做笛卡尔积。

| 测试组 | 参数和数据 | 判定 |
| --- | --- | --- |
| 常规精度 | `H={128,1024,2048,4096}` × 行数 `{1,7,64,257}` × 两种 dtype；随机非单位权重；seed=0 | 独立参考容差、有限性、shape/dtype/device |
| 分块边界 | `H={1023,1025,2047,2049}` × 行数 `{1,7}` × 两种 dtype；seed=1 | 尾部 mask 与跨 1024 列分块精度 |
| 重复执行 | `H=2048`，行数 64，两种 dtype；seed=2；重复 10 次 | 原始字节一致 |
| batch 与行布局 | `H={128,2048}`，batch 行数 `{1,7,64,257}`，目标行位于首部、中间、末尾；两种 dtype；目标行 seed=3，邻居 seed=30/31，padding seed=33 | 单行、padding、排列恢复、分片结果逐位一致 |
| 张量布局 | `H=2048`；一维 `[H]` 与 `[1,H]`；同一组 6 行分别为 `[6,H]`、`[2,3,H]`；额外构造非连续输入、残差和权重；两种 dtype；seed=4 | 恢复相同逻辑布局后逐位一致 |
| 数值边界 | `H=2048`，7 行；全零、幅度约 `1e-4` 的小值、正负混合、幅度 `1e-4` 至 `1e4` 的有限值；两种 dtype；seed=5 | 精度与有限性 |
| epsilon | `H=2048`，7 行；小值输入；`eps={1e-6,1e-5}`；两种 dtype；seed=6 | 对照参考确认 epsilon 在平方根内；零输入不作为唯一证据 |
| 权重语义 | `H=2048`，7 行；随机、全零、全一、全二权重；两种 dtype；seed=7 | 正确应用 `weight`，排除 `1+weight` 语义 |
| fused 残差语义 | `H=2048`，7 行；残差为零、精确抵消、近似抵消；另加 BF16 加法舍入边界样例；seed=8 | 四输出正确；参考数据能区分提前舍入和 FP32 和参与归一化 |
| 接口边界 | 权重 shape 错误；fused 残差 shape、输入间 dtype 不一致、空 hidden、非法 eps；两种 dtype；seed=9 | 符合当前接口的 `ValueError` 异常类型 |
| fused autograd | `H=2048`，7 行；分别使输入、残差、权重 `requires_grad=True`；两种 dtype；seed=9，共 6 例 | 一阶梯度 shape/dtype 正确、有限且非零；FP32 缓存不参与求导 |

BF16 舍入边界样例需先在参考计算中确认“提前舍入”会改变预期结果，再用于测试。
随机非单位权重为主要精度数据，全一权重仅用于特定语义检查。

## 运行与验收

在已安装 `pytest`、`torch`、`torch_musa` 和适配版本 Triton 的云端
MUSA 环境，从仓库根目录运行：

```bash
python -m pytest musa_titan/rl_align_kernels/tests/ops/rmsnorm -v -ra
```

测试注册 `torch_musa` 并检查设备可用性；缺少依赖或 MUSA 设备时明确失败，
不回退 CPU，也不以全部 skip 作为验收通过。执行前检查 GPU 占用，不终止其他任务。

失败信息包含算子、shape、dtype、eps、seed、最大绝对误差、超差元素数及首个差异位置；
逐位检查额外报告字节不匹配数。pytest 报告头记录 kernel 源码路径及 SHA256、
PyTorch/torch_musa/Triton 版本和设备型号。运行时另行保留 Git 版本与日志。
结尾按 `accuracy`、`invariance`、`contract` 分别汇总通过、失败、错误和跳过数量。

## 参考来源

参考 RL-Kernel 提交 `49941801f942428fd52ac5b66a1b0c070418724e`：

- [RMSNorm 测试](https://github.com/RL-Align/RL-Kernel/blob/49941801f942428fd52ac5b66a1b0c070418724e/tests/ops/norm/test_rms_norm.py)
- [PyTorch 参考实现](https://github.com/RL-Align/RL-Kernel/blob/49941801f942428fd52ac5b66a1b0c070418724e/rl_engine/reference/norm/rms_norm.py)

借鉴精度与不变性分开判定的方法及 dtype 容差；本方案限定 MUSA、FP32/BF16，另按 Titan 接口补充 fused autograd 支持检查。
