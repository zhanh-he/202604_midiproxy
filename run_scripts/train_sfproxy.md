# SFProxy 训练说明
```bash
cd /media/mengh/SharedData/zhanh/202604_midiproxy

INSTRUMENT=guitar SEGMENT_SECONDS=2 bash preprocess_sfproxy_data.sh 
INSTRUMENT=guitar SEGMENT_SECONDS=5 bash preprocess_sfproxy_data.sh
INSTRUMENT=guitar SEGMENT_SECONDS=10 bash preprocess_sfproxy_data.sh

MIX_WEIGHTS="0.3 0.4 0.2 0.1" bash preprocess_sfproxy_parallel_train.sh
MIX_WEIGHTS="0.3 0.4 0.2 0.1" bash preprocess_sfproxy_parallel_val.sh

wait
```

这份说明只讲一件事：**sampler 怎么理解，数据怎么准备，脚本怎么跑。**

核心结论先放前面：

- `mixed_v2` **不是** `coverage_v2 + realism_v2` 的二元混合。
- `mixed_v2` 是 **4-way segment-level mixture**：
  - `boundary_v2`
  - `coverage_v2`
  - `realism_v2`
  - `stress_v2`
- 更直白地说：**一条 sample 先决定自己属于哪一类，再用那一类的 sampler 生成整段 note events。**

---

## 1. 先记住 sampler 的层级

SFProxy 的 sampler 可以拆成三层：

| 层级 | 决定什么 | 典型选项 |
|---|---|---|
| segment type | 这一整段属于哪种数据风格 | `boundary / coverage / realism / stress` |
| note skeleton | pitch / duration / IOI / chord size 怎么来 | coverage-style 或 realism-style |
| velocity rule | velocity 怎么采，chord 内是否共享 | `shared / independent / correlated / mixed` |

所以 sampler 不是一个按钮，而是一个 **hierarchical generator**。

---

## 2. 这些 preset 分别是什么意思

### Legacy / v1

| preset | 作用 | 特点 |
|---|---|---|
| `coverage_v1` | old coverage baseline | coverage 骨架，chord 内 velocity 全共享 |
| `realism_v1` | old realism baseline | realism 骨架，chord 内 velocity 全共享 |

这两者可以统称为 **data preprocessing v1**。
旧名字 `coverage_shared_legacy / realism_shared_legacy` 仍然兼容，但现在推荐统一写成 `coverage_v1 / realism_v1`。

### New / v2

| preset | 作用 | 特点 |
|---|---|---|
| `boundary_v2` | 专打 velocity boundary | 低复音，更多 boundary / extreme velocities |
| `coverage_v2` | 新版 coverage | 广覆盖，适合 controlled ablation |
| `realism_v2` | 新版 realism | 结构更像真实 MIDI |
| `stress_v2` | 困难样本 | 更 dense、更 overlap、更极端 |
| `mixed_v2` | 主设定 | 按比例混合上面四类 |

**最重要的一句：`mixed_v2` 是由 `boundary_v2 + coverage_v2 + realism_v2 + stress_v2` 组成。**

---

## 3. 我们现在推荐的理解方式

你可以这样理解，而且现在脚本也支持这样跑：

### data preprocessing v2

一次 v2 预处理，会先导出四个基础数据集：

- `boundary_v2`
- `coverage_v2`
- `realism_v2`
- `stress_v2`

然后：

- 如果你要训练 `coverage_v2`，就只用 `coverage_v2` 那个子集。
- 如果你要训练 `realism_v2`，就只用 `realism_v2` 那个子集。
- 如果你要训练 `mixed_v2`，就按权重把四个子集 compose 成一个 mixed dataset 再训练。

默认权重是：

| component | 默认权重 |
|---|---:|
| `boundary_v2` | 0.30 |
| `coverage_v2` | 0.40 |
| `realism_v2` | 0.20 |
| `stress_v2` | 0.10 |

### data preprocessing v1

legacy 也按同样思路单独导出：

- `coverage_v1`
- `realism_v1`

这就是 **data preprocessing v1**。

---

## 4. 为什么这样更清楚

这样做有三个好处：

| 好处 | 解释 |
|---|---|
| 更模块化 | 先把基础数据集准备好，再决定训练时用哪一个 |
| 更易解释 | `mixed_v2` 不再是一个“黑盒 sampler 名字”，而是明确的 4-way mixture |
| 更易做 ablation | 可以单独比较 `boundary_v2 / coverage_v2 / realism_v2 / stress_v2 / mixed_v2` |

这也是我们现在推荐的组织方式。

---

## 5. chord velocity mode 怎么读

这部分只影响 **同一个 chord onset 内** 的 velocity 关系。

| mode | 意思 | 适合什么 |
|---|---|---|
| `shared` | 一个 chord 里所有 notes 同一个 velocity | legacy baseline |
| `independent` | 每个 note 自己采 velocity | 学 note-wise sensitivity |
| `correlated` | 先采 base，再加小 jitter | 更像真实和弦 |
| `mixed` | 三种模式按概率混合 | v2 主设定 |

当前 v2 的核心升级就是：**不再把 chord 内 velocity 永远绑死成 shared。**

---

## 6. boundary mode 怎么读

| mode | 意思 |
|---|---|
| `fixed` | 固定用默认 boundary，比如 `[0.33, 0.66]` |
| `default` | 有 boundary JSON 就用，没有就退回固定 boundary |
| `discovered` | 显式使用 SoundFont-aware discovered boundaries |

如果你想做最干净的 old vs new ablation：

- legacy: `fixed` 或 `default`
- v2 main: `discovered`

---

## 7. 现在 repo 里新增了什么

为了让上面的理解真的能跑通，现在 repo 里多了两类东西：

### A. 新的顶层 v2 presets

以前 `boundary` 和 `stress` 只存在于 `mixed_v2.components` 里面。

现在它们被提升成了顶层 preset：

- `boundary_v2`
- `stress_v2`

所以你现在可以直接单跑：

- `SAMPLER_PRESET=boundary_v2`
- `SAMPLER_PRESET=stress_v2`

### B. modular preprocessing script

新增脚本：

- `run_scripts/preprocess_sfproxy_data.sh`

它负责做两件事：

1. 导出基础数据集
2. 如果目标是 `mixed_v2`，再把组件数据按比例 compose 成 mixed dataset

### C. optional modular training path

现在这两个训练脚本都支持：

- `PREPROCESSING_LAYOUT=monolithic`
- `PREPROCESSING_LAYOUT=modular`

其中：

- `monolithic`：旧逻辑，直接按一个 sampler preset export 再 train
- `modular`：新逻辑，先走 modular preprocessing，再 train

脚本：

- `run_scripts/train_sfproxy.sh`
- `run_scripts/train_sfproxy_ablations.sh`

---

## 8. 最常用的运行方式

### 8.1 跑主设定 `mixed_v2`

```bash
PREPROCESSING_LAYOUT=modular \
SAMPLER_PRESET=mixed_v2 \
BOUNDARY_MODE=discovered \
SEGMENT_SECONDS=2 \
./run_scripts/train_sfproxy.sh
```

这会做：

1. 导出 `boundary_v2 / coverage_v2 / realism_v2 / stress_v2`
2. 按 `0.30 / 0.40 / 0.20 / 0.10` compose 成 `mixed_v2`
3. 用 compose 后的数据训练 SFProxy

### 8.2 只跑 `coverage_v2`

```bash
PREPROCESSING_LAYOUT=modular \
SAMPLER_PRESET=coverage_v2 \
BOUNDARY_MODE=discovered \
SEGMENT_SECONDS=2 \
./run_scripts/train_sfproxy.sh
```

### 8.3 只做数据预处理，不立刻训练

```bash
TARGET_PRESET=v2_bundle \
BOUNDARY_MODE=discovered \
SEGMENT_SECONDS=2 \
./run_scripts/preprocess_sfproxy_data.sh
```

这会导出：

- `boundary_v2`
- `coverage_v2`
- `realism_v2`
- `stress_v2`
- compose 后的 `mixed_v2`

### 8.4 跑 legacy 数据

```bash
TARGET_PRESET=v1_bundle \
BOUNDARY_MODE=fixed \
SEGMENT_SECONDS=2 \
./run_scripts/preprocess_sfproxy_data.sh
```

这会导出：

- `coverage_v1`
- `realism_v1`

---

## 9. 我们建议的 ablation 顺序

不要一上来全扫。先按这几条主轴：

| 主轴 | 推荐对比 |
|---|---|
| old vs new | `coverage_v1` vs `coverage_v2` |
| component effect | `boundary_v2 / coverage_v2 / realism_v2 / stress_v2` |
| composed main | `coverage_v2` vs `mixed_v2` |
| boundary | `fixed` vs `discovered` |
| segment length | `2 / 5 / 10` |

最核心的是先回答两个问题：

1. v2 比 legacy 有没有明显更好。
2. `mixed_v2` 比单独 `coverage_v2` 有没有更稳。

---

## 10. 一句话总结

最简洁的记法就是：

- **v1 = legacy two-set preprocessing**
- **v2 = four-set preprocessing**
- **mixed_v2 = compose(boundary_v2, coverage_v2, realism_v2, stress_v2)**

所以以后你看到 `mixed_v2`，脑子里不要再想成“一个神秘 sampler 名字”，而要想成：

**a composed training set built from four explicit component datasets.**
