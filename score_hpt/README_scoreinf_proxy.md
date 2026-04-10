# Score-HPT + SFProxy Notes

这份说明只保留当前还在使用的流程，重点是：

- SFProxy 数据怎么准备
- SFProxy 模型怎么训练
- 现阶段 eval 主要评什么
- `score_hpt` 接 SFProxy backend 时该看哪些文件

## 1. 现在的主线

当前 repo 里最常用的是两条线：

- `synth-proxy/`
  - 准备 SoundFont teacher data
  - 训练 SFProxy
  - 跑 monotonicity / velocity recovery eval

- `score_hpt/pytorch/`
  - 训练 score-informed velocity model
  - 可接 frozen SFProxy 作为 backend loss

如果你现在要先把 SFProxy 路线跑通，优先看：

- [preprocess_sfproxy_data.sh](/media/mengh/SharedData/zhanh/202604_midiproxy/run_scripts/preprocess_sfproxy_data.sh)
- [train_sfproxy.sh](/media/mengh/SharedData/zhanh/202604_midiproxy/run_scripts/train_sfproxy.sh)
- [train_sfproxy_ablations.sh](/media/mengh/SharedData/zhanh/202604_midiproxy/run_scripts/train_sfproxy_ablations.sh)
- [eval.ipynb](/media/mengh/SharedData/zhanh/202604_midiproxy/synth-proxy/eval.ipynb)

## 2. SFProxy 数据准备

默认直接跑：

```bash
cd /media/mengh/SharedData/zhanh/202604_midiproxy/run_scripts
bash preprocess_sfproxy_data.sh
```

默认配置是：

- `INSTRUMENT=piano`
- `PIANO_DATASET=maestro`
- 输出乐器名是 `salamander_piano`
- `BOUNDARY_MODE=default`

所以这条默认命令准备的是：

- Salamander piano
- 基于 MAESTRO 的 realism stats

脚本会先导出四个 component：

- `boundary_v2`
- `coverage_v2`
- `realism_v2`
- `stress_v2`

然后再 compose 出：

- `mixed_v2`

### 改 segment

```bash
SEGMENT_SECONDS=2 bash preprocess_sfproxy_data.sh
SEGMENT_SECONDS=5 bash preprocess_sfproxy_data.sh
SEGMENT_SECONDS=10 bash preprocess_sfproxy_data.sh
```

### 改 mixture 配方

顺序固定是：

- `boundary coverage realism stress`

例如：

```bash
MIX_WEIGHTS="0.3 0.4 0.2 0.1" bash preprocess_sfproxy_data.sh
```

如果 component 数据已经存在，这一步主要是重新 compose `mixed_v2`，会比重新渲染快很多。

### 准备 guitar 数据

默认 guitar 数据集是 `francoisleduc`：

```bash
INSTRUMENT=guitar bash preprocess_sfproxy_data.sh
```

如果要 `gaps`：

```bash
INSTRUMENT=guitar GUITAR_DATASET=gaps bash preprocess_sfproxy_data.sh
```

常用 guitar segment：

```bash
INSTRUMENT=guitar SEGMENT_SECONDS=2 bash preprocess_sfproxy_data.sh
INSTRUMENT=guitar SEGMENT_SECONDS=5 bash preprocess_sfproxy_data.sh
```

## 3. SFProxy 训练

单个模型：

```bash
bash train_sfproxy.sh
```

默认是：

- `TRAIN_PRESET=mixed_v2`
- `SEGMENT_SECONDS=2`
- `BOUNDARY_MODE=default`

### 单独训练某个 preset

```bash
TRAIN_PRESET=coverage_v2 bash train_sfproxy.sh
TRAIN_PRESET=realism_v2 bash train_sfproxy.sh
TRAIN_PRESET=stress_v2 bash train_sfproxy.sh
TRAIN_PRESET=mixed_v2 bash train_sfproxy.sh
```

### 训练 5s / 10s

单独跑：

```bash
SEGMENT_SECONDS=5 bash train_sfproxy.sh
SEGMENT_SECONDS=10 bash train_sfproxy.sh
```

或者批量做 ablation：

```bash
bash train_sfproxy_ablations.sh
```

当前默认 sweep 是：

- `SEGMENT_LIST=2`
- `BOUNDARY_MODE_LIST=default`
- `TRAIN_PRESETS="coverage_v2 realism_v2 stress_v2 mixed_v2"`

如果你要把 5s 和 10s 也补齐：

```bash
SEGMENT_LIST="5 10" bash train_sfproxy_ablations.sh
```

如果你想一次扫完 `2/5/10`：

```bash
SEGMENT_LIST="2 5 10" bash train_sfproxy_ablations.sh
```

## 4. 现在推荐训练哪些模型

如果目标是先把 eval 跑通，建议先训练这几类 piano 模型：

- `coverage_v2`, `2s`, `default`
- `realism_v2`, `2s`, `default`
- `stress_v2`, `2s`, `default`
- `mixed_v2`, `2s`, `default`

然后再补：

- `mixed_v2`, `5s`, `default`
- `mixed_v2`, `10s`, `default`

如果时间够，再把 `coverage/realism/stress` 的 `5s/10s` 补齐做完整 ablation。

## 5. Eval 现在看什么

当前主要看 [eval.ipynb](/media/mengh/SharedData/zhanh/202604_midiproxy/synth-proxy/eval.ipynb)。

这份 notebook 现在主要是：

- piano
- Salamander
- 2s

也就是说：

- 现在先用 piano 模型做 eval 是合理的
- 现在这份 eval 不是以 guitar 为主线
- guitar 数据可以同步准备，但不是当前 eval 的前置条件

如果后面要认真评 guitar，需要补一套和 guitar 对齐的 eval 配置。

## 6. Score-HPT 这边的入口

`score_hpt/pytorch/` 里当前常见入口：

- `pytorch/train.py`
  - Route II
  - 纯监督 velocity estimation

- `pytorch/train_ddsp.py`
  - Route III
  - 接 DDSP / DiffSynth backend

- `pytorch/train_proxy.py`
  - Route IV
  - 接 frozen SFProxy backend

常见 Route IV 形式：

```bash
cd /media/mengh/SharedData/zhanh/202604_midiproxy/score_hpt

python pytorch/train_proxy.py \
  proxy.checkpoint=/path/to/sfproxy.ckpt \
  proxy.sfproxy.instrument_name=salamander_piano
```

## 7. 现在已经淘汰的旧说法

下面这些说法现在不要再参考：

- `PREPROCESSING_LAYOUT=monolithic/modular`
- `TARGET_PRESET=v2_bundle`
- 把 `mixed_v2` 理解成在线 sampler 直接生成的唯一主数据

现在真实流程是：

1. 先准备 `boundary / coverage / realism / stress`
2. 再 compose `mixed_v2`
3. 训练时从已有数据目录读

## 8. 一套最常用的命令

准备 piano 默认数据：

```bash
cd /media/mengh/SharedData/zhanh/202604_midiproxy/run_scripts
bash preprocess_sfproxy_data.sh
```

训练 `2s` ablations：

```bash
bash train_sfproxy_ablations.sh
```

补 `5s` 和 `10s`：

```bash
SEGMENT_LIST="5 10" bash train_sfproxy_ablations.sh
```

准备 guitar：

```bash
INSTRUMENT=guitar bash preprocess_sfproxy_data.sh
```

开 eval：

- 打开 [eval.ipynb](/media/mengh/SharedData/zhanh/202604_midiproxy/synth-proxy/eval.ipynb)
- 填好对应 checkpoint
- 先评 piano `2s` 主线
