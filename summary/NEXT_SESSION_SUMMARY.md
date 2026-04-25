# MYD03 接入与小规模训练交接记录

更新时间：2026-04-24
项目目录：`/home/yuq/cloudmask/unet`
推荐环境：`cloudunet`

## 1. 当前目标

本阶段目标是把 MYD03 1km geolocation 加入现有 AGRI + MYD06 融合流程，使 MYD06 的 1km 标签使用 MYD03 的 1km 经纬度匹配到 AGRI，并做小规模训练验证模型流程和 OA 表现。

## 2. 环境信息

使用环境：

```bash
source /home/yuq/anaconda3/bin/activate cloudunet
export LD_LIBRARY_PATH=/home/yuq/anaconda3/envs/cloudunet/lib:$LD_LIBRARY_PATH
```

已确认：

- Python：`/home/yuq/anaconda3/envs/cloudunet/bin/python`
- PyTorch：`2.11.0+cu128`
- CUDA：`12.8`
- GPU：`2 x NVIDIA GeForce RTX 4090`
- `pyhdf` 可用

注意：如果不设置 `LD_LIBRARY_PATH`，`pandas` 可能报：

```text
GLIBCXX_3.4.29 not found
```

## 3. 已完成的代码改动

### 3.1 MYD03 路径配置

文件：`config.py`

新增：

```python
MYD03_ROOT = Path("/data/Data_yuq/MYD03/")
```

### 3.2 MYD03 与 MYD06 匹配

文件：`fusion_io.py`

新增函数：

- `find_matching_myd03(myd06_file, myd03_files)`
- 按 MYD06 文件名中的 `AYYYYDDD.HHMM` 精确匹配 MYD03。

### 3.3 读取 MYD03 1km 经纬度

文件：`fusion_io.py`

新增函数：

- `read_myd03(myd03_file)`
- 读取 MYD03 的 `Latitude` / `Longitude`
- 返回 `(lat_1km, lon_1km)`

`read_myd06(...)` 已扩展为：

```python
read_myd06(modis_file, agri_dt=None, myd03_file=None)
```

如果传入 MYD03，则读取 MYD03 1km 经纬度，并检查 shape 是否与 MYD06 1km 标签一致；不一致则回退到原 MYD06 5km 经纬度上采样逻辑。

### 3.4 聚合阶段优先用 MYD03 1km 经纬度

文件：`fusion_core.py`

`_collect_1km(...)` 现在优先使用：

```python
lat_1km = m.get("lat_1km")
lon_1km = m.get("lon_1km")
```

若 MYD03 缺失或 shape 不匹配，才继续使用原来的 `upsample_5km_to_1km_coords(...)`。

### 3.5 融合入口传递 MYD03

文件：`data_fusion.py`

- `fuse_day(...)` 新增参数：`myd03_day_dir`
- 每个 MYD06 granule 会找对应 MYD03，并以 `(MYD06, MYD03)` 形式传入 worker。
- `_fuse_one_scene(...)` 已支持读取 tuple 输入。

文件：`main.py`

- `stage_fuse(...)` 也已接入 `cfg.MYD03_ROOT`
- 通过 `main.py --stages fuse` 跑主流程时也会使用 MYD03。

### 3.6 训练日志目录修复

文件：`train.py`

新增：

```python
cfg.LOG_DIR.mkdir(parents=True, exist_ok=True)
```

原因：训练结束写 `train_log.csv` 时，如果 `LOG_DIR` 不存在会报错。

## 4. 验证情况

### 4.1 MYD03 / MYD06 读取验证

实际读取过一组：

- MYD06：`MYD06_L2.A2019176.0020.061.2019176174417.hdf`
- MYD03：`MYD03.A2019176.0020.061.2019176151644.hdf`

结果：

- MYD03 `lat_1km` shape：`(2030, 1354)`
- MYD03 `lon_1km` shape：`(2030, 1354)`
- MYD06 1km 标签 shape：`(2030, 1354)`
- shape 匹配成功。

### 4.2 测试日期 MYD03 配对验证

按当前时间窗口，匹配到的 MYD06 都能找到对应 MYD03。

示例统计曾验证：

- `20190225`
- `20190425`
- `20190625`

MYD03 pairs 与 MYD06 matched granules 数量一致。

## 5. 已生成的小规模数据与训练产物

### 5.1 Smoke 测试目录

目录：

```text
unet_workdir/paired/_smoke_myd03
```

包含：

- train：`20190105 05:45`，`144` samples
- val：`20190125 05:00`，`550` samples
- test：`20190225 06:00`，`436` samples

这个目录只用于流程冒烟验证，不建议作为泛化结论依据。

### 5.2 小规模泛化训练目录

目录：

```text
unet_workdir/paired/_gen_small_myd03
```

初始 train/val/test：

- train：`20190105`, `20190305`, `20190505`
- val：`20190125`
- test：`20190225`

后续扩展 train 到 10 个跨日期场景：

- `20190105`
- `20190115`
- `20190205`
- `20190215`
- `20190305`
- `20190315`
- `20190405`
- `20190415`
- `20190505`
- `20190515`

扩展后训练 patch 数：`7085`

val patch 数：`550`

test patch 数：`121`

## 6. 训练结果

### 6.1 第一轮小训练集

目录：

```text
unet_workdir/paired/_gen_small_myd03/model_e20
unet_workdir/paired/_gen_small_myd03/logs_e20
```

结果：

- val OA 峰值：`63.99%`
- best val loss epoch：约第 `3` epoch
- best val OA epoch：约第 `4` epoch

### 6.2 CLP 权重加到 2.0

目录：

```text
unet_workdir/paired/_gen_small_myd03/model_e20_clp2
unet_workdir/paired/_gen_small_myd03/logs_e20_clp2
```

结果：

- val OA 峰值：`64.07%`
- 相比默认权重提升很小，不建议继续只靠加 CLP 权重。

### 6.3 扩展训练集 30 epoch

目录：

```text
unet_workdir/paired/_gen_small_myd03/model_expanded_e30
unet_workdir/paired/_gen_small_myd03/logs_expanded_e30
```

重要文件：

```text
unet_workdir/paired/_gen_small_myd03/model_expanded_e30/expanded_best.pth
unet_workdir/paired/_gen_small_myd03/logs_expanded_e30/train_log.csv
unet_workdir/paired/_gen_small_myd03/stats_gen_expanded.npz
```

结果：

- val OA 峰值：`65.48%`
- 加载 best checkpoint 后复评：
  - val OA：`65.56%`
  - test OA：`27.30%`

## 7. 当前判断

1. MYD03 接入和 1km 匹配流程已跑通。
2. GPU 训练流程已跑通。
3. val OA 从初始约 `56%` 提升到 `65.5%`，说明小规模训练有效。
4. test OA 只有 `27.3%`，说明泛化仍不稳定。
5. 继续在同一小数据集上堆 epoch 不太值得，后续应优先扩展并平衡 train/val/test 的日期、月份、云量和观测条件。

## 8. 推荐下一步

### 8.1 不要立刻全量长训

原因：当前 test OA 偏低，说明 split 分布差异较大。应该先构建更可靠的小到中等规模泛化集合。

### 8.2 建议下一轮数据构成

建议每个 split 至少覆盖多个不同月份和日期：

- train：每月 2 天左右，先覆盖 1–6 月
- val：每月 1 天，不与 train 相邻太近
- test：保留独立月份/日期，暂时不要用于调参

### 8.3 下一轮目标

优先目标：

- val OA 稳定超过 `70%`
- test OA 不再大幅塌陷，至少接近 val OA 的趋势

如果 val OA 提升但 test OA 仍低，需要检查：

- test 场景是否云相类别分布异常
- train/val/test 的地理覆盖是否差异过大
- SZA/VZA 过滤后是否样本分布不均
- MYD06 CER/COT 在部分日期大量为 0 或 NaN 是否影响学习

## 9. 常用命令

### 9.1 激活环境

```bash
source /home/yuq/anaconda3/bin/activate cloudunet
export LD_LIBRARY_PATH=/home/yuq/anaconda3/envs/cloudunet/lib:$LD_LIBRARY_PATH
```

### 9.2 检查 GPU

```bash
nvidia-smi
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
PY
```

### 9.3 跑融合主流程示例

```bash
python main.py --stages fuse --split train --day 20190105 --workers 4 --max_qc 0
```

### 9.4 跑 stats / train 主流程示例

```bash
python main.py --stages stats train --split train
```

注意：当前小规模训练主要是通过临时 Python 脚本覆盖 `cfg.PAIRED_*`、`cfg.MODEL_DIR` 等路径运行的，避免污染正式目录。

## 10. 当前 Git 状态说明

没有执行过 `git commit`，也没有上传 GitHub。

当前源码本地修改主要包括：

- `config.py`
- `fusion_io.py`
- `fusion_core.py`
- `data_fusion.py`
- `main.py`
- `train.py`

本地训练产物在：

```text
unet_workdir/
```

最终是否保留这些代码改动、是否清理临时训练产物，需要用户自行敲定。
