# SAM Studio

同济软件工程24级智能计算系统期末项目。

基于 SAM 2.1 Tiny 的本地图像与短视频交互分割。支持正负点、框和文字定位，视频关键帧修正，透明前景与二值掩膜导出，以及卡通和线稿风格化。

[打开 GitHub Pages 在线演示](https://kinosakuraxuan.github.io/SAM-Studio/) · [实验记录](./experiments)

在线页面使用预计算样例。上传本地媒体、重新分割或运行 Grounding DINO 和风格模型需要启动下面的 Python 应用。

## 本地安装

以下为 Windows PowerShell 示例，项目在 Python 3.12、PyTorch 2.6.0 和 CUDA 12.4 上验证过。CPU 可用于基础图像推理，视频建议用 CUDA。

```powershell
git clone https://github.com/kinosakuraxuan/SAM-Studio.git
cd SAM-Studio
py -3.12 -m venv .runtime
.\.runtime\Scripts\python.exe -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
.\.runtime\Scripts\python.exe -m pip install -r requirements.txt
git clone https://github.com/facebookresearch/sam2.git vendor/sam2
git -C vendor/sam2 checkout 2b90b9f5ceec907a1c18123530e92e794ad901a4
New-Item -ItemType Directory -Force models | Out-Null
curl.exe --fail --location 'https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt' --output models/sam2.1_hiera_tiny.pt
.\start.ps1
```

打开 [本机工作台](http://127.0.0.1:8765/)。若选择 CPU，请把安装 PyTorch 的 `cu124` 索引改为 `cpu`。

- **文字提示**：运行 `.\.runtime\Scripts\python.exe -m tools.fetch_dino` 安装 Grounding DINO Tiny，完成后重启应用。
- **神经风格化**：运行 `.\.runtime\Scripts\python.exe -m tools.fetch_styles`。
- **训练权重**：已训练的解码器权重放入 outputs/training/best.pt，边界细化权重放入 outputs/refinement-training/best.pt。也可以按下文的 DAVIS 训练流程生成这些文件。浏览器根据已存在的权重显示可选版本。车轮联合微调是离线实验。
- **复现实验**：下载 [DAVIS 2017](https://davischallenge.org/davis2017/code.html#downloads) 到 `data/`，按下文解包并生成划分。用 `git clone https://github.com/davisvideochallenge/davis2017-evaluation.git vendor/davis2017-evaluation` 安装 J/F 指标，记录使用提交 `ac7c43fca936f9722837b7fbd337d284ba37004b`。

## 在线样例素材

演示使用 DAVIS 2017 的 `car-turn`、`tennis`，本地内置样例另有 `camel`。图像和视频仅用于非商业课程展示，修改包括裁剪、掩膜、风格化和格式转换。车轮对照使用 `bike-packing` 标注和项目预测，页面附有数据出处及许可。

[DAVIS 2017](https://davischallenge.org/davis2017/code.html) · [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) · [DAVIS 标注条款](https://davischallenge.org/challenge2017/rulesdates.html) · [演示素材说明](./docs/credits.html)

---

## 本地工作台与实验说明


从本地照片或视频中提取单个目标。支持前景点、背景点、框选、文字定位和视频中途修正。SAM 2.1 Tiny 在 GPU 上分割与跟踪，Grounding DINO Tiny 在 CPU 上把描述转换成候选框。中文词表支持“汽车”“自行车”“红色背包”等词，也可以直接输入英文短语。权重和分词器本地加载。

## 在这台电脑上运行

在本目录打开 PowerShell，执行下方命令，浏览器访问 [本机应用](http://127.0.0.1:8765)。服务已经运行时可直接打开该地址。终端中按 Ctrl+C 可停止前台启动的服务。

```powershell
.\.runtime\Scripts\python.exe -m uvicorn sam_app.server:app --host 127.0.0.1 --port 8765
```

1. 点击“打开图片”或“打开视频”选择本地素材，也可以在画布下方点“使用示例”，再选择示例图或示例视频。视频入口接受 MP4、MOV、AVI、MKV、WebM、M4V，文件需包含本机 PyAV 能解码的视频流。导入时会逐帧读取，完成后显示首帧与时间轴。画布默认完整显示，可在预览右上角选择“适合宽度”查看长图，或选择“适合高度”查看超宽素材，再滚动画布处理细节。使用画布下方的前景点、背景点和框选工具选择目标。在视频时间轴上可以逐帧前进或后退。选择“透明 PNG”导出前景，“二值掩膜”导出标签。
2. 输入“汽车”并点“查找”，确认一个候选框。含多个目标的画面由用户决定要提取哪一个。
3. 从“使用示例”选择示例视频，或在本机服务运行时直接打开[汽车转弯示例](http://127.0.0.1:8765/?sample=video)。在首帧框选汽车，点击“网页播放”进入“跟随选区”。播放器边算边播放并支持暂停、回看已处理帧和切换掩膜、卡通、速写。跟随预览无声，速度随本机推理耗时变化。“逐帧编辑”保留先前的编辑帧。需要文件时再点“传播并导出视频”，导出会复用同组选区提示已算出的掩膜。原片仍可从播放菜单打开，遇到浏览器无法解码的格式时会生成无声兼容预览。
4. 在“模型权重”中切换原版与微调版，使用相同提示查看效果。微调选项在存在正式训练权重时可用。
5. 在“选区风格”选择“卡通 · 神经网络”或“速写 · 神经网络”，拖动强度滑块，在原背景上预览目标。可分别下载风格合成 PNG 和透明前景。
6. 视频先完成一次传播，再选风格并点击“生成风格视频”。切换风格会复用已有逐帧掩膜。可以在网页播放菜单切到风格视频，也可以下载无声 MP4。无损风格帧 ZIP 保留源视频解码后掩膜外的每个 RGB 像素。

### 模型与验证

本机的 `models/whitebox.pth` 和 `models/lineart.pth` 分别提供卡通与黑白线稿推理。另一台机器可用下方命令获取并校验两份权重。[White-box PyTorch 实现](https://github.com/vinesmsuic/White-box-Cartoonization-PyTorch) · [Informative Drawings](https://github.com/carolineec/informative-drawings)

```powershell
.\.runtime\Scripts\python.exe -m tools.fetch_styles
.\.runtime\Scripts\python.exe -m tools.verify_styles
```

验证命令用独立应用实例运行，无需预先启动工作台。它对两张 DAVIS 图像、透明 PNG 和汽车转弯视频运行两种神经风格，检查掩膜外像素、透明度、帧数和时间戳。合成 PNG、视频与 `verification.json` 保存到 `outputs/neural-style-demo`。

演示素材位于 `outputs/demo`。其中 `car-turn.mp4` 由 DAVIS 开发集的 80 帧生成，时长 10 秒。`preview.mp4` 与 `foreground.zip` 是中途修正后导出的结果。单击车轮等局部可能得到零件轮廓，框住整车可以更明确地指定对象。

## 训练与评测

数据位于 `data/DAVIS`。按整段视频划分的清单位于 `data/splits.json`。官方 train 中的 50 段用于训练，10 段用于开发，官方 val 的 30 段留作最终评测。按图像文件随机划分会把相邻帧混到训练与评测中。

```powershell
# 校验 DAVIS 并生成固定划分
.\.runtime\Scripts\python.exe -m training.data

# 小样本验证，保存到独立的新目录
.\.runtime\Scripts\python.exe -m training.train_mask_decoder --limit 8 --per-video 1 --epochs 2 --out outputs/my-smoke

# 正式训练，默认每视频最多 10 帧、10 轮、梯度累积 4 次
.\.runtime\Scripts\python.exe -m training.train_mask_decoder

# 成对比较原版和最佳解码器。固定提示随结果保存
.\.runtime\Scripts\python.exe -m training.evaluate_images --split eval --frames 3

# 按 DAVIS 作者的首帧文字描述评测，使用分数最高的检测框
.\.runtime\Scripts\python.exe -m training.evaluate_text --split dev --version original

# GPU 空闲时分别测重新编码与缓存后补点的耗时
.\.runtime\Scripts\python.exe -m tools.benchmark

# 在同一序列上用 DAVIS 官方 J/F 函数衡量首帧框选与一次中途修正
.\.runtime\Scripts\python.exe -m training.evaluate_video --split dev --sequence car-turn --version original
```

正式训练输出为 `outputs/training/history.csv`、`config.json` 和按开发集 IoU 选择的 `best.pt`。每次运行选择新的输出目录以保留已有记录。界面从 `outputs/training/best.pt` 加载正式权重。图片对照保存逐帧 CSV、固定提示和按视频平均的 IoU、Dice。文字实验把无候选的查询记为零。

### 已完成的训练与图片对照

初次正式实验训练了 10 轮。按开发集平均 IoU 选中第 2 轮的权重，随后在 30 段保留视频中每段抽取 3 帧、每帧取 1 个实例，以相同的提示比较两组模型。下表给出按视频平均的 IoU。原始记录见 `outputs/paired-evaluation/summary.json` 及同目录的逐帧 CSV。

| 提示 | 原版 | 微调版 |
|---|---:|---:|
| 目标内部单点 | 0.553 | 0.775 |
| 紧贴目标的框 | 0.834 | 0.826 |
| 框边扰动上限 10% | 0.775 | 0.823 |
| 框边扰动上限 20% | 0.597 | 0.793 |

在这组样本和固定提示下，单点和两档粗框均有改善，紧框的平均 IoU 下降约 0.008。风格编辑复用最终选定的掩膜，目标选择偏差也会反映在编辑区域中。

### 卷积边界细化实验

工作台「模型权重」中的「边界细化 · 实验版」基于同一套微调权重，附加一个 3,913 参数的细化分支。它把最高分辨率的解码器特征和初始掩膜 logits 作为输入，仅在初始轮廓附近叠加残差。训练时冻结原有参数，500 帧训练 5 轮，以 100 帧开发集的边界 F 选择第 5 轮。权重位于 `outputs/refinement-training/best.pt`。

在原来 90 帧、相同四种提示的对照中，紧框边界 F 从 0.90061 到 0.90110，四种提示的边界 F 变化均小于 0.0005，IoU 变化均小于 0.0003。单张图、同一个框、缓存图像特征后交替测试两版 60 次，配对耗时增量中位数约为 0.98 ms。现有结果支持把它作为结构对照实验保留。[分割和边界结果](outputs/refinement-evaluation/summary.json) · [交替耗时记录](outputs/refinement-evaluation/paired_latency.json)

```powershell
.\.runtime\Scripts\python.exe -m training.train_mask_decoder --refine --epochs 5 --out outputs/repro-refinement
.\.runtime\Scripts\python.exe -m training.evaluate_images --split eval --frames 3 --include-refiner --refiner-checkpoint outputs/repro-refinement/best.pt --out outputs/repro-refinement-eval
.\.runtime\Scripts\python.exe -m tools.benchmark --compare-refiner --repeat 60 --out outputs/repro-refinement-latency.json
```

耗时命令比较工作台加载的两份权重，轮流运行微调版与细化版，并单独汇报配对耗时差。图片指标按视频汇总，表中的分组结果来自完全相同的提示。

### 原图像素网格的 U 型细化实验

两组都以当前微调版 SAM 生成粗掩膜 logits。细化头以原图坐标下的 128×128 裁剪训练，输出原像素网格上的掩膜残差。每个通过筛选的训练提示分别抽取细结构、轮廓和普通区域。训练筛掉粗掩膜 IoU 低于 0.45 的提示，正式评测保留全部固定提示。两组用相同的 2,661 个裁剪、模型初始化和训练顺序，去掉 RGB 的对照组把图像通道置零。以开发集 F@2 选择权重。

```powershell
.\.runtime\Scripts\python.exe -m training.prepare_pixel_refiner --out outputs/repro-pixel-refiner
.\.runtime\Scripts\python.exe -m training.train_pixel_refiner --epochs 5 --out outputs/repro-pixel-refiner
.\.runtime\Scripts\python.exe -m training.evaluate_pixel_refiner --out outputs/repro-pixel-refiner
```

正式权重和训练记录保存于 `outputs/pixel-refiner/`。正式评测使用原来 30 段保留视频的 90 帧及四种存档提示，报告 `evaluation/summary.json` 和逐样本 CSV。F@1 与 F@2 的容差分别是 1 和 2 个原图像素。细结构召回衡量 5×5 椭圆开运算剔除的标注前景，作为细小突起和线状结构的近似指标。该阶段使用本地 DAVIS 480p 掩膜，未另外制作发丝或车轮零件级标注。

| 版本 | IoU | F@1 | F@2 | 细结构召回 |
|---|---:|---:|---:|---:|
| 微调解码器 | 0.8045 | 0.6319 | 0.7328 | 0.4294 |
| 低分辨率细化 | 0.8044 | 0.6320 | 0.7326 | 0.4157 |
| U 型头，去掉 RGB | 0.8015 | 0.6237 | 0.7233 | 0.4721 |
| U 型头，输入 RGB | 0.8011 | 0.6153 | 0.7177 | 0.3916 |

上表先按视频汇总，再对四种提示等权平均。去掉 RGB 的 U 型头提高了细结构召回，边界精度和整图 IoU 下降。输入 RGB 的版本在这批数据上各项指标均低于微调解码器。本轮保留两组权重和完整评测记录，联合微调阶段仍待局部细化取得边界收益后再开展。

### 全分辨率与 480p 输入对照

将 DAVIS Semi-supervised Full-Resolution TrainVal 压缩包放在 `data/DAVIS-2017-trainval-Full-Resolution.zip`。程序直接读取配对帧和标注，沿用上一轮的帧号、实例和点框，将两组裁剪对齐到同一个全分辨率像素网格。全分辨率组读原图 RGB。480p 组先降低同一图像的分辨率，再映射回共同网格。两组共用冻结 SAM 生成的粗掩膜、训练与评测标签、初始化、采样、5 轮训练和优化设置。这个实验测量 RGB 输入细节对 U 型细化头的影响。[DAVIS 官方下载](https://davischallenge.org/davis2017/code.html#downloads)

```powershell
.\.runtime\Scripts\python.exe -m training.compare_davis_resolution prepare
.\.runtime\Scripts\python.exe -m training.compare_davis_resolution train
.\.runtime\Scripts\python.exe -m training.compare_davis_resolution evaluate
.\.runtime\Scripts\python.exe -m tools.plot_davis_resolution
```

结果保存在 `outputs/davis-resolution/evaluation/`，包括全分辨率评测分数、逐提示 CSV 和之前的自行车轮局部对照图。

本轮共使用 50 段视频的 470 帧和 2,634 个配对训练裁剪。两组统一评分的 90 帧来自 30 段保留视频。下表按视频汇总，对四种提示等权平均。

| RGB 输入 | IoU | 全分辨率 F@2 | 细结构召回 |
|---|---:|---:|---:|
| 480p | 0.80290 | 0.51680 | 30.76% |
| 全分辨率 | 0.80274 | 0.51684 | 30.55% |

在自行车轮紧框局部，全分辨率版与 480p 版都覆盖了 5,378 个标注细结构像素中的 5,270 个。全分辨率版新覆盖 3 个并漏掉 3 个，辐条间的空隙仍有大片误分。输入细节增加后，这次训练未观察到稳定的细结构分割收益。[评测摘要](outputs/davis-resolution/evaluation/summary.json) · [车轮放大图](outputs/davis-resolution/evaluation/wheel-resolution-comparison.png)

## 代码入口

| 文件 | 职责 |
|---|---|
| `sam_app/server.py`、`sam_app/web/` | 本地接口、交互画布、候选选择、权重切换与导出 |
| `sam_app/models.py`、`sam_app/video.py` | 模型加载、坐标、特征复用、视频传播与逐帧导出 |
| `sam_app/neural_styles.py`、`sam_app/styles.py`、`tools/verify_styles.py` | 两种神经网络推理、掩膜内合成，以及背景、透明度与导出核验 |
| `training/data.py` | 划分校验、抽帧、实例掩膜和提示生成 |
| `training/train_mask_decoder.py`、`sam_app/refinement.py` | 微调掩膜解码器，或冻结原参数并训练卷积边界细化分支 |
| `training/evaluate_images.py`、`training/evaluate_text.py` | 有固定提示或公开描述的独立评测 |
| `tools/verify_app.py` | 用真实图片、视频检查 HTTP 接口、掩膜和导出帧数 |

## 在另一台机器上准备环境

本机已验证 Python 3.12、PyTorch 2.6.0+cu124 和 torchvision 0.21.0+cu124。选择与目标机器驱动兼容的官方 PyTorch 构建，安装同版本的 torchvision，然后执行以下步骤。Windows 可用原生 Python 或 WSL2；GPU 可用性以 `torch.cuda.is_available()` 为准。

```powershell
python -m venv .runtime
.\.runtime\Scripts\python.exe -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
.\.runtime\Scripts\python.exe -m pip install -r requirements.txt
git clone https://github.com/facebookresearch/sam2.git vendor/sam2
git -C vendor/sam2 checkout 2b90b9f5ceec907a1c18123530e92e794ad901a4
git clone https://github.com/davisvideochallenge/davis2017-evaluation.git vendor/davis2017-evaluation
git -C vendor/davis2017-evaluation checkout ac7c43fca936f9722837b7fbd337d284ba37004b
```

- 将 `sam2.1_hiera_tiny.pt` 放入 `models`。SHA-256 为 `7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69`。[SAM 官方权重](https://huggingface.co/facebook/sam2.1-hiera-tiny/tree/main)
- 将 Grounding DINO 的配置、处理器、分词器和 `model.safetensors` 放入 `models/grounding-dino-tiny`，也可运行 `python -m tools.fetch_dino` 断点下载并校验发布方摘要。[Grounding DINO 官方文件](https://huggingface.co/IDEA-Research/grounding-dino-tiny/tree/main)
- 下载 Semi-supervised TrainVal 480p，在项目根目录运行 `python tools/unpack_davis.py <压缩包路径>`。将文字标注压缩包放入 `data/davis_text_annotations.zip`。[DAVIS 数据](https://davischallenge.org/davis2017/code.html) · [配套文字描述](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/video-segmentation/video-object-segmentation-with-language-referring-expressions/)
