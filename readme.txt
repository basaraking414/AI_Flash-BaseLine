环境配置：
构建python=3.13版本的虚拟环境。然后使用pip install -r requirements.txt指令安装相应的依赖项。
安装后即可使用cpu进行inference.py的运行推理。

我的本地的模型训练和测试，都在GTX4060 8GB显卡上进行，因此对应的安装cuda指令如下：
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
如需使用cuda加速，请按照设备对应的版本安装对应的cuda。

推理运行：
inference运行之前，需要将待处理的图像数据集，保存在dataset/test内。需要图像文件夹input和mask文件夹mask_personmask。
运行 `python inference.py`，输出结果默认保存在 result/test 文件夹内（可用 --output_dir 修改）。
跑分时，按照一致的结构放置图片文件即可。

注意：本仓库只发布论文主模型（Restormer + DC-LIDM）的代码与权重，权重（best_Restormer_LIDM.pth）
从 Releases 下载后放入 best_model/；数据集来自 NTIRE 2026 AI Flash Portrait 挑战赛，请按其公开渠道获取。
更完整的说明见 README.md（英文）与 README_zh-CN.md（中文）：
训练 `python -m basicsr.my_train -opt configs/my_aiflash.yaml`（冒烟测试用 configs/smoke_val1.yaml），
评测 `python -m basicsr.test -opt configs/eval_val50.yaml`（复现论文 Table I 的 Ours 行）。


