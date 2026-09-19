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

注意：本仓库不包含权重（best_model/*.pth）和数据集，请参考 README.md 的「权重与数据」一节自行放置；
更完整的训练/推理说明同样见 README.md。


