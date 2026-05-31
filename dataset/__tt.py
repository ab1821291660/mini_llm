
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.version.cuda)      # 查看其依赖的CUDA版本
# print(f"PyTorch版本: {torch.__version__}")
# print(f"GPU可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"GPU名称: {torch.cuda.get_device_name(0)}")
    # 简单GPU运算测试
    a = torch.randn(1000, 1000).cuda()
    b = torch.randn(1000, 1000).cuda()
    c = a @ b
    print(f"✅ GPU 运算测试通过，结果张量尺寸: {c.size()}")
    #2.6.0+cpu
    # False
    # None
# 2.6.0+cu124
# True
# 12.4
# CUDA版本: 12.4
# GPU名称: NVIDIA GeForce RTX 3090
# ✅ GPU 运算测试通过，结果张量尺寸: torch.Size([1000, 1000])
                # True
                # 1.13.1
                # PyTorch版本: 1.13.1
                # GPU可用: True
                # CUDA版本: 11.7
                # GPU名称: NVIDIA GeForce RTX 3090


# modelscope download --model gongjy/minimind-3 --local_dir ./dir
# # 方式1：使用 Transformers 格式模型
# python eval_llm.py --load_from ./minimind-3
# # 方式2：基于 PyTorch 模型（确保./out目录下有对应权重）
# python eval_llm.py --load_from ./model --weight full_sft
# python eval_llm.py --weight full_sft   #用于指定权重名称前缀，例如 pretrain、full_sft 等
#
#
# 注：其它须知
# 1、所有训练脚本均基于 PyTorch 原生实现，并支持多卡加速。
# 2、若你的设备有 N (N > 1) 张显卡，可通过以下方式启动单机 N 卡训练（DDP，也支持扩展到多机多卡）：
# torchrun --nproc_per_node N train_xxx.py
# 3、可根据需要开启 wandb 记录训练过程。
# ... train_xxx.py --use_wandb
# 2025 年 6 月后，国内网络环境通常无法直连 WandB。MiniMind 当前默认转为使用 SwanLab 作为训练可视化工具，其接口与 WandB 基本兼容；通常只需将 import wandb 替换为 import swanlab as wandb，其余调用方式基本无需改动。
#
#
# git clone --depth 1 https://github.com/jingyaogong/minimind
# cd minimind && pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple
#
# pip install modelscope -i https://mirrors.aliyun.com/pypi/simple
# modelscope download --model gongjy/minimind-3-moe --local_dir ./minimind-3-moe
# modelscope download --model gongjy/minimind-3 --local_dir ./minimind-3
#
#
# pip install nvitop
# pip install transformers==4.57.6  -i https://mirrors.aliyun.com/pypi/simple
# pip install datasets==3.6.0  -i https://mirrors.aliyun.com/pypi/simple
# pip install --upgrade datasets -i https://mirrors.aliyun.com/pypi/simple

# # 如果你还没有安装 uv
# pip install uv
# # 使用 uv 安装 PyTorch (CUDA 13.1 版)
# uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu131


# # 在命令行中执行以下命令
# pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124 -i https://mirrors.aliyun.com/pypi/simple
# pip install numpy==1.26.4 -i https://mirrors.aliyun.com/pypi/simple
# pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124


# conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y -i https://mirrors.aliyun.com/pypi/simple
# pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
#
#
# ##===================================
# ##===================================
# 当前 sft_t2t / sft_t2t_mini 已经混入 Tool Call 数据，因此通常不需要再额外做一轮独立的 Tool Calling 监督微调。


# ##===================================
# ##===================================
# # 部署时
# pip install pydantic  -i https://mirrors.aliyun.com/pypi/simple

