# """
# quickstart.py----Train → Chat → Evaluate
# """
# import os, sys
# sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# # Step 1: Generate training data
# print("\n▸ Step 1: Generating training data...")
# from scripts.download_data import create_pretrain_data, create_sft_data
# create_pretrain_data()
# create_sft_data()
#
#
# # Step 2: Test model creation
# print("\n▸ Step 2: Creating model...")
# import torch
# from model.model_miniagent import MiniAgentModel, MiniAgentConfig
# config = MiniAgentConfig(hidden_size=64, num_hidden_layers=2,
#                          num_attention_heads=2, num_key_value_heads=1,
#                          intermediate_size=128, vocab_size=256)
# model = MiniAgentModel(config)
# print(f"   Model: {model.count_params():.2f}M params")
# # Step 3: Test forward pass
# print("\n▸ Step 3: Testing forward pass...")
# x = torch.randint(0, 256, (1, 32))
# out = model(x, labels=x)
# print(f"   Loss: {out['loss']:.4f}")
# # Step 4: Test generation
# print("\n▸ Step 4: Testing generation...")
# model.eval()
# prompt = torch.randint(0, 256, (1, 8))
# generated = model.generate(prompt, max_new_tokens=16)
# print(f"   Generated {generated.shape[1]} tokens")
#
#
# # Step 5: Run benchmarks
# print("\n▸ Step 5: Running benchmarks...")
# from eval.advertising_bench import run_gaql_eval, run_ppc_math_eval, run_cross_platform_eval
# for name, fn in [("GAQL", run_gaql_eval), ("PPC-Math", run_ppc_math_eval), ("Cross-Platform", run_cross_platform_eval)]:
#     r = fn(model=None)
#     print(f"   {name}: {r.score:.0f}% ({r.correct}/{r.total})")


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
                # 1.13.1
                # True
                # 11.7
                # CUDA版本: 11.7
                # GPU名称: NVIDIA GeForce RTX 3090


import time
start = time.time()

# 模拟耗时操作
time.sleep(1.5)

end = time.time()
diff = end - start
print(f"耗时: {diff:.2f} 秒")


