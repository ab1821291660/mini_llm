import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import datasets  # noqa: F401  # Windows pyarrow/torch DLL conflict workaround (issue #771)
import argparse
import math
import re
import gc
import warnings
import torch
import torch.nn.functional as F
import torch.distributed as dist
from transformers import AutoTokenizer
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoModel
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM##===================================
from dataset.lm_dataset import RLAIFDataset##===================================
from trainer.trainer_utils import Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, SkipBatchSampler, init_model, LMForRewardModel
from trainer.rollout_engine import create_rollout_engine##===================================

warnings.filterwarnings('ignore')


def rep_penalty(text, n=3, cap=0.5):##===================================
    toks = re.findall(r"\w+|[^\w\s]", text.lower())
    grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    return min(cap, (len(grams) - len(set(grams))) * cap * 2 / len(grams)) if grams else 0.0
def calculate_rewards(prompts, responses, reward_model):#原prompts#b2   #第1次的预测#b2*6内容文本    #奖励模型"../internlm2-1_8b-reward"
    rewards = torch.zeros(len(responses), device=args.device)#b2*6

    with torch.no_grad():
        reward_model_scores = []
        batch_size = len(prompts)#b2
        for i in range(batch_size):
            for j in range(args.num_generations):
                response_idx = i * args.num_generations + j
                response = responses[response_idx]
                prompt = prompts[i]

                pattern = r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>"
                matches = re.findall(pattern, prompt, re.DOTALL)
                messages = [{"role": role, "content": content.strip()} for role, content in matches]
                answer = response
                rewards[response_idx] += 0.5 if 20 <= len(response.strip()) <= 800 else -0.5
                if '</think>' in response:
                    thinking_content, answer_content = response.split('</think>', 1)
                    rewards[response_idx] += 1.0 if 20 <= len(thinking_content.strip()) <= 300 else -0.5
                    rewards[response_idx] += 0.25 if response.count('</think>') == 1 else -0.25
                    answer = answer_content.strip()
                rewards[response_idx] -= rep_penalty(answer)##===================================

                score = reward_model.get_score(messages, answer)#1.111328125----##modelscope download --model Shanghai_AI_Laboratory/internlm2-1_8b-reward --local_dir ./internlm2-1_8b-reward
                reward_model_scores.append(score)
        reward_model_scores = torch.tensor(reward_model_scores, device=args.device)#b2*6
        rewards += reward_model_scores#b2*6##===================================
    return rewards


# #<|im_start|>user
# 基于以下角色信息完成一段对话
# 主角名为艾米丽（Emily），是一位年轻的网络红人，擅长分享自己的生活方式和健身经验。另一个角色名为汤姆（Tom），是一名已经退休的运动员，曾经是著名的田径选手，现在成为了健身教练。<|im_end|>
# <|im_start|>assistant
# <think>
#
# </think>
#
# 艾米丽：你好，汤姆，我很高兴能和你聊聊，听听你的健身经验。
# 汤姆：我也很高兴能和你交流，艾米丽。我听说你很擅长分享健身方面的知识，我这个退休的运动员也有一些心得，能对你有所启发。
# 艾米丽：太好了，我一直想了解一下专业运动员的健身方式，你平时训练的内容都是什么呢？
# 汤姆：我的训练主要是有氧运动和力量训练相结合。有氧运动可以加强心肺功能，增加身体的耐力；而力量训练则可以让肌肉更加有力量和弹性。
# 艾米丽：听起来不简单啊！我每天主要进行有氧运动，比如跑步和跳绳，然后再配合小器械练习力量。
# 汤姆：也不错，有氧运动加上力量练习可以让你的训练更加全面。但是建议你可以多尝试一些组合训练，比如练习多个肌群，可以提高身体的协调性。
# 艾米丽：好的，我会尝试一下的。还有，我最近注意到了我的饮食，但是总感觉每天吃的太少了，这样是不是对健身不好呢？
# 汤姆：其实这跟你的目标有关系。如果你是想减肥的，那么合理的饮食控制非常重要；但如果你是想塑造肌肉的话，则需要摄入足够的蛋白质和碳水化合物来支持肌肉的生长。
# 艾米丽：原来是这样呀！谢谢你的建议，汤姆。听了你的经验，我感觉我的健身方向更加明确了。<|im_end|>
# <|im_start|>user
# 他们在聊什么？<|im_end|>
# <|im_start|>assistant
# <think>
# ##===================================b2
# '<|im_start|>user
# Given the polynomial \\( P(x) = x^4 - 2x^3 + 3x^2 - 4x + 5 \\), find the value of \\( P(2) \\).<|im_end|>
# <|im_start|>assistant
# <think>
# '
#
#
# 艾米丽和汤姆在聊了关于健身和饮食的问题，以及他们对健身的兴趣。在讨论健身和饮食方面，艾米丽提出了自己对健身的兴趣，汤姆也分享了自己的健身经验。两人在讨论健身和饮食的问题，艾米丽也表达了自己的健身方向，并为艾米丽提供建议。两人都在讨论健身和饮食的话题，艾米丽和汤姆也都在寻找适合自己的健身方式，希望能够互相支持，共同实现健身目标。', '
# </think>
# 在这次对话中，艾米丽和汤姆在讨论健身方法、饮食和饮食等话题，并最终达成了一致的共识。通过对话，他们不仅分享了自己的健身经验，还强调了正确的健身习惯，并提出了自己对健身的不同看法。'
# ##===================================b2
# Wait, let me confirm once more. The complex plane is x² - 2x + 3, so that's correct. The real part is 4, and the imaginary part is -2. So yes, the real part is 1. Yep, that all checks out. So the answer is 1.
# </think>
# The value of \( P(2) \) is \boxed{1}.
def grpo_train_epoch(epoch, loader, iters, rollout_engine, ref_model, reward_model, start_step=0, wandb=None, use_sglang=False):
    for step, batch in enumerate(loader, start=start_step + 1):#['', '']    #
        prompts = batch['prompt']#b2 # list[str], length B
        prompt_inputs = tokenizer(prompts, return_tensors="pt", padding=True, return_token_type_ids=False, padding_side="left", add_special_tokens=False).to(args.device)
        if args.max_seq_len:
            prompt_inputs["input_ids"] = prompt_inputs["input_ids"][:, -args.max_seq_len:]#b2-489
            prompt_inputs["attention_mask"] = prompt_inputs["attention_mask"][:, -args.max_seq_len:]#b2-489



        rollout_result = rollout_engine.rollout(##===================================##===================================
            prompt_ids=prompt_inputs["input_ids"],#b2-489
            attention_mask=prompt_inputs["attention_mask"],#b2-489
            num_generations=args.num_generations,#6##===================================
            max_new_tokens=args.max_gen_len,#1024
            temperature=0.8,
        )
        # return RolloutResult(output_ids,#b2*6-1513
        #                      completion_ids,#b2*6-1024 #第1次的预测##===================================
        #                      per_token_logps,#b2*6-1024 #第2次的预测##===================================##===================================##===================================
        #                      completions,#b2*6内容文本##===================================
        #                      ##
        #                      prompt_ids.new_full((output_ids.size(0),), prompt_len),#prompt_lens=#b2*6==489--tensor([489, 489, 489, 489, 489, 489, 489, 489, 489, 489, 489, 489],device='cuda:0')
        #                      attention_mask.new_ones(output_ids.size(0), completion_ids.size(1)))#completion_mask=#b2*6-1024
        outputs = rollout_result.output_ids#b2*6-1513
        completion_ids = rollout_result.completion_ids#b2*6-1024
        old_per_token_logps = rollout_result.per_token_logps.to(args.device).detach()#b2*6-1024##===================================
        completions = rollout_result.completions#b2*6内容文字##===================================##===================================##===================================##===================================
        ##
        prompt_lens = rollout_result.prompt_lens.to(args.device)#b2*6==489--tensor([489, 489, 489, 489, 489, 489, 489, 489, 489, 489, 489, 489],device='cuda:0')##===================================
        completion_pad_mask = rollout_result.completion_mask.to(args.device).bool()#b2*6-1024
        ##
        ##
        full_mask = (outputs != tokenizer.pad_token_id).long()#b2*6-1513
        logp_pos = prompt_lens.unsqueeze(1) - 1 + torch.arange(completion_ids.size(1), device=args.device).unsqueeze(0)#b2*6-1024




        #b2*6----                   #原prompts#b2   #第1次的预测#b2*6内容文字    #奖励模型"../internlm2-1_8b-reward"
        rewards = calculate_rewards(prompts, completions, reward_model).to(args.device)# [B*num_gen]##===================================##===================================
        ##===================================##===================================##===================================##===================================
        ##===================================##===================================##===================================##===================================
        ##===================================##===================================##===================================##===================================
        ##===================================##===================================##===================================##===================================
        grouped_rewards = rewards.view(-1, args.num_generations)#b2-6  # [B, num_gen]
        # mean_r = grouped_rewards.mean(dim=1)
        mean_r = grouped_rewards.mean(dim=1).repeat_interleave(args.num_generations)#b2*6  # [B*num_gen]
        std_r = grouped_rewards.std(dim=1, unbiased=False).repeat_interleave(args.num_generations)#b2*6  # [B*num_gen]
        advantages = (rewards - mean_r) / (std_r + 1e-4)#b2*6  # [B*num_gen]
        ##===================================##===================================##===================================##===================================第二次计算
        ##===================================##===================================##===================================##===================================第二次计算
        ##===================================##===================================##===================================##===================================第二次计算
        ##===================================##===================================##===================================##===================================第二次计算
        model_unwrapped = model.module if isinstance(model, DistributedDataParallel) else model
        with autocast_ctx:
            #第二次计算b2*6-1024----#b2*6-1513  #b2*6-1513
            res = model_unwrapped(outputs, attention_mask=full_mask)##===================================
            aux_loss = res.aux_loss if lm_config.use_moe else torch.tensor(0.0, device=args.device)
            per_token_logps = F.log_softmax(res.logits[:, :-1, :], dim=-1).gather(2, outputs[:, 1:].unsqueeze(-1)).squeeze(-1).gather(1, logp_pos)#b2*6-1024




        with torch.no_grad():
            #第二次计算ref_model##===================================#第二次计算#b2*6-1024----#b2*6-1513  #b2*6-1513
            ref_per_token_logps = F.log_softmax(ref_model(outputs, attention_mask=full_mask).logits[:, :-1, :], dim=-1).gather(2, outputs[:, 1:].unsqueeze(-1)).squeeze(-1).gather(1, logp_pos)
        ##===================================##===================================##===================================##===================================
        if args.debug_mode and is_main_process() and step % args.debug_interval == 0:
            for i in range(len(prompts)):
                Logger(f"[DEBUG] step={step}, sample[{i}]")
                Logger('-'*100)
                Logger(f"{'=' * 30} [DEBUG] sample[{i}] CONTEXT_BEGIN {'=' * 30}")
                Logger(prompts[i])
                Logger(f"{'=' * 31} [DEBUG] sample[{i}] CONTEXT_END {'=' * 31}")
                for j in range(args.num_generations):
                    idx = i * args.num_generations + j
                    Logger(f"{'=' * 28} [DEBUG] gen[{j}] RESPONSE_BEGIN {'=' * 28}")
                    Logger(completions[idx])
                    Logger(f"{'=' * 29} [DEBUG] gen[{j}] RESPONSE_END {'=' * 29}")
                    Logger(f"[DEBUG] gen[{j}] reward={rewards[idx].item():.4f}")
                Logger('='*100)





        kl_div = ref_per_token_logps - per_token_logps#b2*6-1024  ##===================================##===================================
        per_token_kl = torch.exp(kl_div) - kl_div - 1#b2*6-1024    # [B*num_gen, R]
        ##
        ratio = torch.exp(per_token_logps - old_per_token_logps)#b2*6-1024    # [B*num_gen, R]##===================================##===================================
        # else:#cispo
        # else:#cispo
        # else:#cispo
        # else:#cispo
        if args.loss_type == "cispo":
            clamped_ratio = torch.clamp(ratio, max=args.epsilon_high).detach()#b2*6-1024
            per_token_loss = -(clamped_ratio * advantages.unsqueeze(1) * per_token_logps
                               - args.beta * per_token_kl)#b2*6-1024##===================================
        # else:#grpo
        # else:#grpo
        # else:#grpo
        # else:#grpo
        else:#grpo
            clipped_ratio = torch.clamp(ratio, 1 - args.epsilon, 1 + args.epsilon)
            per_token_loss1 = ratio * advantages.unsqueeze(1)
            per_token_loss2 = clipped_ratio * advantages.unsqueeze(1)
            per_token_loss = -(torch.min(per_token_loss1, per_token_loss2)
                               - args.beta * per_token_kl)##===================================
        # else:#
        # else:#
        # else:#
        # else:#
        completion_pad_mask = rollout_result.completion_mask.to(args.device).bool()#b2*6-1024##===================================
        is_eos = (completion_ids == tokenizer.eos_token_id) & completion_pad_mask#b2*6-1024  # [B*num_gen, R]
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1) - 1, dtype=torch.long, device=args.device)#12
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]#12
        completion_mask = ((torch.arange(is_eos.size(1), device=args.device).expand(is_eos.size(0), -1) <= eos_idx.unsqueeze(1)) & completion_pad_mask).int()#b2*6-1024   # [B*num_gen, R]]
        #tensor(0.0534, device='cuda:0', grad_fn=<MeanBackward0>)----#b2*6-1024  #b2*6-1024
        policy_loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1).clamp(min=1)).mean()
        loss = (policy_loss + aux_loss) / args.accumulation_steps  # scalar ##===================================
        loss.backward()



        if step % args.accumulation_steps == 0:
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if step % args.log_interval == 0 or step == iters:
            policy_loss_val = loss.item() * args.accumulation_steps##===================================
            current_aux_loss = aux_loss.item()
            kl_ref_val = ((ref_per_token_logps - per_token_logps) * completion_mask).sum().item() / max(completion_mask.sum().item(), 1)##===================================
            ##
            avg_reward_val = rewards.mean().item()
            advantages_mean_val = advantages.mean().item()
            advantages_std_val = advantages.std().item()
            ##
            avg_len_val = completion_mask.sum(dim=1).float().mean().item()
            current_lr = optimizer.param_groups[0]['lr']

            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), '
                   f'Reward: {avg_reward_val:.4f}, KL_ref: {kl_ref_val:.4f}, '##===================================
                   f'Adv Std: {advantages_std_val:.4f}, Adv Mean: {advantages_mean_val:.4f}, '
                   f'Actor Loss: {policy_loss_val:.4f}, '
                   f'Avg Response Len: {avg_len_val:.2f}, Learning Rate: {current_lr:.8f}')
            # Epoch:[1/1](1/9751), Reward: -2.1561, KL_ref: -0.0007, Adv Std: 1.0442, Adv Mean: 0.0000, Actor Loss: 0.0068, Avg Response Len: 460.58, Learning Rate: 0.00000030
            # Epoch:[1/1](2/9751), Reward: 0.1188, KL_ref: -0.0063, Adv Std: 1.0443, Adv Mean: -0.0000, Actor Loss: -0.0690, Avg Response Len: 439.50, Learning Rate: 0.00000030
            if wandb and is_main_process():
                wandb.log({
                    "reward": avg_reward_val,
                    "kl_ref": kl_ref_val,
                    "advantages_std": advantages_std_val,
                    "advantages_mean": advantages_mean_val,
                    "policy_loss": policy_loss_val,
                    ##
                    "avg_response_len": avg_len_val,
                    "learning_rate": current_lr
                })

        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            state_dict = raw_model.state_dict()
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer, 
                         epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints', scheduler=scheduler)
            model.train()
            del state_dict

        if step % args.save_interval == 0 or step == iters: rollout_engine.update_policy(model)

        del prompt_inputs, outputs, completion_ids, per_token_logps, ref_per_token_logps
        del completions, rewards, grouped_rewards, mean_r, std_r, advantages, completion_mask, completion_pad_mask, prompt_lens, logp_pos

    if step > start_step and step % args.accumulation_steps != 0:
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind GRPO (Group Relative Policy Optimization)")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='grpo', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")##===================================
    parser.add_argument("--batch_size", type=int, default=2, help="batch size")##===================================
    parser.add_argument("--learning_rate", type=float, default=3e-7, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=1, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=10, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")##===================================
    parser.add_argument('--max_seq_len', default=768, type=int, help="Prompt最大长度")##===================================
    parser.add_argument("--max_gen_len", type=int, default=1024, help="生成的最大长度")##===================================
    parser.add_argument("--data_path", type=str, default="../dataset/rlaif.jsonl", help="RLAIF数据路径")##===================================
    ##
    parser.add_argument("--num_generations", type=int, default=6, help="每个prompt生成的样本数")##===================================
    parser.add_argument("--beta", type=float, default=0.1, help="KL惩罚系数")##===================================
    parser.add_argument("--loss_type", type=str, default="cispo", choices=["grpo", "cispo"], help="loss类型")##===================================##===================================
    parser.add_argument("--epsilon", type=float, default=0.2, help="GRPO的PPO clip epsilon")##===================================
    parser.add_argument("--epsilon_high", type=float, default=5.0, help="epsilon上界")##===================================
    parser.add_argument('--from_weight', default='full_sft', type=str, help="基于哪个权重训练")##===================================##===================================
    # parser.add_argument("--reward_model_path", type=str, default="../../internlm2-1_8b-reward", help="Reward模型路径")##===================================##===================================
    parser.add_argument("--reward_model_path", type=str, default="../internlm2-1_8b-reward", help="Reward模型路径")# parser.add_argument("--reward_model_path", type=str, default="../../internlm2-1_8b-reward", help="Reward模型路径")##===================================##===================================
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")##===================================##===================================
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-GRPO", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    parser.add_argument("--debug_mode", action="store_true", help="是否打印训练调试采样")##===================================
    parser.add_argument("--debug_interval", type=int, default=20, help="debug模式下每隔多少step打印一次采样")##===================================
    parser.add_argument("--thinking_ratio", type=float, default=0.9, help="按概率开启thinking（0.0~1.0）")##===================================


    parser.add_argument("--rollout_engine", type=str, default="torch", choices=["torch", "sglang"], help="rollout引擎类型")##===================================##===================================
    parser.add_argument("--sglang_base_url", type=str, default="http://localhost:8998", help="SGLang服务器URL")
    parser.add_argument("--sglang_model_path", type=str, default="../model", help="SGLang tokenizer路径")
    parser.add_argument("--sglang_shared_path", type=str, default="./sglang_ckpt_grpo", help="SGLang共享存储路径")
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers,
                               max_seq_len=args.max_seq_len + args.max_gen_len, use_moe=bool(args.use_moe))##===================================  ##===================================
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume==1 else None##===================================##===================================
    
    # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    
    # ========== 4. 配wandb ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniMind-GRPO-Epoch-{args.epochs}-BS-{args.batch_size}-LR-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
    
    # ========== 5. 初始化模型和数据 ==========
    base_weight = args.from_weight
    # Policy模型
    model, tokenizer = init_model(lm_config, base_weight, device=args.device)
    ##
    ##
    # Reference模型
    ref_model, _ = init_model(lm_config, base_weight, device=args.device)
    ref_model = ref_model.eval().requires_grad_(False)
    ##
    ##
    # Reward模型=="../../internlm2-1_8b-reward", help="Reward模型路径"
    reward_model = LMForRewardModel(args.reward_model_path, device=args.device, dtype=torch.float16)


    # Rollout引擎（可插拔替换，只负责 policy 推理）
    rollout_engine = create_rollout_engine(
        engine_type=args.rollout_engine,#["torch", "sglang"]##===================================##===================================
        policy_model=model,
        tokenizer=tokenizer,
        device=args.device,
        autocast_ctx=autocast_ctx,
        sglang_base_url=args.sglang_base_url,
        sglang_model_path=args.sglang_model_path,
        sglang_shared_path=args.sglang_shared_path,
    )




    # 数据和优化器
    train_ds = RLAIFDataset(args.data_path, tokenizer, max_length=lm_config.max_seq_len, thinking_ratio=args.thinking_ratio)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None




    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    loader_for_count = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler)
    iters = len(loader_for_count)
    total_optimizer_steps = math.ceil(iters / args.accumulation_steps) * args.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_optimizer_steps, eta_min=args.learning_rate / 10)
    
    # ========== 6. 从ckp恢复状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scheduler.load_state_dict(ckp_data['scheduler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
    
    # ========== 7. 编译和分布式包装 ==========
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
        rollout_engine.update_policy(model)##===================================##===================================
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank])
    rollout_engine.update_policy(model)##===================================##===================================
    
    # ========== 8. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch); indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        if skip > 0: 
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            grpo_train_epoch(epoch, loader, len(loader) + skip, rollout_engine, ref_model, reward_model, start_step, wandb, use_sglang = (args.rollout_engine == "sglang"))
        else:
            grpo_train_epoch(epoch, loader, len(loader), rollout_engine, ref_model, reward_model, 0, wandb, use_sglang = (args.rollout_engine == "sglang"))
    
    # ========== 9. 清理分布进程 ==========
    if dist.is_initialized(): dist.destroy_process_group()

# Epoch:[1/1](1/9751), Reward: -2.1561, KL_ref: -0.0007, Adv Std: 1.0442, Adv Mean: 0.0000, Actor Loss: 0.0068, Avg Response Len: 460.58, Learning Rate: 0.00000030
# Epoch:[1/1](2/9751), Reward: 0.1188, KL_ref: -0.0063, Adv Std: 1.0443, Adv Mean: -0.0000, Actor Loss: -0.0690, Avg Response Len: 439.50, Learning Rate: 0.00000030

