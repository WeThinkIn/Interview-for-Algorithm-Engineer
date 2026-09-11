# DeepSeek高频面试题

> 用于持续沉淀 DeepSeek、推理模型、MoE、强化学习、代码/数学能力、训练与推理优化相关面试题。

## 待补充方向

本文件作为该公司面试题的持续更新入口。后续可按照“岗位/面试轮次/问题/答案/手撕代码/复盘要点”的格式补充内容。

建议优先补充以下方向：

- 大模型、AIGC、多模态、Agent 或推荐搜索相关岗位面经。
- 公司核心业务场景中的算法工程问题。
- 高频手撕代码、系统设计、训练/推理/部署工程题。

## 目录

- [DeepSeek 2026-09-08 AI Agent开发岗社招一面](#deepseek-agent-20260908)
  - [1. 你最近在关注 Agent 领域的什么新技术？——你说你关注了 DeepSeek-Harness，你怎么理解它和 LangChain 的本质区别？Harness 解决了什么 LangChain 解决不了的问题？](#deepseek-agent-20260908-q1)
  - [2. 选一个你做得最满意的 Agent 项目，完整讲一遍。——从业务背景到技术方案到落地效果。不要报菜名（说用了 RAG、用了 Multi-Agent），我要听设计决策背后的思考。](#deepseek-agent-20260908-q2)
  - [3. 你的 Agent 整体架构是分几层的？——每一层的职责是什么？层与层之间的接口是怎么设计的？为什么这么设计？如果换一种分层方式会有什么不同？](#deepseek-agent-20260908-q3)
  - [4. 你的 Agent 是用的 ReAct 还是 Plan-and-Execute？——为什么选这个？如果任务更复杂（比如 50 步的长链路），你的方案还 work 吗？怎么改进？](#deepseek-agent-20260908-q4)
  - [5. 多 Agent 协作是怎么实现的？——通信模式是什么？如果子 Agent 返回的结果是自然语言（而不是结构化数据），主 Agent 怎么解析？怎么保证解析不出错？](#deepseek-agent-20260908-q5)
  - [6. 你的 Agent 评测体系是怎么搭建的？——评测集覆盖了多少场景？怎么构造的？如果评测集里 30% 的 case 答案有主观性，你怎么处理？](#deepseek-agent-20260908-q6)
  - [7. Badcase 怎么回流到训练/优化流程里？——你是不是直接把失败轨迹拿出来重新做 SFT？如果是，有什么问题？如果不是，你的完整处理链路是什么？](#deepseek-agent-20260908-q7)
  - [8. 如果 Agent 效果不达预期，你怎么判断瓶颈在哪？——是模型能力不够？工具调用不准？上下文丢失？还是评测标准有问题？怎么区分？](#deepseek-agent-20260908-q8)
  - [9. Agent Loop 的核心设计是什么？——每一步的输入输出格式是什么？如果 LLM 返回的 Tool Call 不合法，怎么处理？如果执行完工具后观察结果太长，怎么截断？](#deepseek-agent-20260908-q9)
  - [10. 如果 Agent 跑了 20 分钟还没结束，用户等不及了要取消，你的系统怎么响应？——取消后，已经在执行的工具调用怎么处理？任务状态怎么回滚？](#deepseek-agent-20260908-q10)
  - [11. 上下文压缩时，你怎么做信息丢失的评估？——用模型打分还是规则？如果压缩丢了关键信息导致 Agent 后面决策错误，你能检测到吗？怎么恢复？](#deepseek-agent-20260908-q11)
  - [12. 怎么看待现在的 AI Coding 工具（Cursor、Claude Code、Codex）？——你觉得它们最大的瓶颈是什么？如果让你做一个更好的，你会怎么设计？](#deepseek-agent-20260908-q12)
  - [13. 你觉得 3 年后的 Agent 开发框架会和现在有什么不同？——Harness 会变成什么形态？模型和工程的分界线在哪里？](#deepseek-agent-20260908-q13)
  - [14. 如果你来做 DeepSeek-Harness 的下一个版本，你会加什么功能？](#deepseek-agent-20260908-q14)
  - [15. 你读过 DeepSeek-Harness 的源码吗？——你觉得它的核心设计亮点是什么？有什么你觉得可以改进的地方？](#deepseek-agent-20260908-q15)
  - [16. 除了 Harness，你还关注哪些 Agent 方向的开源项目？——你 Fork 过吗？提过 PR 吗？](#deepseek-agent-20260908-q16)
  - [17. 设计一个"多人协作的 Agent 平台"，支持多个用户同时使用，每个用户可以创建多个 Agent，Agent 之间可以协作。——多租户隔离怎么做？Agent 之间的通信怎么设计？权限控制怎么实现？](#deepseek-agent-20260908-q17)
  - [18. 实现一个简单的 FST（有限状态机）来管理 Agent 的状态。——状态包括：INIT、PLANNING、EXECUTING、CHECKING、COMPLETED、FAILED。状态转换的触发条件是什么？](#deepseek-agent-20260908-q18)
  - [19. 手写一个简单的 Agent Loop 框架（Python 伪代码），支持：工具注册、ReAct 循环、最大步数限制、错误处理。](#deepseek-agent-20260908-q19)
  - [20. 面试官问：你还有什么想问我的？——可以反问团队的技术路线、未来的方向、面临的挑战等深度问题。](#deepseek-agent-20260908-q20)

- [1. DeepSeek 大模型算法岗笔试面经（2026-07-26）](#deepseek-llm-algo-20260726)
  - [1. 题目一：手写完整的Multi-Head Attention，不能只写框架](#deepseek-llm-algo-20260726-q1)
  - [2. 题目二：DPO的完整训练流程推导，从数据准备到梯度更新](#deepseek-llm-algo-20260726-q2)
  - [3. 题目三：MOE模型的通信开销计算和负载不均衡问题分析](#deepseek-llm-algo-20260726-q3)
  - [4. 题目四：推理加速的底层实现（vLLM的PagedAttention原理、投机解码的工程实现）](#deepseek-llm-algo-20260726-q4)
  - [5. 题目五：DSpark推理加速框架的核心机制推导](#deepseek-llm-algo-20260726-q5)
  - [6. 题目六：DeepSeek-V4的KV缓存压缩原理与内存占用计算](#deepseek-llm-algo-20260726-q6)
  - [7. 题目七：多模态视觉原语推理框架的数学建模](#deepseek-llm-algo-20260726-q7)
  - [8. 题目八：DeepSeek MoE的无辅助损失负载均衡机制推导](#deepseek-llm-algo-20260726-q8)

<a id="deepseek-llm-algo-20260726"></a>
### 1. DeepSeek 大模型算法岗笔试面经（2026-07-26）

#### 面试问题汇总

<a id="deepseek-llm-algo-20260726-q1"></a>
##### 1. 题目一：手写完整的Multi-Head Attention，不能只写框架

**回答：**

这道题不能只写四个线性层，完整回答至少要把张量形状、缩放点积、掩码语义、数值稳定性和输出投影讲清楚。

设输入为 $X\in\mathbb{R}^{B\times S\times d_{model}}$，头数为 $H$，则每个头的维度为 $d_h=d_{model}/H$。先做三组线性投影：

$$
Q=XW_Q,\quad K=XW_K,\quad V=XW_V
$$

其中 $Q,K,V$ 的形状都是 $[B,S,d_{model}]$。将最后一维拆成多头并交换维度后，形状变为 $[B,H,S,d_h]$。每个头的注意力为：

$$
A=\operatorname{softmax}\left(\frac{QK^{\mathsf T}}{\sqrt{d_h}}+M\right),\qquad
O=AV
$$

最后把 $H$ 个头拼回 $[B,S,d_{model}]$，再经过 $W_O$ 得到输出。除以 $\sqrt{d_h}$ 是为了控制点积方差，避免 logits 随维度增大而使 Softmax 饱和、梯度变小。

一个可以现场运行的简化实现如下。这里约定布尔 `attn_mask=True` 表示允许关注，形状可以是 `[B, 1, Q, K]` 或可广播到该形状的张量：

```python
import math
import torch
from torch import nn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None, is_causal=False):
        bsz, seq_len, _ = x.shape
        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim)
        q = q.transpose(1, 2)  # [B, H, S, Dh]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        if is_causal:
            causal = torch.tril(
                torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device)
            )
            scores = scores.masked_fill(~causal, torch.finfo(scores.dtype).min)
        if attn_mask is not None:
            scores = scores.masked_fill(~attn_mask, torch.finfo(scores.dtype).min)

        weights = torch.softmax(scores.float(), dim=-1).to(scores.dtype)
        weights = self.dropout(weights)
        out = weights @ v
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)
        return self.out_proj(out)
```

需要主动说明几个边界。第一，`view` 只能在内存布局满足条件时安全使用，合并转置后的头通常要先 `contiguous()`；第二，因果 Mask 与 Padding Mask 的形状和广播规则不能混为一谈；第三，Cross-Attention 时 $Q$ 的长度可以与 $K,V$ 不同，不能把所有序列长度写死为 $S$。在现代大模型中还可能使用 GQA/MQA、RoPE、FlashAttention 或 fused kernel，但这些是对 QKV 组织和内核的优化，不改变上述注意力的数学骨架。

<a id="deepseek-llm-algo-20260726-q2"></a>
##### 2. 题目二：DPO的完整训练流程推导，从数据准备到梯度更新

**回答：**

DPO 的关键不是“把语言模型直接当成奖励模型”，而是从 KL 约束下的最优策略形式中消去显式奖励模型，得到只依赖偏好对数概率的目标。

训练数据是一组三元组 $(x,y_w,y_l)$：$x$ 是提示，$y_w$ 是偏好回答，$y_l$ 是非偏好回答。参考模型 $\pi_{ref}$ 通常是冻结的 SFT 模型，待训练策略是 $\pi_\theta$。

从 KL 正则化的 RLHF 目标出发：

$$
\max_\pi\;\mathbb{E}_{y\sim\pi(\cdot|x)}[r(x,y)]
-\beta D_{KL}\left(\pi(\cdot|x)\middle\|\pi_{ref}(\cdot|x)\right)
$$

其最优策略满足：

$$
\pi^*(y|x)=\frac{1}{Z(x)}\pi_{ref}(y|x)\exp\left(\frac{r(x,y)}{\beta}\right)
$$

于是隐式奖励可以写成：

$$
r(x,y)=\beta\log\frac{\pi^*(y|x)}{\pi_{ref}(y|x)}+\beta\log Z(x)
$$

偏好数据常用 Bradley-Terry 模型：

$$
P(y_w\succ y_l|x)=\sigma\left(r(x,y_w)-r(x,y_l)\right)
$$

两个回答的 $\beta\log Z(x)$ 会相互抵消，把 $\pi^*$ 用待训练策略近似，就得到 DPO 损失：

$$
\mathcal{L}_{DPO}(\theta)=-\log\sigma\left(\beta\left[
\log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)}
-\log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}
\right]\right)
$$

其中序列对数概率是回答 token 条件概率之和：

$$
\log\pi(y|x)=\sum_{t\in\text{completion}}\log\pi(y_t|x,y_{<t})
$$

实际训练流程如下：

1. 清洗偏好对，确认 chosen/rejected 的 prompt 一致，处理重复、长度异常和标签噪声。
2. 用策略模型和冻结参考模型分别对 chosen、rejected 做 teacher forcing，只对 completion 部分累加 log-prob，不能把 prompt 和 padding 算进去。
3. 按上式计算 log-ratio、margin 和损失；只对策略模型反向传播，参考模型不更新。
4. 使用 AdamW、学习率和 $\beta$ 等超参数更新策略模型；实践中常从 SFT 模型开始，并监控 KL 漂移、偏好准确率、长度偏差和通用能力。
5. 用独立偏好集、事实性集和安全集评测，不能只看训练损失。

令括号内的量为 $z$，梯度方向可写为 $-\beta\sigma(-z)\nabla z$：当策略模型还没有把 chosen 与 rejected 拉开时，梯度大；margin 已经很大时，样本梯度自然变小。DPO 不需要在线 rollout、单独 Reward Model 或 PPO，但它依然依赖高质量偏好数据和合理参考模型。长回答的 log-prob 还可能带来长度偏置，因此要关注长度归一化、数据配比以及 IPO、KTO、ORPO 等相邻方法的适用边界，不能把 DPO 当成对所有偏好学习问题的唯一答案。

<a id="deepseek-llm-algo-20260726-q3"></a>
##### 3. 题目三：MOE模型的通信开销计算和负载不均衡问题分析

**回答：**

MoE 的计算量与通信量要分开分析。设一个并行批次有 $T$ 个 token，隐藏维度为 $d$，每个 token 选择 $k$ 个专家，专家数为 $E$，专家并行组大小为 $P$，每个激活值使用 $b$ 字节。

路由器先为每个 token 计算专家分数并选出 Top-$k$。在 Expert Parallel 中，token 的激活需要通过 All-to-All Dispatch 发到拥有目标专家的设备；专家计算后再通过 All-to-All Combine 将结果送回原 token 所在设备。若负载均匀、每台设备本地命中比例约为 $1/P$，网络上的有效激活载荷可近似写为：

$$
V_{one-way}\approx Tkd b\left(1-\frac{1}{P}\right)
$$

两次交换的总载荷约为：

$$
V_{comm}\approx 2Tkd b\left(1-\frac{1}{P}\right)
$$

这是粗略的聚合估计；实际还要加上路由索引、padding、元数据、协议头以及拓扑带来的非均匀开销。用延迟带宽模型表达单次通信时间：

$$
T_{comm}\approx \alpha N_{collective}+\frac{V_{comm}}{BW_{eff}}
$$

其中 $\alpha$ 是集体通信启动延迟，$BW_{eff}$ 是受 PCIe、NVLink、RoCE、拥塞和消息大小共同影响的有效带宽。小 batch 时延迟项突出，大 batch 时带宽和负载倾斜更关键。

计算侧，每个专家接收的 token 数约为 $Tk/E$，若专家是标准两层 FFN，单 token 的矩阵乘开销大致与 $d\times d_{ff}$ 成正比；因此 MoE 通过只激活少数专家降低每 token 的计算量，但并不免费：路由、All-to-All、专家权重驻留和负载倾斜会吞掉收益。比如总参数很大而激活参数较小，只能说明稀疏计算特征，不能直接推出端到端吞吐。

负载不均衡要同时看三件事：专家接收 token 数的均值和最大值、负载变异系数 `std(load)/mean(load)`，以及因为 capacity 溢出被丢弃或重路由的 token 比例。常见 capacity 为：

$$
C=\left\lceil \frac{c\,Tk}{E}\right\rceil
$$

其中 $c$ 是 capacity factor。$c$ 太小会丢 token，太大则浪费显存和通信；实际需按序列长度、batch、Top-$k$ 和目标丢弃率调参。

造成倾斜的原因包括路由器偏置、热门 token、领域分布变化、专家容量不足和跨节点拓扑差异。可用辅助负载均衡损失、专家偏置控制、容量约束、随机路由、动态 batching 以及拓扑感知放置缓解。回答时还要区分“训练时路由均衡”和“推理时服务均衡”：后者除了 token 数，还要考虑专家所在设备的真实排队时间和通信路径。

<a id="deepseek-llm-algo-20260726-q4"></a>
##### 4. 题目四：推理加速的底层实现（vLLM的PagedAttention原理、投机解码的工程实现）

**回答：**

这道题应拆成两部分：PagedAttention 解决 KV Cache 的内存管理和批处理问题；投机解码通过较便宜的草稿预测减少目标模型的串行迭代次数。两者可以结合，但不是同一个算法。

### PagedAttention

标准自回归解码中，每条请求的 KV Cache 随序列长度增长。如果为每条请求预留一段连续显存，长短请求交错会产生内部碎片，动态扩容和请求结束后的回收也很困难。PagedAttention 把 KV Cache 切成固定大小的 block：

- 逻辑序列按 token 位置映射到逻辑 block；
- block table 将逻辑 block 映射到不连续的物理显存 block；
- 新 token 到来时只分配需要的 page，结束时按引用计数回收；
- 共享前缀可以让多个请求指向相同物理 block，写入时再做 Copy-on-Write。

Attention 内核按照 block table 分页读取 K、V，并完成分块归约。这样做的本质是把“连续大数组”变成操作系统式的分页地址空间，显著降低碎片、提高并发请求的显存利用率，并支持 continuous batching 和 prefix caching。它不改变注意力的理论复杂度，也不会凭空减少每个 token 的 KV 字节数；速度收益来自更高的可服务并发、较少的内存浪费和更好的调度。

### 投机解码

给定已经确认的前缀，草稿模型先自回归生成 $\gamma$ 个候选 token；目标模型随后一次前向并行验证这些位置。贪心解码可以逐位置接受与目标模型一致的前缀；随机采样若要严格保持目标分布，需要使用基于目标分布与草稿分布的 rejection sampling，并在拒绝位置进行修正采样。若所有候选都接受，还可以额外接收目标模型给出的下一个 token。

设每个候选位置在此前位置均被接受的条件下的生存概率为 $p_i$，则期望接受长度可近似写为：

$$
\mathbb{E}[L]\approx\sum_{j=1}^{\gamma}\prod_{i=1}^{j}p_i
$$

一次迭代的收益不是只看 $\mathbb{E}[L]$，还要除以草稿生成、目标验证、KV 读写和调度的总时间：

$$
\text{speedup}\approx
\frac{\text{baseline target steps time}}
{T_{draft}(\gamma)+T_{verify}(\gamma)+T_{schedule}}
\times (1+\mathbb{E}[L])
$$

这是直觉化表达，真正评测应直接测端到端 ITL、TPOT、吞吐和请求尾延迟。

工程实现的难点包括：草稿模型与目标模型 tokenizer/位置编码兼容；验证阶段正确复用和回滚 paged KV；接受长度变化时避免无效显存写入；连续 batching 下不同请求的候选长度不同；低接受率时及时退化为普通解码；以及在量化、张量并行和 prefix cache 下保持数值与调度正确。现代实现还可能使用 n-gram、EAGLE、Medusa 或半自回归草稿器，但核心判断始终是“草稿成本是否小于它节省的目标模型串行步数”。

<a id="deepseek-llm-algo-20260726-q5"></a>
##### 5. 题目五：DSpark推理加速框架的核心机制推导

**回答：**

DSpark 这类方法可以按“更强的草稿器 + 接受率感知的验证调度”来理解。题目中给出的版本名和基准数字应以对应正式论文、代码和硬件配置为准；面试回答的重点是把机制和可计算的收益说清楚。

传统自回归草稿器逐 token 生成，候选长度为 $\gamma$ 时草稿成本近似随 $\gamma$ 增长；完全并行的草稿器虽然一次预测多个位置，却缺少块内前缀依赖，后面位置的接受率容易下降。半自回归设计通常将两者折中：并行主干先产生各位置的候选表示，再由轻量顺序模块注入前缀依赖。顺序模块可以是只依赖前一位置的 Markov head，也可以通过循环状态累积更长前缀信息。

对每个候选位置，草稿器不仅输出 token，还输出一个接受置信度 $p_i$。它应表示在前面候选已经被目标模型接受的条件下，该位置继续被接受的概率，而不是未经校准的 softmax 最大值。可以用独立验证集做逐位置温度缩放或其他校准，使预测置信度与经验接受率一致。

若一次最多验证 $\gamma$ 个 token，接受长度的近似期望为：

$$
\mathbb{E}[L(\gamma)]\approx\sum_{j=1}^{\gamma}\prod_{i=1}^{j}p_i
$$

在并发服务中，验证长度还受目标模型 batch、KV Cache、SM 利用率和通信路径影响。因此调度器不是对每个请求独立选择一个固定 $\gamma$，而是根据当前 batch 的置信度序列和实测吞吐曲线，近似求解：

$$
\max_{\gamma_1,\ldots,\gamma_N}
\frac{\sum_{r=1}^{N}\left(1+\mathbb{E}[L_r(\gamma_r)]\right)}
{T_{verify}(\gamma_1,\ldots,\gamma_N)+T_{draft}+T_{overhead}}
$$

因此“置信度调度”真正优化的是单位时间确认 token 数，而不是盲目把候选块做长。结构化代码和数学文本往往具有更强的局部可预测性，接受率可能高于开放式对话，但这只是数据分布现象，不能当成所有任务的保证。

要讲清正确性边界：只要最终由目标模型验证，并在随机采样场景使用正确的拒绝采样修正，草稿器可以是近似的，目标分布仍可保持；置信度调度只影响效率，不应改变目标模型的输出分布。工程上还要做接受长度、草稿开销、ITL、吞吐、P99 延迟、显存和不同并发度的联合评测。只有在这些指标上稳定胜出，才算真正的推理加速，而不是单用户生成速度的局部提升。

<a id="deepseek-llm-algo-20260726-q6"></a>
##### 6. 题目六：DeepSeek-V4的KV缓存压缩原理与内存占用计算

**回答：**

KV Cache 的内存占用首先由层数、KV 头数、头维度、序列长度、批量和精度决定，与模型总参数量不是一回事。

标准 MHA 中，若 decoder 层数为 $L$，每层 KV 头数为 $H_{kv}$，每头维度为 $d_h$，上下文长度为 $T$，batch 为 $B$，每个元素占 $b$ 字节，则：

$$
M_{KV}=B\times T\times L\times H_{kv}\times d_h\times 2\times b
$$

最后的 $2$ 对应 K 和 V。GQA/MQA 通过降低 $H_{kv}$ 减少 Cache；量化通过降低 $b$ 减少字节数，但会引入误差。

以压缩 latent KV 为例，如果每个 token 每层只缓存一个维度为 $d_c$ 的共享 latent，并额外保留维度为 $d_r$ 的解耦位置编码 Key，则可近似写为：

$$
M_{compressed}\approx B\times T\times L\times(d_c+d_r)\times b
$$

解码时由 latent 经过投影恢复部分 K/V，计算换来的收益是显存和带宽下降，代价是额外矩阵乘、访问模式变化、量化/重建误差和实现复杂度。若使用分组量化，还要额外计入 scale、zero-point 等元数据；若采用分层或混合精度，不能用一个统一压缩比代替真实的逐层统计。

举例说，若某配置为 $L=80$、$H_{kv}=8$、$d_h=128$、$T=10^6$、FP16、batch 为 1，则标准 MHA 的 KV 字节数为：

$$
80\times10^6\times8\times128\times2\times2
=327.68\times10^9\text{ bytes}
\approx305.2\text{ GiB}
$$

这个例子只用于说明量级，不能套用到其他模型。要从一个宣称的 GB 数字反推压缩率，必须知道实际层数、KV 维度、缓存精度、是否分页、是否把临时 workspace 和权重算入显存，以及 GB 还是 GiB。仅凭“模型有多少参数”无法推出 KV Cache 大小，也不能据此证明某种压缩结构。

所以这道题的严谨答法是：先写标准公式，再写 MLA/GQA/量化等压缩如何改变公式，最后从显存、带宽、重建 FLOPs 和质量损失做联合预算。任何具体版本号、参数规模或单卡数字，都应以公开模型配置和可复现实验为准，不能把未经配置支撑的数字当作理论推导结论。

<a id="deepseek-llm-algo-20260726-q7"></a>
##### 7. 题目七：多模态视觉原语推理框架的数学建模

**回答：**

视觉原语的核心是把空间参照从模糊的自然语言描述提升为模型可以生成、比较和验证的几何对象。可以把文本 token 与视觉原语统一成一个混合序列建模问题。

给定图像 $I$、用户文本 $x$，定义视觉原语集合：

$$
\mathcal{P}=\{\text{point}(u,v),\text{box}(u_1,v_1,u_2,v_2),
\text{mask},\text{polygon},\ldots\},
$$

其中坐标通常归一化到 $[0,1]$，并约束 $u_1\le u_2$、$v_1\le v_2$。令输出空间为文本 token 空间 $\mathcal{V}$ 与原语空间 $\mathcal{P}$ 的并集，模型生成混合序列 $z=(z_1,\ldots,z_n)$：

$$
p(z|I,x)=\prod_{t=1}^{n}p(z_t|I,x,z_{<t})
$$

实际实现可以将原语离散化为特殊 token 加坐标 bin，例如 `[BOX, x1, y1, x2, y2]`；也可以让模型输出连续坐标分布或调用专门的 grounding head。一个通用的训练目标可以写成：

$$
\mathcal{L}=\mathcal{L}_{text}
+\lambda_{coord}\mathcal{L}_{coord}
+\lambda_{geom}\mathcal{L}_{geom}
+\lambda_{ground}\mathcal{L}_{ground}
$$

其中文本部分使用交叉熵，点/框可用 L1、Smooth L1、IoU/GIoU 损失，grounding 部分约束生成的原语确实指向图像中的目标。若是计数或空间关系任务，还需要任务损失与几何一致性约束。

视觉原语的价值在于让“左上角的红色物体”变成可计算的引用：模型可以生成一个框，再对框内目标分类；也可以比较两个框的中心坐标、IoU、包含关系和相对方向。这比完全依赖语言中的“那个”“附近”“左边”更容易验证。

面试中还要说清三个难点。第一，坐标误差会传播到后续引用，因此需要坐标量化校准、IoU 阈值和不确定性表示；第二，文本 token 与几何 token 的概率空间和损失尺度不同，$\lambda$ 需要通过验证集校准；第三，视觉原语必须有语法约束和合法性校验，否则可能生成越界、反向框或无法落到图像的坐标。

因此，视觉原语不是简单给多模态模型加几个坐标 token，而是把感知、指代、空间关系和生成动作放进一个可执行的中间表示。它是否带来领先结果，要看数据集、标注质量、坐标编码、grounding 评测和与纯文本 CoT 的公平对比，不能只凭“引入点和框”推出效果。

<a id="deepseek-llm-algo-20260726-q8"></a>
##### 8. 题目八：DeepSeek MoE的无辅助损失负载均衡机制推导

**回答：**

这类“无辅助损失”方案的要点不是取消均衡控制，而是把均衡控制从主损失中的可学习惩罚项，改成路由器外部的专家偏置反馈。

设 token $t$ 对专家 $i$ 的原始路由分数为 $s_{t,i}$，专家偏置为 $b_i$。路由选择使用：

$$
\tilde{s}_{t,i}=s_{t,i}+b_i,
\qquad
\mathcal{E}_t=\operatorname{TopK}_i(\tilde{s}_{t,i},k)
$$

关键实现细节是：偏置用于影响 Top-$k$ 的选择，但专家组合权重通常仍使用未加偏置的原始路由分数并做归一化。这样偏置负责纠正“谁被选中”，尽量不直接改变主任务中的混合权重。

在一个统计窗口内，设专家 $i$ 实际接收的 token 数为 $c_i$，平均负载为 $\bar c=\frac{1}{E}\sum_i c_i$。高负载专家应被降低偏置，低负载专家应被提高偏置。一种离散更新可写成：

$$
b_i\leftarrow b_i-\gamma\,\operatorname{sign}(c_i-\bar c)
$$

也可以用带裁剪的比例反馈：

$$
b_i\leftarrow\operatorname{clip}\left(
b_i-\gamma\frac{c_i-\bar c}{\bar c+\epsilon},
b_{min},b_{max}\right)
$$

其中 $\gamma$ 控制反馈速度。偏置一般按 batch 或固定 token 窗口更新，而不是对每个 token 更新，否则噪声会使路由剧烈抖动。实践还要考虑专家容量、跨节点通信、局部负载与全局负载不一致，以及偏置版本在分布式设备之间同步。

与辅助损失相比，外部偏置的优势是不会把一个需要调权重的均衡目标直接加入语言建模损失，降低主任务梯度被干扰的风险；它也能在训练早期通过反馈快速纠正热门专家。代价是它引入了一个非梯度控制回路：$\gamma$、更新窗口、偏置上下界和同步延迟都需要调节；统计窗口太短会抖动，太长又响应不及时；只看 token 数还可能掩盖某些专家计算更慢或通信更远的问题。

辅助损失的典型思想是同时惩罚选择频率和平均路由概率的不均衡，例如让各专家的 `load` 与 `importance` 接近均匀；无辅助损失并不等价于“天然均衡”，它仍然要通过实时负载监控和偏置更新实现闭环控制。最后要评估的不是偏置公式本身，而是负载变异系数、token 丢弃率、All-to-All 时间、训练损失、验证效果和下游质量的综合变化。

<a id="deepseek-agent-20260908"></a>
### DeepSeek 2026-09-08 AI Agent开发岗社招一面

#### 面试问题汇总

<a id="deepseek-agent-20260908-q1"></a>
##### 1. 你最近在关注 Agent 领域的什么新技术？——你说你关注了 DeepSeek-Harness，你怎么理解它和 LangChain 的本质区别？Harness 解决了什么 LangChain 解决不了的问题？

**回答：**

这道题的核心不是比较两个框架的 API 数量，而是区分“应用编排抽象”和“可运行、可观测、可恢复的 Agent 执行时”。如果这里所说的 DeepSeek-Harness 是一个面向 Agent 运行时的 Harness，那么我会先基于它公开的代码和实际使用范围回答，不把未经确认的内部实现当成结论。

LangChain 更像一组构建 LLM 应用的组件和编排抽象：Prompt、模型、Retriever、Tool、Parser、Runnable、Workflow 等可以组合成调用链。它降低了把模型、检索器和工具接在一起的门槛，但应用最终仍然需要自己定义任务状态、重试、超时、权限、幂等、持久化和观测。LangGraph 等状态图抽象已经进一步补足了循环和持久化能力，但这仍属于应用编排层。

Harness 的关注点更靠近运行时控制面：它围绕一次任务维护状态机和执行循环，负责把模型输出解析为下一步动作，校验工具调用，执行受控副作用，写入事件轨迹，处理中断、超时、重试、恢复和停止条件，并为评测或训练保留完整轨迹。可以把两者的边界概括为：

$$
\text{Application Components} \xrightarrow{\text{LangChain-like orchestration}}
\text{Runnable Graph}
\xrightarrow{\text{Harness runtime}}
\text{Auditable Execution}
$$

LangChain 解决“怎样快速组装一次 LLM 应用调用”；Harness 重点解决“这次多步任务怎样在不确定、可失败、有副作用的环境中可靠运行”。例如，模型要求转账时，Harness 需要在真正执行前做 schema 校验、用户授权、金额上限、幂等键和人工确认；工具超时后需要知道该工具是否可能已经成功，不能简单重试造成重复扣款。

因此，Harness 不是天然替代 LangChain。实际系统可以用 LangChain 或其他 SDK 提供模型、检索和工具适配，再由自研 Harness 统一状态、策略、观测与恢复。Harness 也不能解决模型本身的推理能力、事实性和工具选择准确率问题，它只能把失败边界显式化、把不可靠输出限制在可验证的执行协议内。评价一个 Harness 是否有价值，应看任务完成率、错误恢复率、重复副作用率、P95/P99 延迟、单任务成本和轨迹可复现性，而不是看抽象层名字。

<a id="deepseek-agent-20260908-q2"></a>
##### 2. 选一个你做得最满意的 Agent 项目，完整讲一遍。——从业务背景到技术方案到落地效果。不要报菜名（说用了 RAG、用了 Multi-Agent），我要听设计决策背后的思考。

**回答：**

一个合格的项目介绍应该沿着“业务目标 -> 任务不确定性 -> 系统决策 -> 验证结果”展开，而不是罗列技术名词。可以用下面的结构回答：

1. **背景与目标。** 说明服务对象、原流程、痛点和成功标准。例如，用户需要在企业知识、工单和业务系统之间完成一个多步骤问题，原来由人工检索、判断并填写结果，耗时长且过程不可追踪。
2. **为什么需要 Agent。** 如果路径固定、规则明确，就使用 Workflow；只有当任务需要根据中间观察动态选择工具、拆解子任务或处理异常时，才引入 Agent。Agent 的合理性来自决策路径不固定，而不是因为系统中调用了大模型。
3. **系统边界。** 接入层负责鉴权和会话；编排层负责状态机、预算和停止条件；模型层负责规划、结构化动作和自然语言生成；知识层负责检索证据；工具层负责受控访问业务 API；状态层保存检查点和事件轨迹；评测层记录每一步输入、输出和结果。
4. **关键设计决策。** 说明为什么采用混合检索、为什么把高风险工具放在人工确认后、为什么使用结构化状态而不是把全部历史对话塞给模型、为什么某些固定步骤用确定性代码而不是交给模型。
5. **效果和证据。** 报告任务成功率、工具调用正确率、引用支持率、P95 延迟、单任务 token 成本、人工接管率等指标，同时说明评测集规模、时间窗口、基线和统计口径。没有真实线上数字时，应明确说这是离线实验或 PoC 结果，不能临场编造。

Agent 项目的本质不是“模型回答得像不像人”，而是从输入到最终结果形成可验证的闭环：

$$
\text{Goal}\rightarrow\text{Plan}\rightarrow\text{Action}\rightarrow\text{Observation}
\rightarrow\text{Verification}\rightarrow\text{Result}
$$

落地效果必须能对应到业务指标，例如平均处理时长降低、人工转交率下降、关键字段准确率提高或审计覆盖率提升。若只有模型离线打分上升，却没有任务级收益，不能称为完整落地。

## 架构设计深度追问

<a id="deepseek-agent-20260908-q3"></a>
##### 3. 你的 Agent 整体架构是分几层的？——每一层的职责是什么？层与层之间的接口是怎么设计的？为什么这么设计？如果换一种分层方式会有什么不同？

**回答：**

我会采用面向职责的六层结构，而不是按具体框架拆层：

1. **接入与会话层：** 负责身份认证、租户识别、流式响应、会话创建和请求取消，不决定业务动作。
2. **策略与编排层：** 维护任务状态机、预算、最大步数、超时、重试和人工接管规则，决定当前是否允许继续执行。
3. **模型决策层：** 负责意图识别、计划生成、工具选择、参数生成和最终语言表达。模型只能提出候选动作，不能绕过服务端策略直接产生副作用。
4. **知识与上下文层：** 管理短期消息、摘要、长期记忆、RAG 证据、权限过滤和 token 预算，向模型提供当前决策所需的最小充分上下文。
5. **工具与执行层：** 统一封装数据库、搜索、代码执行、企业 API 等工具，负责 schema 校验、鉴权、幂等、超时、重试、结果规范化和沙箱隔离。
6. **状态、观测与评测层：** 持久化事件、检查点、工具结果和指标，支持 trace、回放、离线评测、告警和 badcase 回流。

层间接口应该是版本化、结构化和可审计的。例如模型层返回 `ActionProposal`，执行层接收后重新校验；工具层返回包含 `status`、`data`、`error`、`retryable`、`request_id` 的 `ToolResult`；状态层以事件追加和检查点保存为主，而不是只存一段最终文本。典型动作协议可以表示为：

```json
{
  "type": "tool_call",
  "tool_name": "search_orders",
  "arguments": {"user_id": "...", "limit": 20},
  "request_id": "task-42-step-3",
  "expected_output": "order_list"
}
```

这样设计的原因是把“模型不确定性”和“系统确定性”隔开：模型可以规划，但权限、参数范围、预算和副作用由代码执行。若改成纯函数链，路径更简单、延迟更低、可测试性更好，但不适合动态任务；若改成全由一个 Agent 自由调用，开发初期灵活，生产环境却会增加越权、循环、状态丢失和故障定位成本；若所有状态都放在数据库，恢复容易但延迟和序列化成本增加，因此通常采用事件日志加关键检查点的折中。

<a id="deepseek-agent-20260908-q4"></a>
##### 4. 你的 Agent 是用的 ReAct 还是 Plan-and-Execute？——为什么选这个？如果任务更复杂（比如 50 步的长链路），你的方案还 work 吗？怎么改进？

**回答：**

ReAct 是“思考/决策 -> 行动 -> 观察 -> 下一步决策”的交错循环。它每一步都利用最新观察，适合搜索、问答和工具结果会改变后续路径的任务；缺点是模型调用次数多，容易局部贪心、重复调用或在相似状态中循环。

Plan-and-Execute 先生成一个较完整的计划，再由执行器逐步完成，并在必要时重新规划。它减少了每一步规划开销，适合子任务边界清晰、计划相对稳定的任务；缺点是初始计划可能基于错误假设，若执行器不反馈给规划器，错误会被连续放大。

真实系统通常采用混合方案：先生成粗粒度计划，由确定性执行器控制预算和依赖；每个高不确定性节点内部使用 ReAct；当观察与计划偏差超过阈值时触发局部重规划，而不是每一步从头规划。选择标准可以写成：

$$
\text{Policy}=
\begin{cases}
\text{ReAct}, & \text{环境反馈强、动作短且路径不确定};\\
\text{Plan-and-Execute}, & \text{子任务边界清晰、依赖稳定};\\
\text{Hybrid}, & \text{长链路且局部不确定性高}.
\end{cases}
$$

50 步长链路不能简单把最大步数改成 50。需要增加：任务分段与层级规划、每段的完成条件、阶段性 checkpoint、依赖图、错误预算、幂等执行、局部重试、全局超时和人工接管。上下文也不能保留全部原始 Observation，而应将每一步压缩成结构化状态，只保留未完成目标、已确认事实、工具结果引用、失败原因和下一步约束。

长链路的评测不能只看最终成功率，还要看每步错误率如何累积、失败恢复率、平均有效步数、重复副作用率和成本。若单步成功概率为 $p$，50 步完全正确的概率近似为 $p^{50}$，这说明系统必须通过验证、重试和分段恢复降低误差传播，而不能只依赖更大的上下文窗口。

<a id="deepseek-agent-20260908-q5"></a>
##### 5. 多 Agent 协作是怎么实现的？——通信模式是什么？如果子 Agent 返回的结果是自然语言（而不是结构化数据），主 Agent 怎么解析？怎么保证解析不出错？

**回答：**

多 Agent 协作首先要定义角色和边界，而不是简单复制多个模型。常见通信模式包括：

- **编排器-工作者：** 主 Agent 拆分任务，子 Agent 返回结果，适合有明确依赖的任务。
- **黑板模式：** 子 Agent 把中间结果写入共享任务空间，其他 Agent 订阅或读取，适合并行研究。
- **流水线模式：** 一个 Agent 的结构化输出作为下一个 Agent 的输入，适合固定阶段。
- **评审模式：** 生成 Agent 与验证 Agent 分离，适合高风险结论，但要防止评审也只是重复生成。

我更倾向于让通信协议以结构化消息为主，至少包含 `task_id`、`agent_id`、`schema_version`、`status`、`result`、`evidence`、`confidence`、`errors` 和 `side_effects`。主 Agent 不应直接相信自然语言结果，而应做以下降级处理：

1. 优先要求子 Agent 按 JSON Schema 输出，并在接收端使用严格解析器校验类型、必填字段、枚举和长度；
2. 对 Markdown、代码块或带解释的 JSON 做受限提取，但把它视为兼容层，不把正则表达式当成可靠协议；
3. 解析失败时返回可定位的校验错误，允许子 Agent 只修复格式，不重新生成整个事实结论；
4. 对关键字段做确定性校验，例如 ID 是否存在、数值是否越界、证据引用是否可回查；
5. 对有副作用的结果采用两阶段提交或人工确认，解析成功不等于允许执行。

自然语言不可避免时，可以通过“语义解析器 -> schema 校验 -> 事实验证 -> 置信度门控”转成内部对象；无法验证的字段应标记为 `unknown`，不能默认为空字符串或模型猜测。这样保证的是协议和风险边界，而不是保证模型永远不出错。多 Agent 还要处理消息重复、乱序、超时、版本冲突和权限隔离，最好用任务 ID、消息 ID、状态版本和幂等键贯穿全链路。

## 评测与 Badcase 回流

<a id="deepseek-agent-20260908-q6"></a>
##### 6. 你的 Agent 评测体系是怎么搭建的？——评测集覆盖了多少场景？怎么构造的？如果评测集里 30% 的 case 答案有主观性，你怎么处理？

**回答：**

评测体系应分为数据集、执行器、指标和诊断四部分。数据集不能只收集“问答题”，而要覆盖真实任务的入口、工具、权限、长短上下文、失败依赖和风险等级。可以按场景分层：正常路径、歧义请求、知识缺失、工具错误、权限拒绝、长链路、并发、恶意提示和人工接管。

每个 case 至少包含用户目标、初始状态、允许工具、权限范围、参考事实或约束、期望终态、可接受答案集合和风险标签。数据来源包括人工编写的 golden case、线上脱敏样本、历史 badcase、程序生成的边界样本和对抗样本。训练、调参和测试集要按用户/任务/文档去重，避免同一模板泄漏造成虚高成绩。

指标要分层：

- **任务层：** 任务完成率、终态正确率、关键字段准确率、人工接管率。
- **轨迹层：** 工具选择准确率、参数正确率、无效调用率、重试恢复率、停止正确率。
- **证据层：** 检索 Recall@K、引用覆盖率、事实支持率、权限过滤错误率。
- **系统层：** P50/P95/P99 延迟、token 成本、失败率、并发吞吐和副作用次数。

主观性 case 不能强行用一个 exact match。先定义可接受答案的评分维度，例如事实正确、任务完成、覆盖要点、表达清晰、安全合规；对开放式结论建立 rubric，并用两名以上标注者独立评分，报告一致性，如 Cohen's $\kappa$ 或 Krippendorff's $\alpha$。模型评审可以做规模化筛选，但不能单独作为高风险最终标准，要用人工抽检校准评审器偏差。

还要区分“答案多样但都正确”和“没有标准答案”。前者采用集合式参考答案或事实约束，后者使用 pairwise preference、分维度评分和任务结果评估。最终报告平均分之外的置信区间、分桶结果和人工复核样本，避免 70% 客观 case 的高分掩盖 30% 主观 case 的系统性偏差。

<a id="deepseek-agent-20260908-q7"></a>
##### 7. Badcase 怎么回流到训练/优化流程里？——你是不是直接把失败轨迹拿出来重新做 SFT？如果是，有什么问题？如果不是，你的完整处理链路是什么？

**回答：**

失败轨迹不是天然的 SFT 正样本。首先要保留完整上下文：用户输入、版本化 Prompt、模型输出、工具 schema、工具参数、Observation、环境状态、错误码、最终结果、人工修改和评测信息，并对隐私字段脱敏。然后将失败归因到具体层：意图识别、检索、规划、参数生成、执行器、上下文压缩、模型知识、评测标准或产品交互。

一个完整的回流链路是：

1. **采集与去重：** 统一 trace ID，过滤重复重试和无效日志，保护敏感数据。
2. **分类与归因：** 由规则、工具错误码和人工标注联合确定 failure type，并记录置信度。
3. **构造修复信号：** 客观错误可生成正确工具调用、拒答或格式样本；规划错误可由专家重写计划；检索错误可补充相关文档、hard negative 和查询改写样本。
4. **选择优化手段：** Prompt/Schema/工具描述修复适合协议问题；检索和索引修复适合证据问题；规则或 Workflow 修复适合确定性约束；SFT 适合稳定、可复现的行为模式；DPO/偏好优化适合多个候选中有明确偏好；奖励或 Agentic RL 适合有可验证环境反馈的长轨迹策略。
5. **离线验证和回归：** 在原 badcase、相邻场景、对抗样本和通用能力集上做单变量消融，确认修复没有引入过拟合、长度偏置或安全回退。
6. **灰度与监控：** 通过版本化配置、小流量灰度和回滚开关观察线上成功率、成本、延迟和副作用。

直接把失败轨迹做 SFT 有几个问题：失败动作会被模型模仿；一条长轨迹包含大量错误前缀，训练信号定位不清；线上用户输入可能含隐私；重复样本会让模型过拟合；只学习“怎么答”而不学习“何时拒绝或调用工具”。更可靠的做法是把失败轨迹当诊断材料，从中提取经过验证的修复片段，必要时保留失败轨迹作为负样本、偏好对或 verifier 的测试样本。

<a id="deepseek-agent-20260908-q8"></a>
##### 8. 如果 Agent 效果不达预期，你怎么判断瓶颈在哪？——是模型能力不够？工具调用不准？上下文丢失？还是评测标准有问题？怎么区分？

**回答：**

先固定模型、Prompt、工具版本、检索索引和评测集，建立一条可回放的基线，然后将端到端失败拆成阶段性指标。不能看到最终回答错误就直接归因于模型。

可以采用“分层替换 + 轨迹定位”的诊断矩阵：

| 检查项 | 替换或对照实验 | 主要判断 |
| --- | --- | --- |
| 评测标准 | 人工复核、双标注、对照参考答案 | 标准是否不可判定或偏置 |
| 模型能力 | 用强模型/人工 oracle 固定工具和证据 | 强模型也失败则不一定是当前模型 |
| 工具调用 | 给定正确工具和参数，跳过模型选参 | 仍失败则检查工具或执行器 |
| 上下文 | 注入人工整理的最小充分上下文 | 恢复成功说明检索/压缩/组装有问题 |
| 检索 | 给定 gold evidence，比较召回与生成 | gold 成功而真实检索失败，瓶颈在知识链路 |
| 规划 | 给定人工计划，只评估执行 | 区分计划错误与执行错误 |

同时检查轨迹事件：是否选择了错误工具、参数是否通过 schema、检索是否命中支持证据、压缩前后关键事实是否保留、Observation 是否被下一步消费、是否提前停止或陷入循环。用受控 ablation 比较：

$$
\Delta_i = M(\text{system with component }i\text{ replaced by oracle})
-M(\text{baseline})
$$

其中 $M$ 可以是任务成功率或事实支持率。$\Delta_i$ 越大，说明该组件更可能是当前瓶颈，但还需要统计显著性和分桶验证。

最后要关注评测与线上目标是否一致。若离线评分下降但用户任务完成率提升，可能是评测标准错配；若离线提升而线上没有变化，可能是数据分布、延迟、权限或交互流程问题。模型能力不足只能在工具、证据、上下文和评测均被控制后再下结论。

## Loop Engineering

<a id="deepseek-agent-20260908-q9"></a>
##### 9. Agent Loop 的核心设计是什么？——每一步的输入输出格式是什么？如果 LLM 返回的 Tool Call 不合法，怎么处理？如果执行完工具后观察结果太长，怎么截断？

**回答：**

Agent Loop 的核心是一个受预算约束的状态转换器，而不是 `while True` 加一次模型调用。每一步至少有以下状态：任务目标、当前计划、历史事件、可用工具、权限与预算、模型提议、工具结果、验证结果和停止原因。

输入给模型的内容应尽量结构化，包括系统策略、用户目标、当前状态摘要、相关证据、允许工具 schema、最近 Observation 和明确的终止条件。模型输出则限定为 `final`、`tool_call`、`ask_user`、`replan` 或 `fail` 等有限动作类型，并要求参数符合 JSON Schema。服务端不能把模型输出直接拼成 shell 命令或 URL。

一个抽象的循环是：

```python
def run_agent_loop(model, task_id, max_steps):
    for step in range(max_steps):
        state = load_checkpoint(task_id)
        if cancelled(state) or budget_exhausted(state):
            return fail("cancelled_or_budget_exhausted")

        proposal = model.decide(build_context(state))
        action = parse_and_validate(proposal, allowed_tools(state))
        if action.is_invalid:
            record_error(state, action.error)
            if action.retryable and state.format_retries < FORMAT_RETRY_LIMIT:
                state.format_retries += 1
                save_checkpoint(state)
                continue
            return fail("invalid_action")

        if action.type == "final":
            return verify_final(action, state)
        if action.type == "ask_user":
            return pause_for_user(action)

        result = execute_with_policy(action, state)
        observation = normalize_and_bound(result)
        state = apply_observation(state, observation)
        save_checkpoint(state)

    return fail("max_steps_exceeded")
```

非法 Tool Call 要区分格式错误、schema 错误、权限错误、业务参数错误和工具不存在。格式错误可以进行一次受限修复；权限和业务错误应直接拒绝并告知模型或用户；工具不存在不能通过模糊匹配随便替换；重复副作用必须依靠幂等键和执行记录判断。每个重试都应消耗预算并记录原因。

Observation 截断不能只取前 N 个字符。应先结构化解析，保留状态码、错误、关键字段、摘要、分页游标、证据 ID 和结果统计；大列表按相关性或确定性规则筛选，并把完整结果放在受控存储中，给模型返回引用。对日志、代码和文档可以采用头尾保留、分块摘要、查询相关片段和按 token 预算压缩。截断后要保留 `truncated=true`，否则模型会把不完整结果误认为完整结果。对于最终决策所需的金额、ID、权限和错误字段，应设置不可丢弃字段，超预算时暂停或请求用户，而不是静默删除。

<a id="deepseek-agent-20260908-q10"></a>
##### 10. 如果 Agent 跑了 20 分钟还没结束，用户等不及了要取消，你的系统怎么响应？——取消后，已经在执行的工具调用怎么处理？任务状态怎么回滚？

**回答：**

取消是跨层传播的控制信号，不是前端断开连接就结束。请求层生成 `cancel_token`，通过任务队列、Agent Loop、模型流式请求和工具执行器逐层传递；每个耗时阶段在安全点检查，并将任务状态从 `RUNNING` 原子地改为 `CANCELLING`，防止新的步骤继续提交。

工具调用分三类处理：

1. **可安全中断的纯读操作：** 取消 HTTP 请求、释放连接和临时资源，标记为 cancelled。
2. **支持取消协议的可中断写操作：** 向工具发送 cancel，等待确认或超时后进入补偿流程。
3. **不可撤销的副作用：** 不能假设取消等于回滚。必须使用幂等键、操作状态查询、事务或补偿动作；例如支付提交后取消，只能查询最终状态并做业务允许的退款/冲正，不能重新执行。

任务状态最好使用显式状态机：

$$
RUNNING\rightarrow CANCELLING\rightarrow
\begin{cases}
CANCELLED, & \text{所有可控子任务已停止};\\
UNKNOWN\_IN\_FLIGHT, & \text{副作用状态无法确认};\\
COMPLETED, & \text{取消到达前已完成}.
\end{cases}
$$

“回滚”也要分层：内存中的规划和未提交状态可以恢复到最近 checkpoint；数据库事务可以回滚未提交写入；已经提交的外部副作用只能通过查询和补偿，不能把事件日志删掉假装没发生。用户界面应收到明确状态和任务 ID，允许稍后查询；后台还要设置 orphan task 扫描、资源租约、最大运行时和告警，防止客户端断开后任务泄漏。

<a id="deepseek-agent-20260908-q11"></a>
##### 11. 上下文压缩时，你怎么做信息丢失的评估？——用模型打分还是规则？如果压缩丢了关键信息导致 Agent 后面决策错误，你能检测到吗？怎么恢复？

**回答：**

上下文压缩的目标不是让文本更短，而是保留后续决策所需的最小充分状态。压缩前先把信息分为硬约束、已确认事实、未完成目标、工具结果、证据引用、用户偏好、过程噪声和可重建内容。硬约束、权限、金额、时间、ID、错误状态和未完成任务不能仅依靠自由摘要保留，应写入结构化状态。

评估需要规则与模型结合：

- **规则校验：** 比较实体、数字、日期、ID、否定词、权限和状态枚举；检查所有未完成任务、工具结果引用和约束是否存在。
- **任务重放：** 把压缩结果输入后续决策器，与未压缩上下文的工具选择、参数和终态比较。
- **检索式覆盖：** 对关键事实生成查询，检查压缩上下文是否能找回对应证据。
- **模型评审：** 让独立评审器按事实保真、任务可继续性和冗余度打分，但必须用人工标注校准，并防止评审器与生成模型共享同一错误。

可以定义关键事实保留率：

$$
\mathrm{Retention}=
\frac{\sum_i w_i\,\mathbf{1}[f_i\text{ 在压缩后可正确恢复}]}
{\sum_i w_i}
$$

其中高风险约束的权重 $w_i$ 应显著高于普通闲聊。若压缩后工具选择与 gold context 不一致，或规则检测发现关键字段丢失，就在继续执行前阻断。

发生错误时要能恢复：保留原始事件日志和分段消息，压缩结果只是派生缓存；从最近完整 checkpoint 重新压缩，或按当前失败点检索相关原文；对高风险任务直接扩大上下文并降低自动执行权限；如果已产生副作用，先查询外部状态再决定补偿。真正可靠的设计是“可丢弃的摘要 + 不可丢失的结构化状态 + 可重放的原始事件”，而不是把一次模型摘要当成唯一事实来源。

## 开放性讨论

<a id="deepseek-agent-20260908-q12"></a>
##### 12. 怎么看待现在的 AI Coding 工具（Cursor、Claude Code、Codex）？——你觉得它们最大的瓶颈是什么？如果让你做一个更好的，你会怎么设计？

**回答：**

这类工具已经从单轮代码补全发展为围绕代码库进行检索、编辑、执行、测试和修复的 Agent。它们的核心差异不只在模型，还在上下文构造、文件操作协议、终端沙箱、补丁应用、测试反馈和人机确认流程。

当前最大瓶颈我会概括为“可靠地把局部代码修改映射到全局工程目标”：

1. 代码库上下文存在噪声，模型可能找到相似但不相关的文件；
2. 长任务中计划、修改和测试反馈容易漂移；
3. 测试不充分时，模型会把“命令执行成功”误认为“功能正确”；
4. 自动编辑和终端操作具有副作用，权限、密钥和数据泄露风险高；
5. 评测常用短题或单测，不能充分衡量真实仓库中的回归、维护性和长期成本。

如果设计一个更好的系统，我会把重点放在 Harness 和评测闭环，而不是只换一个更大的模型：

- 建立代码库地图，包括模块依赖、构建入口、测试目录、配置和 ownership；
- 使用符号级、语义级和文本级混合检索，优先返回定义、调用方、测试和相关配置；
- 将任务拆成可验证的计划，每次编辑采用小补丁、格式化、静态检查和增量测试；
- 所有文件写入和命令执行进入沙箱，区分只读、可写和高风险权限；
- 保存事件轨迹和 checkpoint，支持失败恢复、差异审查和人工批准；
- 用真实 issue、隐藏测试、回归测试、代码审查质量、修改范围和成本建立评测集。

最终交互应让用户看到“改了什么、依据是什么、测试证明了什么、还存在什么不确定性”，而不是只返回一段看似完整的代码。Agent 的自主性应随可验证性提升，不能把不可观察的自由操作误当成智能。

<a id="deepseek-agent-20260908-q13"></a>
##### 13. 你觉得 3 年后的 Agent 开发框架会和现在有什么不同？——Harness 会变成什么形态？模型和工程的分界线在哪里？

**回答：**

三年后的框架大概率不会只是更多 Chain、Tool 和 Prompt 模板，而会向“模型原生能力 + 可组合运行时 + 领域策略”发展。具体形态可能包括：

1. **协议更标准化：** 工具、资源、任务、事件、取消、权限和结果 schema 有稳定的跨框架接口，应用不必绑定某个编排库。
2. **执行时更像操作系统：** 统一管理上下文、检查点、调度、沙箱、资源配额、凭证、日志和恢复，模型是其中一个可替换的决策组件。
3. **编排更混合：** 对固定部分使用可验证 Workflow，对不确定部分使用 Agent；计划图和运行时事件可以互相修正。
4. **评测成为一等公民：** 框架原生支持轨迹回放、仿真环境、verifier、灰度、成本预算和线上指标，而不是只提供 tracing 看板。
5. **上下文从文本升级为状态：** 结构化事实、权限、工具结果、代码图、长期记忆和证据引用共同组成模型的工作状态。

Harness 会从“包住模型调用的循环”变成任务运行时：负责资源、权限、可恢复性和证据，而模型负责在给定状态下提出预测或动作建议。分界线不是“模型写的代码还是工程师写的代码”，而是该部分能否用确定性规则、事务和测试验证。模型适合处理开放语义、候选生成、模糊分类和动态规划；工程必须掌握身份、权限、金额、状态提交、幂等、超时、审计和安全边界。

这不是把所有问题都工程化。若一个任务需要真正开放的探索，过度固定会损失能力；但任何不可逆的副作用都应由工程策略和外部验证约束。未来竞争点会从“谁能让模型调用更多工具”转向“谁能让模型在更大任务空间里以可控成本稳定完成，并且失败后可解释、可恢复”。

<a id="deepseek-agent-20260908-q14"></a>
##### 14. 如果你来做 DeepSeek-Harness 的下一个版本，你会加什么功能？

**回答：**

在没有把某个具体实现细节当成已确认事实的前提下，我会优先补强以下能力：

1. **可恢复任务状态机。** 原生支持事件日志、版本化 checkpoint、暂停/恢复、取消和重放；每次模型决策、工具调用和 Observation 都能关联 `task_id` 与 `step_id`。
2. **策略门控。** 在工具执行前统一检查 schema、权限、预算、数据分级、幂等键和人工确认策略，并允许按工具风险配置不同门槛。
3. **可靠的上下文管理。** 将消息、摘要、证据、记忆和工具结果分层管理，提供 token 预算、关键事实保护、压缩评估和原文回溯，而不是只做字符串截断。
4. **仿真与回放评测。** 支持固定环境、mock 工具、故障注入、隐藏测试、轨迹回放和单步对比，使模型、Prompt、工具描述和策略可以做可重复消融。
5. **失败分类与自动回流。** 从 trace 中区分模型、检索、工具、权限、执行和评测失败，生成待人工确认的修复候选，不直接把失败轨迹当训练正样本。
6. **多租户与密钥隔离。** 让工具凭证、上下文、缓存、事件和长期记忆在租户边界内隔离，默认禁止把敏感信息写入模型 Prompt 和普通日志。
7. **成本与可靠性调度。** 根据任务风险和剩余预算选择模型、并行度、上下文规模和验证强度，对长任务设置阶段性预算和自动降级。

每个功能都要有验收指标。例如恢复能力看 checkpoint 恢复成功率和重复副作用率；上下文能力看关键事实保留率和任务成功率；评测能力看线上 badcase 到回归用例的闭环时间。Harness 的下一个版本不应只增加“更强的自主性”，而应增加“自主性可被验证和撤销”的能力。

## 技术视野

<a id="deepseek-agent-20260908-q15"></a>
##### 15. 你读过 DeepSeek-Harness 的源码吗？——你觉得它的核心设计亮点是什么？有什么你觉得可以改进的地方？

**回答：**

这道题必须诚实区分“完整读过源码”“读过公开部分”“了解其设计思想但没有逐文件阅读”。如果没有实际阅读，就不应声称知道内部模块、性能数字或线上架构。基于公开代码能够确认的内容，应具体指向模块、调用路径和可复现实验；无法确认的部分直接标记为推测。

阅读一个 Agent Harness，我会沿着一条任务从入口到终态的路径检查：任务对象怎样创建；上下文怎样组装；模型输出怎样解析；工具怎样注册和调用；状态怎样持久化；错误怎样传播；取消和超时怎样处理；事件怎样记录；测试怎样覆盖循环和恢复。比起文件数量，我更关注三个不变量：

1. 任意工具副作用是否都经过权限、参数和幂等检查；
2. 任意一步失败后是否能恢复或明确进入失败终态；
3. 任意最终答案是否能追溯到模型决策、工具结果和证据。

如果公开实现确实把模型调用、工具执行、状态管理和事件观测放在清晰的运行时边界内，这会是核心亮点：模型不再直接驱动不可控代码，而是通过协议向 Harness 提议动作。另一个亮点可能是把 Loop 设计成可插拔组件，使不同任务可以替换规划、验证和停止策略；但具体是否如此必须以代码为证。

改进方向通常包括：严格的类型化状态和 schema 版本；工具调用的取消、幂等和副作用分类；持久化 checkpoint 与崩溃恢复；长 Observation 的结构化裁剪；多租户凭证隔离；故障注入和隐藏评测；以及对模型、Prompt、工具版本的可复现锁定。源码阅读的最终结论应落到调用链和测试证据上，而不是泛泛评价“架构先进”。

<a id="deepseek-agent-20260908-q16"></a>
##### 16. 除了 Harness，你还关注哪些 Agent 方向的开源项目？——你 Fork 过吗？提过 PR 吗？

**回答：**

回答这道题要按真实经历展开，不能为了显得熟悉而虚构 Fork 或 PR。可以按照“项目类别 -> 关注问题 -> 实际动作 -> 学到什么”组织：

- **图编排与状态执行：** 关注循环、条件分支、checkpoint、人工介入和恢复如何表达。
- **代码 Agent 与终端 Harness：** 关注仓库检索、补丁应用、沙箱、测试反馈和命令权限。
- **协议与工具互操作：** 关注工具发现、schema、能力协商、连接生命周期和权限边界。
- **评测与轨迹平台：** 关注任务级成功率、仿真环境、verifier、回放和 badcase 管线。
- **记忆与检索系统：** 关注写入门控、冲突消解、过期、引用、权限和上下文预算。

如果 Fork 过，应该讲清 Fork 的目的、改动文件、测试方式和结果；如果提过 PR，说明问题复现、设计讨论、兼容性和维护者反馈；如果只是阅读和本地实验，就直接说“做过本地实验，没有提交 PR”，并展示实验脚本、对比指标或 issue 分析。面试官真正想判断的是是否能从开源实现中提出可验证的技术判断，而不是 GitHub 账号上有多少动作。

一个成熟的学习方法是：先锁定版本，画出入口到工具执行的调用图；再用最小任务运行，记录事件和状态变化；随后针对超时、非法调用、重复消息和恢复写测试；最后提交文档、测试或小范围 bug fix。这样比只浏览 README 更能理解 Agent 框架的工程边界。

## 系统设计

<a id="deepseek-agent-20260908-q17"></a>
##### 17. 设计一个"多人协作的 Agent 平台"，支持多个用户同时使用，每个用户可以创建多个 Agent，Agent 之间可以协作。——多租户隔离怎么做？Agent 之间的通信怎么设计？权限控制怎么实现？

**回答：**

我会先定义平台边界：用户创建 Agent 配置和工具权限，提交任务后由编排器调度多个 Agent；Agent 可以并行执行子任务，但共享状态和外部副作用必须受控。整体采用“控制面中心化、执行面可水平扩展”的架构。

核心资源层级为 `tenant -> user -> workspace -> agent -> task -> run -> step`。每条资源都带 `tenant_id`，服务端从认证身份推导租户，禁止信任客户端传入的租户字段。数据库使用租户条件、行级策略或独立 schema/实例隔离；缓存、对象存储、向量索引、队列和日志同样使用租户命名空间。高敏租户可以采用物理隔离，普通租户使用逻辑隔离并通过自动化测试验证越权不可见。

Agent 通信不建议直接互相发送任意自然语言，而采用带 schema 的消息总线或任务协议：

```text
Message {
  message_id, tenant_id, task_id, parent_step_id,
  sender_agent, receiver_agent, schema_version,
  type, payload, evidence_refs, deadline,
  idempotency_key, state_version
}
```

编排器维护任务 DAG 或状态图，能并行的子任务放入队列；子 Agent 返回结构化结果、证据、置信度和副作用记录；Reducer 按任务版本合并，冲突进入仲裁 Agent 或人工队列。消息至少一次投递时，消费端用 `message_id` 和幂等键去重；需要严格顺序的共享资源采用单写者、版本号或事务，而不是让多个 Agent 直接覆盖同一状态。

权限分为用户权限、Agent 权限、工具权限和数据权限。使用 RBAC 解决角色授权，用 ABAC 表达租户、资源归属、环境、数据等级和时间等条件；每一次工具调用都在服务端重新鉴权，不能因为模型选择了工具就放行。工具凭证采用短期、最小权限 token，Agent 只能获得任务所需范围；文件、数据库、网络和代码执行进入沙箱。高风险写操作需要审批、额度、双人复核或策略引擎门控。

平台还要具备配额、限流、超时、取消、审计、密钥轮换、敏感信息脱敏和数据删除能力。可观测性要记录租户级成本、延迟、失败和资源使用，但日志默认不保存完整 Prompt、Token 或敏感工具返回。最终用租户越权测试、消息重复测试、故障恢复测试和副作用审计验证设计，而不是只画一张组件图。

## 手撕（穿插其中）

<a id="deepseek-agent-20260908-q18"></a>
##### 18. 实现一个简单的 FST（有限状态机）来管理 Agent 的状态。——状态包括：INIT、PLANNING、EXECUTING、CHECKING、COMPLETED、FAILED。状态转换的触发条件是什么？

**回答：**

有限状态机要显式定义状态集合、事件集合、合法转移和副作用。一个简单且安全的转移表如下：

| 当前状态 | 事件 | 下一状态 |
| --- | --- | --- |
| `INIT` | 任务校验通过 | `PLANNING` |
| `INIT` | 参数、权限或资源校验失败 | `FAILED` |
| `PLANNING` | 计划生成且通过校验 | `EXECUTING` |
| `PLANNING` | 计划非法、超预算或规划失败 | `FAILED` |
| `EXECUTING` | 工具执行完成 | `CHECKING` |
| `EXECUTING` | 可恢复错误且重试预算未耗尽 | `EXECUTING` |
| `EXECUTING` | 不可恢复错误、取消或超时 | `FAILED` |
| `CHECKING` | 结果满足完成条件 | `COMPLETED` |
| `CHECKING` | 需要下一步动作 | `PLANNING` 或 `EXECUTING` |
| `CHECKING` | 校验失败且无法修复 | `FAILED` |

`COMPLETED` 和 `FAILED` 是终态，除非业务明确提供补偿或重启事件，否则不能隐式跳出。每次转移应记录旧状态、事件、操作者、任务版本、时间和错误信息，并用 compare-and-set 或数据库条件更新保证并发下不会发生双重转移。

```python
from enum import Enum, auto


class State(Enum):
    INIT = auto()
    PLANNING = auto()
    EXECUTING = auto()
    CHECKING = auto()
    COMPLETED = auto()
    FAILED = auto()


TRANSITIONS = {
    (State.INIT, "validated"): State.PLANNING,
    (State.INIT, "validation_failed"): State.FAILED,
    (State.PLANNING, "plan_ready"): State.EXECUTING,
    (State.PLANNING, "plan_failed"): State.FAILED,
    (State.EXECUTING, "tool_done"): State.CHECKING,
    (State.EXECUTING, "retry"): State.EXECUTING,
    (State.EXECUTING, "execution_failed"): State.FAILED,
    (State.CHECKING, "accepted"): State.COMPLETED,
    (State.CHECKING, "needs_next_step"): State.PLANNING,
    (State.CHECKING, "check_failed"): State.FAILED,
}


def transition(state: State, event: str) -> State:
    if state in (State.COMPLETED, State.FAILED):
        raise ValueError(f"terminal state: {state.name}")
    try:
        return TRANSITIONS[(state, event)]
    except KeyError as exc:
        raise ValueError(f"illegal transition: {state.name} + {event}") from exc
```

生产实现还要将状态机与任务数据、事件日志和取消令牌结合，避免只在内存中修改枚举。状态机解决“现在处于哪一步、哪些事件合法”，不等于解决工具幂等、分布式一致性和最终结果正确性。

<a id="deepseek-agent-20260908-q19"></a>
##### 19. 手写一个简单的 Agent Loop 框架（Python 伪代码），支持：工具注册、ReAct 循环、最大步数限制、错误处理。

**回答：**

下面的伪代码将模型决策、工具注册、参数校验、循环预算和异常处理分开。真实生产代码还应增加权限、超时、取消、幂等、持久化、流式输出和审计。

```python
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class Tool:
    name: str
    description: str
    function: Callable[..., Any]


class AgentLoop:
    def __init__(self, model, max_steps: int = 8):
        self.model = model
        self.max_steps = max_steps
        self.tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if not tool.name or tool.name in self.tools:
            raise ValueError("invalid or duplicated tool name")
        self.tools[tool.name] = tool

    def run(self, user_input: str) -> str:
        messages = [{"role": "user", "content": user_input}]
        errors = 0

        for step in range(self.max_steps):
            try:
                decision = self.model.decide(
                    messages=messages,
                    tools=[
                        {"name": t.name, "description": t.description}
                        for t in self.tools.values()
                    ],
                )
                # decide must return {type: final|tool_call, ...}.
                if decision["type"] == "final":
                    return decision["content"]
                if decision["type"] != "tool_call":
                    raise ValueError("unknown decision type")

                name = decision["name"]
                arguments = decision.get("arguments", {})
                if name not in self.tools:
                    raise ValueError(f"unknown tool: {name}")
                tool = self.tools[name]
                validate_arguments(tool, arguments)
                result = tool.function(**arguments)
                observation = bound_observation(result)
                messages.append({"role": "assistant", "content": decision})
                messages.append({"role": "tool", "name": name, "content": observation})
                errors = 0
            except RetryableToolError as exc:
                errors += 1
                messages.append({"role": "tool", "content": f"retryable error: {exc}"})
                if errors >= 2:
                    return "工具连续失败，任务暂停，请检查后重试。"
            except (ValueError, ToolError) as exc:
                messages.append({"role": "system", "content": f"action rejected: {exc}"})
                errors += 1
                if errors >= 2:
                    return "工具调用未通过校验，任务终止。"
            except Exception:
                # 真实代码应记录 trace，不把堆栈直接暴露给模型或用户。
                return "执行过程中发生内部错误，任务终止。"

        return "已达到最大执行步数，任务未完成。"
```

关键点有四个：第一，工具注册信息要有 schema，不只是一段描述；第二，模型返回的工具名和参数必须在服务端重新校验；第三，异常要区分可重试、不可重试和未知异常，不能无限循环；第四，达到最大步数必须有明确失败状态，并保存轨迹供诊断。若工具有写操作，还要把 `request_id` 作为幂等键传入执行层。

## 反问

<a id="deepseek-agent-20260908-q20"></a>
##### 20. 面试官问：你还有什么想问我的？——可以反问团队的技术路线、未来的方向、面临的挑战等深度问题。

**回答：**

反问应帮助自己理解岗位真实边界，而不是重复询问招聘信息。可以选择以下方向：

1. 团队当前更核心的 Agent 问题是模型能力、工具调用、上下文工程、评测体系还是运行时基础设施？未来半年最重要的技术目标是什么？
2. 线上主要优化哪些指标：任务成功率、人工替代率、事实一致性、延迟、token 成本，还是高风险操作的安全性？这些指标如何定义和归因？
3. 当前系统更偏固定 Workflow、ReAct、Plan-and-Execute 还是混合架构？在什么条件下会从 Agent 退回确定性流程？
4. 团队如何构造评测集、做线上灰度和 badcase 回流？模型、Prompt、工具和 Harness 的版本如何管理与回滚？
5. 这个岗位在模型训练、Agent Runtime、代码基础设施和业务落地之间的职责比例怎样？入职后前三个月最希望解决哪一个具体问题？

高质量反问的共同点是围绕“目标、约束、证据和责任边界”。听到回答后还可以追问一个具体指标或失败案例，例如“如果任务成功率提升但延迟和成本同时上升，团队通常如何做权衡？”这比泛泛地问“团队氛围怎么样”更能体现对真实工程问题的理解。
