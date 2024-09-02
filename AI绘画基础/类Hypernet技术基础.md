- [1.Hypernetwork是什么？](#1.Hypernetwork是什么？)
- [2.HyperDreamBooth是什么？](#2.HyperDreamBooth是什么？（2023年7月发布）)
- [3.DiffLoRA是什么？](#3.DiffLoRA是什么？(2024年8月发布))

<h2 id="1.Hypernetwork是什么？">1.Hypernetwork是什么？</h2>

Hypernetwork，也被称为“超网络”，是一种附加到 Stable Diffusion 模型的小型神经网络。它的主要作用是通过插入到噪声预测器 UNet 的交叉注意力模块中，从而改变模型的风格表现。

#### 2. Hypernetwork 与其他模型的区别

##### Hypernetwork VS Checkpoint（大模型）

- **Checkpoint 模型**：包含生成图像的所有必要信息，文件体积较大，通常在 2 GB 到 7 GB 之间。
- **Hypernetwork**：文件体积较小，通常低于 200 MB，但不能单独使用，必须与 Checkpoint 模型配合才能生成图像。

##### Hypernetwork VS LoRA 模型

- **相似性**：Hypernetwork 和 LoRA 模型在文件大小上相似，通常都在 200 MB 以下，比 Checkpoint 模型要小。
- **效果对比**：LoRA 模型一般能产生更好的效果，因此逐渐取代了 Hypernetwork 的位置。

##### Hypernetwork VS Embeddings

- **Embeddings**：通过“文本反转”（Textual Inversion）技术生成，它定义新的关键词来实现特定风格，不会改变模型结构。Embeddings 创建新的嵌入在文本编码器中。
- **Hypernetwork**：通过将一个小型网络插入到噪声预测器的交叉注意力模块中来改变模型的输出风格。

#### 3. Hypernetwork 的现状

- **使用减少**：由于 LoRA 和 Embeddings 的出现，Hypernetwork 的使用频率逐渐下降。在一些社区资源库中，Hypernetwork 文件数量非常有限。
- **效果有限**：虽然 Hypernetwork 的文件体积较大，但其效果往往不如更小的 Embeddings 文件，而这些效果可以通过其他方式实现，例如使用 Embeddings 或 LoRA 模型。



<h2 id="2.HyperDreamBooth是什么？">2.HyperDreamBooth是什么？(2023年7月发布)</h2>

论文链接：https://arxiv.org/pdf/2307.06949

这篇论文提出了一种名为 HyperDreamBooth 的新方法,用于快速和轻量级的主体驱动个性化文本到图像扩散模型。主要内容包括:

1. **轻量级 DreamBooth (LiDB)**: 提出了一种新的低维权重空间,用于模型个性化,可以将个性化模型的大小减少到原始 DreamBooth 的 0.01%。

2. **超网络架构**: 设计了一个超网络,可以从单个图像生成 LiDB 参数。超网络由 ViT 编码器和 Transformer 解码器组成。

3. **rank-relaxed 快速微调**: 提出了一种技术,可以在几秒钟内显著提高输出主体的保真度。

4. 性能

   : 与 DreamBooth 和 Textual Inversion 等方法相比,HyperDreamBooth 在速度和质量上都有显著提升:

   - 速度提高了 25 倍
   - 模型大小减少了 10000 倍
   - 在主体保真度和风格多样性方面取得了相当或更好的结果

整体框架如下图：

![image-20240902192807641](.\imgs\HyperDreamBooth.png)

Lightweight DreamBooth结构如下：

![image-20240902193005109](.\imgs\Lightweight DreamBooth.png)

HyperDreamBooth 实现了快速、轻量级和高质量的文本到图像模型个性化,为创意应用开辟了新的可能性。



<h2 id="3.DiffLoRA是什么？">3.DiffLoRA是什么？(2024年8月发布)</h2>

论文链接：https://arxiv.org/pdf/2408.06740

DiffLoRA框架包含以下关键组成部分:

1. LoRA权重自动编码器(LAE):将LoRA权重压缩到隐空间并进行重构。LAE采用1D卷积层作为主要压缩层,并引入权重保留损失来提高重构精度。
2. 混合图像特征(MIF):利用MoE启发的门控网络,将人脸特征和图像特征相结合,更好地提取身份信息。
3. 去噪过程:使用DiT架构和条件集成,通过迭代去噪生成LoRA隐表示。
4. LoRA权重数据集构建:自动化流程生成多身份LoRA权重数据集,用于训练DiffLoRA。

整体框架如下图：

![difflora](.\imgs\difflora.png)

MIF结构图:

![MIF](.\imgs\MIF.png)

这是一种利用扩散模型作为超网络来根据参考图像预测个性化低秩适应（LoRA）权重的方法。通过将这些 LoRA 权重集成到文本到图像模型中，DiffLoRA 无需进一步训练即可在推理过程中实现个性化。这是第一个利用扩散模型来生成面向身份的 LoRA 权重的模型