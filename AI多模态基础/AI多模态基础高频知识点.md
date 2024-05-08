# 目录

- [1.BLIP的原理？](#1.BLIP的原理？)
- [2.CLIP的原理？](#2.CLIP的原理？)
- [3.为什么StableDiffusion使用CLIP而不使用BLIP?](#3.为什么StableDiffusion使用CLIP而不使用BLIP?)

<h2 id="1.BLIP的原理？">1.BLIP的原理？</h2>

BLIP是一种统一视觉语言理解和生成的预训练模型。BLIP的特点在于它采用了一种编码器-解码器混合架构（MED），并且引入了CapFilt机制来提高数据质量和模型性能。BLIP的主要组成部分包括：

1. MED架构：包括单模态编码器、图像引导的文本编码器和图像引导的文本解码器，这使得BLIP能够同时处理理解和生成任务。
2. 预训练目标：BLIP在预训练期间联合优化了三个目标，包括图文对比学习、图文匹配和图像条件语言建模。
3. CapFilt机制：包括Captioner和Filter两个模块，Captioner用于生成图像的文本描述，而Filter用于从生成的描述中去除噪声，从而提高数据集的质量。

![](./imgs/BLIP.png)

<h2 id="2.CLIP的原理？">2.CLIP的原理？</h2>

CLIP是由OpenAI提出的一种多模态预训练模型，它通过对比学习的方式，使用大规模的图像和文本数据对来进行预训练。CLIP模型包括两个主要部分：

Text Encoder：用于提取文本的特征，通常采用基于Transformer的模型。

Image Encoder：用于提取图像的特征，可以采用CNN或基于Transformer的Vision Transformer。
![](./imgs/CLIP.png)
CLIP的训练过程涉及将文本特征和图像特征进行对比学习，使得模型能够学习到文本和图像之间的匹配关系。CLIP能够实现zero-shot分类，即在没有特定任务的训练数据的情况下，通过对图像进行分类预测其对应的文本描述。

<h2 id="3.为什么StableDiffusion使用CLIP而不使用BLIP?">3.为什么StableDiffusion使用CLIP而不使用BLIP? </h2>

CLIP是通过对比学习的方式训练图像和文本的编码器，使得图像和文本之间的语义空间能够对齐。CLIP的架构和训练方式可能更适合Stable Diffusion模型的目标，即生成与文本描述相匹配的高质量图像。

BLIP由于其图像特征受到了图文匹配（ITM）和图像条件语言建模(LM)的影响，可以理解为其图像特征和文本特征在语义空间不算对齐的。

最大区别：损失函数，CLIP和BLIP针对任务不同，不同任务不同损失函数。