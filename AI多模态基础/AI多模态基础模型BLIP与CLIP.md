# 目录

- [1.BLIP的原理？](#1.BLIP的原理？)
- [2.CLIP的原理？](#2.CLIP的原理？)
- [3.为什么StableDiffusion使用CLIP而不使用BLIP?](#3.为什么StableDiffusion使用CLIP而不使用BLIP?)
- [4.CLIP的textEncoder能输入多少个单词?](#4.CLIP的textEncoder能输入多少个单词?)

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


<h2 id="4.CLIP的textEncoder能输入多少个单词?">4.CLIP的textEncoder能输入多少个单词?</h2>

**CLIP 模型中的 context_length 设置为 77**，表示每个输入句子会被 tokenized 成最多 77 个token。这个 77 并不是直接对应到 77 个单词，
因为一个单词可能会被拆分成多个 token，特别是对于较长的或不常见的单词。

在自然语言处理中，**token 通常指的是模型在处理文本时的最小单位**，可以是单个词，也可以是词的一部分或多个词的组合。
这是因为 CLIP 模型使用了 Byte-Pair Encoding (BPE) 分词器，这种方法会将常见的词作为单个 token，但会把不常见的词拆分成多个 token。

**实际例子**

为了更好地理解，我们来看一个具体的例子：

```Python
import clip

# 加载模型
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device)

# 示例句子
text = "a quick brown fox jumps over the lazy dog."

# 对句子进行 tokenization
tokenized_text = clip.tokenize([text])

print(tokenized_text)
print(tokenized_text.shape)
```

在这个例子中，我们对句子 `"a quick brown fox jumps over the lazy dog."` 进行了 tokenization。让我们看看它的输出：

```Python
tensor([[49406,    320,  1125,  2387,   539,  1906,   315,   262,   682,  1377,
            269, 49407,      0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            0]])
torch.Size([1, 77])
```

在这个例子中，句子被转换成了 77 个 token ID，其中包含了句子的 token ID 和填充的零。句子的 token 包括起始和结束的特殊 token (49406 和 49407)，
剩余的空位用 0 进行填充。

可以看到，虽然句子有 9 个单词，但经过 tokenization 后得到了 11 个 token（包括起始和结束 token），加上填充后的长度为 77。

**总结**

- context_length 设置为 77 表示模型的输入长度限制为 77 个 token。
- 77 个 token 不等同于 77 个单词，因为一个单词可能会被拆分成多个 token。
- 实际的单词数量会少于 77 个，具体取决于句子的复杂度和分词方式。
- 通常情况下，77 个 token 可以容纳大约 70 个左右的单词，这取决于句子的内容和复杂度。

为了在实际应用中得到精确的单词数量与 token 数量的关系，可以对输入文本进行 tokenization 并观察其输出。通过这种方式，可以更好地理解模型的输入限制。
