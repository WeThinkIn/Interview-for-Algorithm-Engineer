1.Stable diffusion是什么
Stable Diffusion是2022年发布的深度学习文本到图像生成模型。它主要用于根据文本的描述产生详细图像，尽管它也可以应用于其他任务，如内补绘制、外补绘制，以及在提示词指导下产生图生图的转变。
它是一种潜在扩散模型，由慕尼黑大学的CompVis研究团体开发的各种生成性人工神经网络之一。它是由初创公司StabilityAI、CompVis与Runway合作开发，并得到EleutherAI和LAION的支持。截至2022年10月，StabilityAI筹集了1.01亿美元的资金。
Stable Diffusion的源代码和模型权重已分别公开发布在GitHub和Hugging Face，可以在大多数配备有适度GPU的电脑硬件上运行。而以前的专有文生图模型（如DALL-E和Midjourney）只能通过云计算服务访问。

2.模型分类
2.1主模型（checkpoint）
●主模型（大模型）是Al绘画的基础，影响画面的整体风格。
●主模型通常较大，一般有几G到十几G。
2.2文本嵌入（Embedding）
●Embedding是指将自然语言文本（如句子或段落）转换为计算机可以理解的数值向量的过程。我们可以理解为打包了很多提示词进一个词
●embeding的大小为几十kb
●embeding倾向于训练角色特征
2.3LoRA（Low-Rank Adaptation of Large Language Models）
●LoRA是一种用于微调大模型的技术。LoRA模型应用于修改图片局部或优化图片
●LoRA模型的大小通常为几百Mb
2.4超网络（Hypernetwork）
●Hypernetwork用法和功能和embedding类似，更适合训练风格，而不是特定具象的物体。
●Hypernetwork的大小为几十kb 
2.5VAE
使用VAE可以优化画面颜色，有些VAE也会对画风产生影响

3.常见问题
3.1网址打不开?
在维护/需要魔法
3.2安装目录找不到?
在桌面右键启动器——查看安装目录
3.3安装后找不到模型?
安装后需要手动刷新模型列表
3.4下载模型后画出来的图不好看?
1.是否有触发词
2.是不是底模和VAE不一致
3.检查参数是否正确
3.5下载模型需要预留多少空间?

4.ControlNet 
4.1作用
让生成结果更可控。能够提取参考图中的重要信息，以一定权重影响生成图效果，可以用来控制角色动作、样貌、所处空间等等
4.2安装
●在StableDiffusion\extensions中删除controlnet文件夹
●扩展-可下载-加载扩展列表- 'ctrl+f'在输入框里输入ControlNet-点击install-安装完成后返回已安装-点击应用更改并重载前端

5.提示词公式
●质量词放在最前面
●其他元素根据重要性由前向后排列
●相关性比较高的词放在一起
●光照、镜头、风格、描述等
