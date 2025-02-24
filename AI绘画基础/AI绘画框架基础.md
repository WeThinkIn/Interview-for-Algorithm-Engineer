# 目录

- [1.目前主流的AI绘画框架有哪些？](#1.目前主流的AI绘画框架有哪些？)
- [2.Stable Diffusion WebUI中Variation seed如何工作的？](#2.Stable-Diffusion-WebUI中Variation-seed如何工作的？)


<h2 id="1.目前主流的AI绘画框架有哪些？">1.目前主流的AI绘画框架有哪些？</h2>

Rocky从AIGC时代的工业界、应用界、竞赛界以及学术界出发，总结了目前主流的AI绘画框架：

1. Diffusers：`diffusers` 库提供了一整套用于训练、推理和评估扩散模型的工具。它的设计目标是简化扩散模型的使用和实验，并提供与 `Hugging Face` 生态系统的无缝集成，包括其 `Transformers` 库和 `Datasets` 库。在AIGC时代中，每次里程碑式的模型发布后，Diffusers几乎都在第一时间进行了原生支持。
![diffusers](./imgs/diffusers图标.png)
2. Stable Diffusion WebUI：`Stable Diffusion Webui` 是一个基于 `Gradio` 框架的GUI界面，可以方便的使用Stable Diffusion系列模型，使用户能够轻松的进行AI绘画。
![Stable Diffusion WebUI](./imgs/WebUI图标.png)
3. ComfyUI：`ComfyUI` 也是一个基于 `Gradio` 框架的GUI界面，与Stable Diffusion WebUI不同的是，ComfyUI框架中侧重构建AI绘画节点和工作流，用户可以通过连接不同的节点来设计和执行AI绘画功能。
![ComfyUI](./imgs/comfyui图标.png)
4. SD.Next：`SD.Next` 基于Stable Diffusion WebUI开发，构建提供了更多高级的功能。在支持Stable Diffusion的基础上，还支持Kandinsky、DeepFloyd IF、Lightning、Segmind、Kandinsky、Pixart-α、Pixart-Σ、Stable Cascade、Würstchen、aMUSEd、UniDiffusion、Hyper-SD、HunyuanDiT等AI绘画模型的使用。
![SDNext](./imgs/SDNext图标.jpeg)
5. Fooocus：`Fooocus` 也是基于 `Gradio` 框架的GUI界面，Fooocus借鉴了Stable Diffusion WebUI和Midjourney的优势，具有离线、开源、免费、无需手动调整、用户只需关注提示和图像等特点。
![Fooocus](./imgs/Fooocus图标.png)


<h2 id="2.Stable-Diffusion-WebUI中Variation-seed如何工作的？">2.Stable Diffusion WebUI中Variation seed如何工作的？</h2>

在Stable Diffusion WebUI中，**Variation Seed**（变体种子）是一个关键参数，用于在保持图像整体结构的前提下，生成与原始种子（Seed）相关联但有细微变化的图像。它的核心思想是通过 **噪声插值** 和 **种子偏移** 等方式控制生成结果的多样性。

通过 **Variation Seed**，用户可以在保留生成图像核心特征的同时，探索细节的多样性，是平衡“稳定性”与“随机性”的重要工具。掌握 Variation Seed 的使用技巧，可显著提升创作效率，尤其在需要批量生成或精细调整的场景中表现突出。

### **1. Variation Seed相关基础概念**

- **Seed（种子）**：  
  在 Stable Diffusion 中，种子决定了生成过程的初始随机噪声。**相同的种子 + 相同参数 + 相同提示词** 会生成完全一致的图像。
- **Variation Seed（变体种子）**：  
  通过引入第二个种子（Variation Seed），并结合 **Variation Strength（变体强度）** 参数，系统会在原始种子和变体种子生成的噪声之间进行插值，从而产生可控的随机变化。

### **2. 核心工作原理**
**Variation Seed 的实现逻辑可以分为以下步骤：**

1. **生成初始噪声**：  
   - 使用 **原始种子（Original Seed）** 生成初始噪声图 $N_{\text{original}}$ 。
   - 使用 **变体种子（Variation Seed）** 生成另一个噪声图 $N_{\text{variation}}$ 。

2. **噪声混合**：  
   根据 **Variation Strength** 参数（取值范围 $[0,1]$ ），对两个噪声图进行线性插值：  

   $N_{\text{final}} = (1 - \alpha) \cdot N_{\text{original}} + \alpha \cdot N_{\text{variation}}$
     
   其中 $\alpha$ 为 Variation Strength 的值：
   - $\alpha=0$ ：完全使用原始种子噪声，结果与原始种子一致。
   - $\alpha=1$ ：完全使用变体种子噪声，等同于直接替换种子。
   - $0 < \alpha <1$ ：混合两种噪声，生成介于两者之间的结果。

3. **去噪生成**：  
   基于混合后的噪声 $N_{\text{final}}$ ，通过扩散模型的去噪过程生成最终图像。由于噪声的微小变化，输出图像会保留原始种子的整体结构，但在细节（如纹理、光照、局部元素）上产生差异。

### **3. 参数交互与效果控制**

- **Variation Strength**：  
  - 控制变化的剧烈程度。较小的值（如 0.2）产生细微调整，较大的值（如 0.8）导致显著差异。
  - **示例**：当生成人像时， $\alpha=0.1$ 可能仅改变发丝细节，而 $\alpha=0.5$ 可能调整面部表情和背景。

- **Resize Seed from Image**：  
  在 WebUI 中，可通过上传图像反推其种子（使用 "Extra" 选项卡），再结合 Variation Seed 生成相似但不同的变体。

### **4. 应用场景与案例**

#### **场景 1：微调生成结果**
- **需求**：生成一张基本满意的图像，但希望调整局部细节（如云层形状、服装纹理）。
- **操作**：  
  1. 固定原始种子，设置 Variation Seed 为新值。  
  2. 逐步增加 Variation Strength（如从 0.1 到 0.3），观察变化是否符合预期。

#### **场景 2：探索多样性**
- **需求**：基于同一提示词生成多张不同但风格统一的图像。
- **操作**：  
  1. 固定原始种子，批量生成时随机设置多个 Variation Seed。  
  2. 设置 Variation Strength 为中等值（如 0.5），平衡一致性与多样性。

#### **场景 3：修复缺陷**
- **需求**：原始种子生成的图像存在局部缺陷（如扭曲的手部），需调整而不改变整体构图。
- **操作**：  
  1. 使用 Inpainting 功能局部修复。  
  2. 结合 Variation Seed 生成多个修复版本，选择最优结果。

### **5. Variation Seed技术扩展应用**
- **噪声空间的连续性**：  
  Stable Diffusion 的噪声空间是连续的，微小的噪声变化会导致生成结果的平滑过渡。这种特性使得 Variation Seed 能够实现可控的多样性。

- **数学扩展**：  
  某些高级实现（如 WebUI 的 "X/Y/Z Plot" 脚本）允许同时测试多个 Variation Seed 和 Strength 组合，生成对比网格图。

- **与 CFG Scale 的交互**：  
  Variation Seed 的变化效果受 **Classifier-Free Guidance Scale（CFG Scale）** 影响。较高的 CFG Scale（如 12）会放大提示词的控制力，可能减弱 Variation Seed 的多样性表现。


### **6. 示例流程（WebUI 操作）**
1. **生成原始图像**：  
   - 输入提示词：`A futuristic cityscape at sunset, neon lights, cyberpunk style`  
   - 设置 Seed：`12345`，生成图像 A。

2. **启用 Variation Seed**：  
   - 勾选 "Enable Variation Seed"，设置 Variation Seed：`67890`，Variation Strength：`0.3`。  
   - 生成图像 B，观察霓虹灯颜色和建筑细节的变化。

3. **调整 Strength**：  
   - 将 Variation Strength 提高到 `0.6`，生成图像 C，对比云层形态和光照方向的差异。
  
