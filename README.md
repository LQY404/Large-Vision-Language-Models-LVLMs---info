# Large-Vision-Language-Models-LVLMs--info
This is a repository of Large-scale Vision-language models.

- 通义千问
  - paper: [Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond](https://arxiv.org/abs/2308.12966)，  [code](https://github.com/QwenLM/Qwen-VL)
  - 模型基本结构：多模态大模型，包括Large Language Model，Visual Encoder，Position-aware Vision-Language Adapter三部分
    - Large Language Model：直接使用Qwen-7B的结构，并使用对应的预训练权重进行初始化
    - Visual Encoder：使用ViT架构，使用Openclip’s ViT-bigG的权重对模型进行初始化，训练和推理时均会将image resize到固定尺寸，并且将其切片成若干patch
    - Position-aware Vision-Language Adapter：主要为了缓解图片序列过长引起的性能问题。解决方案是使用一个预定义的\[N, D\]维度可学习query、以及cross attention模块，以Visual Encoder得到的图像特征为key、query为value，从而将整个图像特征的长度固定到N（论文中做了消融实验），相当于对图像特征进行压缩。除此之外，在做cross-attention时还会添加绝对位置信息
    - 处理流程：输入图像信息先经过Visual Encoder得到若干图像特征，再经过Position-aware Vision-Language Adapter模块对图像特征进行压缩，得到固定长度的图像特征。最后将图像特征和文本信息一起输入到Large Language Model中得到输出结果
      
  - 训练：包括三个阶段
    - stage1：pretraining，frozen Large Language Model
      - 主要使用公开的image-text数据集进行调整，主要数据来源是LAION-5B，LAION-COCO等，包括77%的英文数据和23%的中文数据。这里使用的标签可能存在问题，质量不高，并且只使用224x244的低分辨率图像
    - stage2：Multi-task Pretraining，全参数调整
      - 使用高质量、细粒度数据进行模型调整，包括VAQ、Referring系列（Refercoco等）。除此之外，还使用了OCR数据
      - 将图像分辨率从224x224提高到448x448
      - 提高压缩后的图像特征长度（256->1024）
    - stage3：finetuneing，frozen Visual Encoder
      - 主要增强模型的对话和prompt能力（因此没有frozen LLM部分）
      - 构造了跨图像的对话理解数据，增强模型跨图像的理解能力
    - 三个阶段Position-aware Vision-Language Adapter都进行调整

  - 输入处理细节
    - 图像特征会使用\<img\>...\</img\>标签包围，与text特征进行区分
    - 输入的数据包括图像-文本描述（region descriptions）、问答（VQA），以及grouding（detections）
    - 对于detections，bbox都将其转化成"(X_top_left, Y_top_left),(X_bottom_right, Y_bottom_right)"，并都使用\<box\>和\</box\>包括以和普通的text进行区分。同时，对于bbox指代的文本，使用\<ref\>和\</ref\>包括
   
  - 重要的点（个人）
    - 模型特征压缩
    - 数据构造（论文附录比较清楚）
    - 分阶段微调

- deepseek
  - paper: [Deepseek-VL](https://arxiv.org/abs/2403.05525), code: https://github.com/deepseek-ai/DeepSeek-VL
  - 模型基本结构：基本和Qwen-VL一样，三部分，a hybrid vision encoder, a vision adaptor, and a language model（LLM）
    - hybrid vision encoder
      - 使用SigLIP处理低分辨率的图像（384x384）, SAM-B处理高分辨率图像（1024x1024），分别能够得到hxwxD的feature map
    - vision adaptor
      - 处理由hybrid vision encoder送过来的feature map
      - 具体来说，先将feature map上采样2倍并输入两层卷积，随后将feature map拉直，得到NxD维特征（类似token），在论文中每个feature map处理后都得到576x1024的token，最后将两种token在通道维度拼接得到576x2048的visual token
      - 最后使用一层GELU+MLP做embedding，作为LLM的输入
    - LLM
      - 使用DeepSeek LLM，包括1B和7B
  - 训练阶段：和Qwen-VL一样分为三个阶段训练
    - stage 1: Training VL Adaptor
      - 对vision adaptor进行训练，其他部分均frozen，相当于固定视觉和文本编码器，训练两者的融合模块。这里有一点可以关注，VLA的参数量很少，scaling law几乎无效，甚至会起到反作用，因此在这个阶段没有用很多数据进行调整
    - stage 2: Joint VL Pre-training
      - 对除了hybrid vision encoder外的所有参数进行调整，主要用来训练模型的多模态能力。在这个阶段需要谨慎的控制好用于训练的text-image和text数据的比率，否则会造成模型的language能力下降
    - stage 3: Supervised Finetuning
      - 全参数调整（但frozen SAM-B，显存限制）
  - 数据部分：通常的处理，划分为pretraining dataset和fine-tuning dataset
    - pretraining dataset主要由一些比较杂的数据构成（论文table 1），主要参与训练的stage1
    - fine-tuning dataset数据比较干净，包括LAION等，主要参与训练的stage3
    - 两者共同参与训练的stage2
  - 重要的点（个人）
    - 高质量图像数据（1024x1024）， hybrid vision encoder
    - modality warm-up，逐步增加text-image数据，初始保持纯text数据在训练过程中占主导，防止模型language能力出现degradation问题
    - 论文中的性能对比上，基本能干过当时的开源LVLMs，但和GPT4-v有差距

- deepseek
  - paper: [Deepseek-VL2](https://arxiv.org/abs/2412.10302), code: https://github.com/deepseek-ai/DeepSeek-VL2
  - 模型结构：整体上同样由三部分构成，具体来说
    - LLM部分
      - 与一代版本不同的是，LLM部分采用了基于MoE架构的DeepseekMoE
    - vision encoder
      - 不同于上一代使用两个不同的visual encoder来处理低、高分辨率图像，VL2只使用一个ViT作为visual encoder（SigLip）
      - 为了处理不同尺寸的图像，VL2不会进行resize操作，而是将任意尺寸的的图像全部使用resize+padding的方式扩充到n*384xm*384，这样就可以将整个图像划分成n*m个384x384的patch（大小尺寸与SigLip一致），以次处理任意分辨率的图像
      - 同时，这样resize再划分的策略产生的特征基本都是局部性质的，因此会再添加一个对整体图像encode的feature，使用分隔符与patch的feature分隔开后一起拼接
      - 整体设计上感觉是参考了llava-next
    - vision adaptor
      - 相当简化，由几层MLP构成
  - 训练阶段：三个阶段训练
    - stage 1
      - 同时对visual encoder和vision-language adaptor（MLP）进行调整
    - stage 2
      - 与deepseek-vl的stage 2一样，对vision-language adaptor和LLM进行调整
    - stage 3
      - 全参数调整
    
  - 重要的点（个人）
    - 更强力的LLM
    - 更先进的图像编码方式，保存了更多的细节信息
    - 更精细的数据构造
- Meta
  - paper: [LLaMA: Open and efficient foundation language models](https://arxiv.org/pdf/2302.13971), code: https://github.com/facebookresearch/llama
  - 数据
    - 只用到公开的数据集（因此能够做到完全公开，包括模型、代码、数据、模型参数），完整可复现
    - 从互联网、arxiv、GitHub、Wikipedia等来源爬取了大量数据，并且对数据进行了大规模清洗（清洗方法需要注意）
    - 使用大家都使用的tokenizer——byte pair encoding(BPE)对字符串进行编码
  - 架构：整体上是transformer结构，细节不一样
    - 归一化：使用pre-normalization策略（每一层的输入先做归一化再进MHA，而不是对每一层的输出做归一化），归一化使用RMSNorm而不是原始的LN（RMSNorm不需要计算方差，速度很快，且效果更好）
    - 激活函数：原始transformer使用类ReLU，这里LLAMA使用SwiGLU，至于为什么用这个，We offer no explanation as to why these architectures seem to work; we  attribute their success, as all else, to divine benevolence.）
    - 位置编码使用RoPE（可以从二维的情况理解，此时位置编码就是用来对q/k做旋转的）
  - 个人评价
    - 开源的大模型，并且发布了完整的cpp部署文档
    - 同时支持非常多的LLM以及LVLM，对落地部署非常友好
    - 和vllm有得一拼

- personal
  - paper: [Visual Instruction Tuning](https://arxiv.org/pdf/2304.08485), code: https://llava-vl.github.io
  - 模型
    - 简称LLaVA（Large Language and Vision Assistant），名称已经体现了该论文主要关注多模态大模型
    - 模型结构很简单，使用CLIP对image进行编码，使用LLM对文本进行编码
    - 使用简单的MLP对image token进行映射，将映射后的image token和经过embedding layer得到的text token进行拼接，直接送入LLM即可
   
  - 数据
    - 使用chatgpt来构造image-text pair数据集
   
- personal
  - paper: [LLaVA-Mini: Efficient Image and Video Large Multimodal Models with One Vision Token](https://arxiv.org/abs/2501.03895), code: https://github.com/ictnlp/LLaVA-Mini
  - 核心点
    - 实验发现，在多模态融合时，更前面的层visual token更重要，越到后面的trans层，visual token越不重要
    - 因为随着layer的加深，visual token中的信息已经逐渐被cross-attention机制注入到text token中了
    - 因此，该论文提出在更早期的层进行多模态token的融合，需要注意的是，多模态特征融合在送入LLM之前
    - 融合后的text token中已经充满了visual token的信息。因此，对于visual token，则使用一个预定义的query，与原本的visual token做cross attention做token的compression
    - 最后将compression后的visual token（C^2，可调为1）和融合后的text token拼接送入LLM即可

- personal
  - paper: [LISA: Reasoning Segmentation via Large Language Model](https://arxiv.org/pdf/2308.00692), code: https://github.com/dvlab-research/LISA
  - 解决问题
    - 借助LLM来做referring segmentation
  - 框架结构
    - VLLM，使用LLaVA，但freeze，加了个LoRA来fine-tune
    - visual encoder，用来对输入的image进行编码，有用到SAM或者Mask2Former，freeze
    - decoder，对来自visual encoder的图像特征，以及来自VLLM的多模态特征（聚合到\[SEG\]中）进行融合，生成mask，结构和SAM类似
  - 个人评价
    - 这个工作可以重点关注，后LLM时代下，referring segmentation究竟该怎么走，和T-Rex2一样成为令我印象比较深刻的工作
    - 在之前的工作中，借助LLM来做理解（无论时det.还是seg.），基本都是以只输出text的形式（比如做det./grounding时，则是将bbox的坐标变成str来生成）。LISA比较有开创性，CVPR oral还是可以的
    - 除了解决问题上的范式变化，还有就是造数据的方式同样值得学习，数据包括semantic+text、“古老时代”的referring segmentation数据，以及VQA数据共同组成
    - 从流程上来说，其结构可以理解为
      - 使用VLLM处理多模态信息，得到多模态特征
      - 将多模态特征在SAM的decoder阶段进行融合，最后输出mask
    - 所以其实结构上很简单的，主要是解决问题的思路（讲故事的方式doge）
  - 后续
    - 后续还有[LISA+++](https://arxiv.org/pdf/2312.17240)，就是使用的LISA的模型，主要在数据设计上，以增强模型的推理能力

- personal
  - paper: [Sa2VA: Marrying SAM2 with LLaVA for Dense Grounded Understanding of Images and Videos](https://arxiv.org/pdf/2501.04001), code: https://github.com/magic-research/Sa2VA
  - 个人评价
    - 出来不久（20240107上传的arxiv），只能说太卷了（）
    - 整体结构上来看其实和LISA的范式没有大的区别，包括模型框架组成也是类似（VLLM+SAM2），只是在细节上有一定差别
    - 主要亮点的认为主要在于：1）图像，视频，text/visual/image级别的prompt均得到支持；2）数据构造方式（这个非常值得借鉴）：从object到scene，再到video

- personal
  - paper: [GLaMM: Pixel Grounding Large Multimodal Model](https://arxiv.org/abs/2311.03356), code: https://mbzuai-oryx.github.io/groundingLMM
  - 解决问题
    - 和LISA专注的类似
    - 但增加了visual prompt（single or multiple）的输入
    - 同时，增加了多轮对话能力，以及visual部分的encoder
  - 模型结构
    - visual encoder，global visual encoder编码得到feature maps，visual prompt使用RoIAlign的方法从feature maps中得到14x14的visual prompt特征
    - grounding image encoder，专门对全局图像进行编码
    - pixel decoder，接受来自VLM的输出，以及grounding image encoder的输出作为输入，得到binary mask的分割结果
    - VLM，以输入text和visual prompt特征作为输入，产生多模态特征
    - 总体来说，从结构上，比LISA多visual级别的prompt编码，以及一个额外单独完整的对image的encoder+decoder，其他没有大的区别

    
