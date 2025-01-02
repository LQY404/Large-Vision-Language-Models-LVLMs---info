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


  
