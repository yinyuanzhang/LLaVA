1. 一. llava-android-control 数据集 输出token过多，指令不跟随。
2. 
3. 二. llava-cacheblend实现

1. 一个理解，另一个规划，我去通读代码。





1. 先梳理现状，即把llava的现有实现代码看完再说。





所有的llama如果能跑完，实验部分基本就可行了。




15-17 完成cacheblend-write 模式
    - llava_llama.py 真实调用 vit，返回背景 & 前景
    - 背景token单独调用llm推理，持久化kv cache
    - 调用原生逻辑，可以正常输出
17-18 完成pipeline流程控制
    - 缓存写入后的正常调用其pipeline是否正确，输出是否正确。
    - 其他pipeline的理论调用逻辑是否正确


cacheblend的read-load 实现
    - 


- 缓存库写入时对齐qwen25vl的缓存库写入
    -- qwen25vl的处理好像也不对
- 元数据处理
- 逐层实现




- 












先让llava跑出来结果，然后

今天任务-项目实现。[先把llava项目解决，然后]
目的：我们需要在该llava项目上完成llava模型的cacheblend实现(前-背景分离复用cache)。
1. /data/zyy/Qwen2.5-VL 目录下有关于Qwen2.5-VL关于cacheblend的实现，已经能够准确运行，注意cacheblend算法的真正实现核心是：llm层面选择重要的token，并且对于复用的vit token对应的llm层面的kv cache进行重计算。
  输入图像
      ↓
  【ViT层面】- 只做特征提取和相似度匹配
  ├── 视觉编码器提取图像特征
  ├── 前背景分离 (复用LLaVA现有逻辑)
  ├── 特征归一化用于相似度搜索
  └── 缓存命中检测 (bg_hit, fg_hit)
      ↓
  【Language Model层面】- KV融合和重计算的核心
  ├── Layer 0 (Status=1): 重要性计算 + Query稀疏化
  ├── Layer 1+ (Status=2): KV张量融合
  └── 标准Attention计算
2. 请仔细阅读当前项目，该目录下有目前llava的实现，包括支持 原生、完整图片模糊缓存、背景-前景分离等多种方式，仔细分析代码是在哪里实现的vit，哪里实现的llm的推理。
3. 你拥有对所有文件的读权限，你能否在不改变对当前多种方法(兼容现有方法)的情况下，规划llava模型对cacheblend的实现。
4. 由于现有的 llava实现(支持 原生、完整图片模糊缓存、背景-前景分离等)逻辑等验证是无误的，因此 cacheblend的实现务必遵循最小修改原则，尽量复用现有的实现。
5. 由于我们的实现需要支持前/背景命中后复用的，即只要存在部分命中，另一部分需全量重计算。
6. 缓存持久化方式和现有的llava-segementation cache实现一致(也需要归一化)。
7. 请不要先直接实现，你需要在阅读代码后，仔细进行规划(我目前的理解是vit层面复用现有的segementation-cache实现，llm层面对命中缓存的vit token执行重计算等。目前的注意力计算在 HuggingFace Transformers 的 Llama 模型里完成（transformers.models.llama.modeling_llama 中的 LlamaAttention.forward）。LLaVA 本仓库不重写 Llama 的注意力算子（native 模式），因此我们是不是需要通过继承的方式，以最小化修改的方式重写其整体的注意力计算流程；不一定正确，仅供参考)，有不确定的可以和我先讨论。




1. 一致性 每天都要有收获。而不是2天工作很多，5天又休息很多。
2. 论文要求，论文不是赶出来的。
3. 看到别人中了一篇研究，觉得我也也能在这个基础上做这个研究，需要solid



9.24 要求：
1. llava-cacheblend 实现




中午：
- 解决 past_key_value 的问题


注意点1:GQA的实现位置
注意点2:llava中 apply_rotary_pos_emb(q, k, cos, sin, position_ids) 位置编码的实现原理
注意点3:attention_mask 这里为none
注意力4:稀疏注意力的实现方式：
    - Sparse Hidden States (写入/重建完整的kv cache需要把 Sparse Hidden States 和 old kv cache结合)
    - All Hidden States (直接用 old kv cache 替换 Hidden States即可)
注意力5:prefill阶段和decodejie 是ru h




下午通读理解 cacheblend的实现。






llava-cacheblend的实现：
- native-generate 实现了多模态的处理；cacheblend的实现依赖于






问题1: 出现 key与value的size不匹配的问题
- 研读目前的llava-cacheblend全部实现

- 研究原生的forward全部实现，分析哪里出了问题






过两遍逻辑：
原生llava实现：
- 需要过vit内部的实现了
- attention_mask 怎么处理的？在哪处理的？什么格式？
- position_ids/position_embeddings 怎么处理的？在哪处理的？什么格式？
    - vit内部：ViT 内部 (视觉编码)   可学习的绝对位置编码 [可以理解为 经过vit后的embedding是无问题的，但是llm处理的序列是乱序的]
    - llm (rope 处理，目前乱序，是有问题的)
- past_key_value 怎么传递的？
重点观察 mqa时如何实现的，传递的kv cache到底是否完整。




llava-cacheblend实现：
- 写入时的key-value设计  [value存储了什么？]
    key:归一化之后的key
    value: bg_kv、fg_kv、bg_pos_ids、fg_pos_ids(分别是0...512 和 0-63)、bg_tokens、fg_tokens
- 层与层之间 kv cache传递的元数据的格式
    cached_data = {
        "hit": cache_hit,
        "bg_kv": bg_kv, "bg_tokens": bg_tokens, "bg_pos_ids": bg_pos_ids,
        "fg_kv": fg_kv, "fg_tokens": fg_tokens, "fg_pos_ids": fg_pos_ids,
        "original_bg_len": bg_embeds_flat.shape[1]
    }
    [这是很奇怪的,与qwen25vl中的元数据传递实现不符。]
    修改方向：
    - 没有old_kv_cache [仔细查看后，发现是_align_cached_kvs 完成的kv cache对齐]
    - original_bg_len由bg_tokens代替
    self.cache_fuse_metadata.update({
        "old_kvs": old_kvs,
        "image_span": (image_start_index, image_len),
        "bg_len": cached_data['bg_tokens'],
        "fg_len": cached_data['fg_tokens'],
        "is_bg_hit": cached_data['bg_kv'] is not None,
        "is_fg_hit": cached_data['fg_kv'] is not None,
    })
    [这是最后的更新后的]
- _rebuild_inputs_from_cache 设计[关键，主要为了适配输入序列改变而添加的任务]
    目前的设计实现 其embedding 是用占位符替代的，而不是真的获取到的key/value [qwen25vl 中的实现也是不对的]
    new_input_ids 中 image_ids 是用 -200 这种占位符代替的
    new_attention_mask 全为True
    new_position_ids 是从0到n-1
- _align_cached_kvs 这里没有考虑 gqa/mqa
    缓存写入/存储时用的是全部的kv cache，但其实只需要存储 对应部分，然后推理时再repeat_kv即可。


- _compute_qkv 对gqa/mqa的处理
    初始处理后 k_raw.shape torch.Size([1, 32, 638, 128])

- position_ids 是怎么处理的？尤其是 图片作了拆分
    q, k_new = apply_rotary_pos_emb(q_raw, k_raw, cos, sin, position_ids)  在attention实现内部，通过position_ids实现位置编码
    位置编码的旋转是在哪里发生作用的？
- _compute_imp_indices 
    是根据 v_new(hidden_states得来，从中取的v_new_img) 和 v_old_img 计算重要token
    img_indices 默认进行升序
    imp_indices = torch.cat([text_indices, img_indices]).unique() 目前的处理漏掉了 靠后的 sys_indices


- attention_mask 怎么处理的？在哪处理的？什么格式？
    传递给llm时 attention_mask 全为true
- past_key_value 怎么传递的？
    past_key_value 貌似是传给forward的，从最初通过 LlavaLlamaForCausalLM 传递给forward的
    past_key_value 中传递的是完整的 k_final.shape torch.Size([1, 32, 638, 128])
    self.num_heads // self.num_key_value_heads 各自为32，这里mqa难道是用了么？

- prefill & decode路由使用是否正确？
    - _should_fallback 
        -- prefill阶段(通过shape!=1判断) & cacheblend & read-load & old_kv不为空

- old-kv cache存储的是什么？ 后续的kv cache存储的是什么？
    - 只包含图片的 kv cache，且是完整的，不是多头减少的

- sparse_attention 如何计算的
    attn_output[:, :, imp_indices, :] = attn_output_sparse  相当于从始至终，不论是传递，都是使用完整的sequence，只是说 在进行attention计算的时候，对query进行截取，相当于其效果的是query部分。其他部分依然传递，但未生效。(最后一个token也是需要重计算的)
    与qwen25vl 不同，这里传递完整的 out_put, decodelayer不用单独处理，残差也能直接获取



明天把图表补齐:
- llava-cacheblend [上午]
- finegym数据集-qwen25vl & qwen2vl [下午]
- finegym数据集-llava-cacheblend [晚上]
- 小的case：qwen2vl关于 ui-agent数据集

后天:
- 




cacheblend原生实现 - qwen25vl实现 - qwen25vl原生实现 - llava实现 - llava原生实现




今天主要关注：
llava原生实现
llava-cacheblend实现
qwen25vl-cacheblend实现


逻辑修正：




1. 仔细阅读 llava1.5vl 的原生实现，理解其对于 position_ids、attention_mask 的实现设置
2. 修改 llava-cacheblend，解决其主要问题(诸多问题：一个一个解决就好)。

- 


无时无刻不感受到自己的幸运，加油！

