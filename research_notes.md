## ivr数据集使用说明
1. ivr数据集路径：/mnt/data/oss_wanjun/pai_work/open_research/dataset/ivr_sample_v16_ctcvr_sample/；
2. 该数据集已经包含的 train和 test 目录，分别代表训练集和测试集，不需要再次划分，也不需要任何采样；
3. ctcvr 模型或者 cvr 模型统一使用 ctcvr_label, ctr模型使用 ctr_label；
4. 所有特征都默认是类别特征，不需要再做任何处理；
5. deviceid 标识用户 id，一般不做为特征；

## 实验流程规范
1. Git 仓库统一在 /mnt/workspace/git_project/AutoResearchClaw/，
项目开发根路径统一在 /mnt/workspace/open_research/autoresearch/project_name/，每个项目单独创建自己的项目目录, 所有日志、临时文件、数据、模型等文件以及超过 1M 的文件不要提交到 git；                                  
1. 提交代码时，要同步更新 research_list.md，包括开发路径和 Git 仓库里的；
2. 模型评估指标要包括 AUC、PCOC，包括全局维度和 business_type维度；
3. CTR/CVR 模型只跑 1 epoch（广告流式数据，多 epoch 会过拟合）；
4. 每个项目必须有 README.md（记录项目背景介绍，实验内容，实验方向，数据集，目录结构等等）和experiment_report.md（记录每个实验的评估结果以及简介）；
5. 每个项目下面，用 experiments 目录存放所有的实验，每个实验一个目录，存放实验的所有文件，如实验设计、参考 paper、代码、数据、结果等等;
6. 所有实验尽可能共用一个 baseline，避免重复训练；
7. 参考/mnt/workspace/open_research/autoresearch/multi_grained_id/EXPKIT_REUSE_GUIDE.md下面的实验管理系统实现 GPU 智能调度和实验智能调度；
