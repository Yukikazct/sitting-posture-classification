# 论文与结果材料

本目录用于整理正式实验结果和最终英文 IEEE Conference 论文，建议正文 6–9 页。当前没有真实实验结果或论文终稿；以下仅为写作结构和材料清单，不能作为已经完成的论文提交。

## English paper outline

1. **Title and Abstract** — State the four-class sitting-posture task, pose-to-MLP pipeline, evaluation setting, and results obtained from actual runs. Write numerical claims after experiments finish.
2. **Introduction** — Explain the intended single-person posture reminder use, the course-project objectives, and the contributions supported by the completed experiments.
3. **Related Work** — Cover pose-based posture classification, lightweight classifiers, and temporal filtering. Cite the original sources for the pose model, data, and compared methods.
4. **Method** — Describe the frozen YOLO11n-pose detector, 25-dimensional normalized pose features, 64/32 MLP, angle-rule baseline, rejection conditions, EMA, and duration/recovery/cooldown reminder logic. Explain the limitation of 2D viewpoints.
5. **Data and Experimental Setup** — Document actual data versions/licenses, human label definitions and review, connected-group splits, independent participants, viewpoints, seeds 42/43/44, train-only standardization, validation-only selection, hardware, and metric definitions.
6. **Results and Discussion** — Present rules vs MLP, raw vs EMA, cross-view evaluation, confusion matrices, rejection coverage, event-level reminder measurements, model complexity, runtime, and concrete failure cases. Discuss negative results and missing measurements openly.
7. **Conclusion and Limitations** — Summarize supported findings and limitations from identity gaps, sample size, 2D geometry, occlusion, and timing evaluation. Keep the claims within the tested single-person setting.
8. **References** — Include traceable citations and dataset/model references in the required IEEE format.

## 准备的图表与证据

| 材料 | 状态 |
|---|---|
| 系统流程图与特征说明 | 待制作 |
| 来源、划分、类别及视角分布表 | 待真实数据审计后填写 |
| 真实训练/验证曲线 | 待正式训练 |
| 规则与 MLP 三种子对比、完整混淆矩阵 | 待正式评估 |
| Raw/EMA 误提醒、事件召回和延迟表 | 待连续视频及事件对齐评估 |
| 跨视角分类指标与拒识比例 | 待独立测试 |
| 分类器和全系统复杂度、资源及 FPS | 待实际测量 |
| 代表性成功与失败样例 | 待实际运行并获展示授权 |
| 英文 IEEE PDF、LaTeX 源文件与引用 | 待撰写 |
| 2–5 分钟未参与训练样本演示 | 待录制 |

正式表格依据 [实验记录模板](../docs/实验记录模板.md) 填写，并能追溯到清单、配置、权重、日志和评估输出。保留测试拒识及漏检，说明各速度/运算量统计范围。模拟检查、手写示例和未核验数字不得进入结果图表。
