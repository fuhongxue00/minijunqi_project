
# 简化版四国军棋（6x6，一对一）AI 项目骨架（已按固定部署顺序改造）

主要更新：
- 修复脚本导入路径（sys.path 注入）。
- 渲染使用中文字体（优先 WenQuanYi / WQY Micro Hei），文本尺寸改为 textbbox，ASCII 标注使用真实棋子 owner。
- 部署策略改为“固定顺序给定棋子，仅预测落点”，并在网络输入中加入 `is_deploy` 通道（总通道=13）。

## 依赖
```bash
pip install -r requirements.txt
```

## 运行
```bash
python scripts/ai_vs_ai.py --step
python scripts/ai_vs_human.py --human_side RED
python scripts/ai_vs_input.py --ai_side RED
python scripts/render_replay.py --replay replays/ai_vs_ai.json --step
```

## 训练
```bash
python -m minijunqi.ai.train_supervised --replays replays/*.json --epochs 1 --out checkpoints/sup.pt
python -m minijunqi.ai.train_rl --episodes 10 --out checkpoints/rl.pt
```
