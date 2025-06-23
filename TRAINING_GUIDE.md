# SS-UIE增强训练代码使用指南

## 主要改进

### 1. AMP (Automatic Mixed Precision) 支持
- 使用 `--use_amp` 参数启用
- 可以节省约30-50%的显存占用
- 可以提升训练速度10-20%
- 对精度影响很小

### 2. 自适应单卡/多卡训练
- 自动检测可用GPU数量
- 单GPU时不会创建不必要的DataParallel包装
- 多GPU时自动启用DataParallel
- 智能的模型保存/加载机制

### 3. 灵活的GPU配置
- 通过 `--gpu_ids` 参数指定使用的GPU
- 支持任意GPU组合，如 "0,2,3"
- 自动处理CUDA_VISIBLE_DEVICES设置

### 4. Jupyter Notebook兼容
- 支持在Jupyter Notebook中运行
- 自动检测运行环境，提供合适的默认参数

## 使用示例

### 基础训练

```bash
# 单GPU训练 (使用GPU 0)
python train.py

# 单GPU训练 (使用GPU 1)
python train.py --gpu_ids 1

# 多GPU训练 (使用GPU 0,1)
python train.py --gpu_ids 0,1
```

### AMP训练 (推荐)

```bash
# 单GPU + AMP
python train.py --use_amp --gpu_ids 0 --batch_size 6

# 多GPU + AMP
python train.py --use_amp --gpu_ids 0,1 --batch_size 12
```

### 完整参数配置

```bash
python train.py \
    --use_amp \
    --gpu_ids 0,1,2,3 \
    --batch_size 16 \
    --epochs 600 \
    --lr 0.0002
```

### 后台运行

```bash
nohup python train.py --use_amp --gpu_ids 0,1 > log/train_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

## 参数说明

- `--use_amp`: 启用自动混合精度训练
- `--gpu_ids`: 指定GPU ID，多个用逗号分隔 (默认: "0")
- `--batch_size`: 批次大小 (默认: 4)
- `--epochs`: 训练轮数 (默认: 600)
- `--lr`: 学习率 (默认: 0.0002)

## 显存使用建议

| 配置 | 单GPU显存 | 双GPU显存 | 推荐batch_size |
|------|-----------|-----------|----------------|
| 无AMP | 8-10GB | 4-5GB/GPU | 4 |
| AMP | 5-7GB | 3-4GB/GPU | 6-8 |

## 性能优化建议

1. **启用AMP**: 在现代GPU上建议总是启用AMP
2. **调整batch_size**: 根据显存大小调整，AMP下可以增大1.5-2倍
3. **多GPU**: 如果有多个GPU，建议使用多GPU训练
4. **数据加载**: 调整num_workers参数以优化数据加载速度

## 常见问题

1. **单GPU上使用多GPU预训练模型**: 代码已自动处理module.前缀问题
2. **显存不足**: 尝试启用AMP或减小batch_size
3. **训练速度慢**: 检查数据加载瓶颈，调整num_workers
4. **Jupyter运行**: 代码自动检测环境，无需额外配置

## 环境测试

运行测试脚本检查环境配置：
```bash
python test_setup.py
```
