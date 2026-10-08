# 3D 开奖数据机器学习实验系统

本项目针对三位数字开奖历史数据，采用严格的时间序列方法：
- 数据质量检查
- 无未来数据泄漏的特征工程
- Random / Rolling Frequency 基线
- LightGBM
- XGBoost
- CatBoost
- LSTM（数据量足够时启用）
- Walk-Forward 回测
- LogLoss / Brier / Top-K / 完整号码命中
- 验证集概率融合
- 概率校准
- SHAP（可用时）
- 最终预测

> 重要：本项目是统计/机器学习实验，不保证也不能证明能够稳定预测随机开奖结果。最终以严格盲测结果为准。

## 一、Windows 11 安装

建议安装 Miniconda，然后打开 Anaconda Prompt：

```bat
conda create -n lottery3d python=3.11 -y
conda activate lottery3d
cd /d D:\3d_prediction
pip install -r requirements.txt
```

如果使用 NVIDIA GPU，可以根据你的 CUDA 版本单独安装 PyTorch 官方对应版本；第一版即使没有 GPU 也可以运行。

## 二、目录

```text
D:\3d_prediction\
├─ data\
│  ├─ raw\
│  │  └─ 3d_history.csv
│  └─ processed\
├─ models\
├─ results\
│  ├─ backtest\
│  ├─ predictions\
│  └─ reports\
├─ src\
│  ├─ __init__.py
│  ├─ config.py
│  ├─ data_loader.py
│  ├─ validator.py
│  ├─ features.py
│  ├─ baselines.py
│  ├─ tabular_models.py
│  ├─ lstm_model.py
│  ├─ walk_forward.py
│  ├─ calibration.py
│  ├─ ensemble.py
│  ├─ evaluation.py
│  ├─ reporting.py
│  └─ predict.py
├─ config\
│  └─ config.yaml
├─ main.py
├─ predict.py
├─ requirements.txt
└─ run.bat
```

## 三、你的数据格式

当前提供的文件已经确认是：

```text
期号,开奖日期,开奖号码
2026001,2026/1/1,123
2026002,2026/1/2,507
```

因此程序已经按照这三个中文字段兼容。

把完整数据最终放到：

```text
D:\3d_prediction\data\raw\3d_history.csv
```

如果以后追加数据：
1. 仍然只维护这个原始 CSV；
2. 保留同样的三列；
3. 不要手工修改 processed、models、results；
4. 重新运行 `python main.py`。

程序会重新从原始数据建立特征和回测。

## 四、第一次运行

当前只有 2 期数据，无法进行有意义的机器学习训练。程序会自动完成数据检查，并明确提示数据不足，而不会假装训练出模型。

数据达到配置文件中的最低数量后：

```bat
conda activate lottery3d
cd /d D:\3d_prediction
python main.py
```

## 五、正式数据量建议

代码不会人为要求某个固定“中奖预测”样本量，但为了让滚动特征、Walk-Forward、模型比较有统计意义，建议尽可能提供完整、连续的历史数据。

特别是 LSTM 对数据量要求更高；数据不足时自动跳过，不影响 LightGBM/XGBoost/CatBoost 主流程。

## 六、预测

训练完成后：

```bat
python predict.py
```

预测结果会写入：

```text
results\predictions\
```

## 七、关键原则

1. 所有特征只能使用预测期之前的数据。
2. 禁止随机 train_test_split。
3. 测试集不参与调参。
4. 不人为“杀号”。
5. 不因某一期预测失败临时修改规则。
6. 任何模型都必须和随机/频率基线比较。
7. 如果盲测不能稳定超过基线，就报告“没有发现稳定优势”。
