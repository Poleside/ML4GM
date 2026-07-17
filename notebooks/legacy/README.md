# ML 分步 Notebook 说明

原 `ML.ipynb` 已按**每个方法单独一个 notebook** 拆分为 7 个文件。

## 文件列表

| 文件 | 内容 |
|------|------|
| `01_dependencies_and_data.ipynb` | 依赖、路径、数据加载与预处理；末尾保存 `models/preprocessed_data.pkl` |
| `02_rf.ipynb` | RF 模型训练与评估，输出 `models/rf_model.pkl` |
| `03_gbdt.ipynb` | GBDT 模型训练与评估，输出 `models/gbdt_model.pkl` |
| `04_ann.ipynb` | ANN 模型训练与评估，输出 `models/ann_model.pkl` |
| `05_bp.ipynb` | BP 模型训练与评估，输出 `models/bp_model.pt` 与 `models/bp_test_pred.pkl` |
| `06_lstm.ipynb` | LSTM 模型训练与评估，输出 `models/lstm_model.pt` 与 `models/lstm_test_pred.pkl` |
| `07_compare.ipynb` | 模型对比（RF/GBDT/ANN/BP/LSTM 预测曲线与 R2 对比图） |

## 运行顺序

1. **先运行** `01_dependencies_and_data.ipynb`（生成 `preprocessed_data.pkl`）。
2. **再运行** 以下任意顺序或并行：
   - `02_rf.ipynb`
   - `03_gbdt.ipynb`
   - `04_ann.ipynb`
   - `05_bp.ipynb`
   - `06_lstm.ipynb`
3. **最后运行** `07_compare.ipynb`（会读取上述 5 个模型的输出与 05、06 保存的测试集预测）。

## 备份

- 完整未拆分版本已备份为 `ML_full.ipynb`。
