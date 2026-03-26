"""Split ML.ipynb into 7 Jupyter notebooks: 01_data, 02_rf, 03_gbdt, 04_ann, 05_bp, 06_lstm, 07_compare."""
import json

def get_source(cell):
    s = cell.get("source", [])
    if isinstance(s, list):
        return "".join(s)
    return s

def make_cell(cell_type, source, execution_count=None):
    cell = {"cell_type": cell_type, "metadata": {}, "source": source if isinstance(source, list) else [source]}
    if cell_type == "code":
        cell["outputs"] = []
        cell["execution_count"] = execution_count
    return cell

def copy_cell(cell):
    c = {k: v for k, v in cell.items() if k in ("cell_type", "metadata", "source")}
    if c["cell_type"] == "code":
        c["outputs"] = []
        c["execution_count"] = None
    return c

# Shared load block for model notebooks (02-06): load preprocessed_data.pkl
def load_cell_sklearn(title, extra_imports=None):
    imports = [
        '"""' + title + ' Run 01_dependencies_and_data.ipynb first."""\n',
        "import os\n",
        "import dill\n",
        "import numpy as np\n",
        "from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
    ]
    if extra_imports:
        imports = [imports[0]] + extra_imports + imports[1:]
    return imports + [
        "\n",
        'root_dir = "C:\\\\ML4GM"\n',
        "outputs_dir = os.path.join(root_dir, \"models\")\n",
        "\n",
        "def load_pkl(filepath):\n",
        "    with open(filepath, \"rb\") as fr:\n",
        "        return dill.load(fr)\n",
        "\n",
        "def save_pkl(filepath, data):\n",
        "    with open(filepath, \"wb\") as fw:\n",
        "        dill.dump(data, fw)\n",
        "    print(f\"[{filepath}] data saving...\")\n",
        "\n",
        "_data = load_pkl(os.path.join(outputs_dir, \"preprocessed_data.pkl\"))\n",
        "X_train = _data[\"X_train\"]\n",
        "X_test = _data[\"X_test\"]\n",
        "y_train = _data[\"y_train\"]\n",
        "y_test = _data[\"y_test\"]\n",
        "X_scaler = _data[\"X_scaler\"]\n",
        "y_scaler = _data[\"y_scaler\"]\n",
        "feature_columns = _data[\"feature_columns\"]\n",
    ]

def load_cell_torch(title):
    return [
        '"""' + title + ' Run 01_dependencies_and_data.ipynb first."""\n',
        "import os\n",
        "import random\n",
        "import dill\n",
        "import numpy as np\n",
        "import torch\n",
        "from torch import nn\n",
        "from torch.utils.data import Dataset, DataLoader\n",
        "from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score\n",
        "from tqdm import tqdm\n",
        "import matplotlib.pyplot as plt\n",
        "\n",
        'root_dir = "C:\\\\ML4GM"\n',
        "outputs_dir = os.path.join(root_dir, \"models\")\n",
        "\n",
        "def load_pkl(filepath):\n",
        "    with open(filepath, \"rb\") as fr:\n",
        "        return dill.load(fr)\n",
        "\n",
        "def save_pkl(filepath, data):\n",
        "    with open(filepath, \"wb\") as fw:\n",
        "        dill.dump(data, fw)\n",
        "    print(f\"[{filepath}] data saving...\")\n",
        "\n",
        "_data = load_pkl(os.path.join(outputs_dir, \"preprocessed_data.pkl\"))\n",
        "X_train = _data[\"X_train\"]\n",
        "X_test = _data[\"X_test\"]\n",
        "y_train = _data[\"y_train\"]\n",
        "y_test = _data[\"y_test\"]\n",
        "X_scaler = _data[\"X_scaler\"]\n",
        "y_scaler = _data[\"y_scaler\"]\n",
        "feature_columns = _data[\"feature_columns\"]\n",
    ]

def main():
    with open("ML.ipynb", "r", encoding="utf-8") as f:
        nb = json.load(f)
    cells = nb["cells"]

    section_markers = [
        "## 模型训练及评估",
        "### RF模型训练及评估",
        "### GBDT模型训练及评估",
        "### ANN模型训练及评估",
        "### BP模型训练及评估",
        "### LSTM模型训练及评估",
        "### 模型对比",
    ]
    section_at = [None] * len(section_markers)
    for idx, cell in enumerate(cells):
        src = get_source(cell)
        for i, m in enumerate(section_markers):
            if m in src:
                section_at[i] = idx
                break

    # Part 1: data (before 模型训练)
    end_part1 = section_at[0] if section_at[0] is not None else len(cells)
    part1_cells = [copy_cell(c) for c in cells[:end_part1]]

    # Part 2: RF (RF .. before GBDT)
    start_rf, end_rf = section_at[1] or 0, section_at[2] if section_at[2] is not None else len(cells)
    part2_cells = [copy_cell(c) for c in cells[start_rf:end_rf]]

    # Part 3: GBDT (GBDT .. before ANN)
    start_gbdt, end_gbdt = section_at[2] or 0, section_at[3] if section_at[3] is not None else len(cells)
    part3_cells = [copy_cell(c) for c in cells[start_gbdt:end_gbdt]]

    # Part 4: ANN (ANN .. before BP)
    start_ann, end_ann = section_at[3] or 0, section_at[4] if section_at[4] is not None else len(cells)
    part4_cells = [copy_cell(c) for c in cells[start_ann:end_ann]]

    # Part 5: BP (BP .. before LSTM)
    start_bp, end_bp = section_at[4] or 0, section_at[5] if section_at[5] is not None else len(cells)
    part5_cells = [copy_cell(c) for c in cells[start_bp:end_bp]]

    # Part 6: LSTM (LSTM .. before 模型对比)
    start_lstm, end_lstm = section_at[5] or 0, section_at[6] if section_at[6] is not None else len(cells)
    part6_cells = [copy_cell(c) for c in cells[start_lstm:end_lstm]]

    # Part 7: 模型对比 (to end)
    start_compare = section_at[6] or 0
    part7_cells = [copy_cell(c) for c in cells[start_compare:]]

    # Part 1: append save cell
    save_cell_source = [
        "# --- Save preprocessed data for 02-07 ---\n",
        "_data = {\n",
        '    "X_train": X_train, "X_test": X_test,\n',
        '    "y_train": y_train, "y_test": y_test,\n',
        '    "X_scaler": X_scaler, "y_scaler": y_scaler,\n',
        '    "feature_columns": feature_columns,\n',
        "}\n",
        'save_pkl(os.path.join(outputs_dir, "preprocessed_data.pkl"), _data)\n',
        'print("Preprocessed data saved to", os.path.join(outputs_dir, "preprocessed_data.pkl"))'
    ]
    part1_cells.append(make_cell("code", save_cell_source))

    nb_template = {"nbformat": 4, "nbformat_minor": 5, "metadata": nb.get("metadata", {}), "cells": []}

    # ----- 01: Dependencies + Data -----
    nb1 = {**nb_template, "cells": part1_cells}
    with open("01_dependencies_and_data.ipynb", "w", encoding="utf-8") as f:
        json.dump(nb1, f, ensure_ascii=False, indent=1)
    print("Wrote 01_dependencies_and_data.ipynb")

    # ----- 02: RF -----
    load_rf = load_cell_sklearn("Part 2: RF.", ["from sklearn.ensemble import RandomForestRegressor\n"])
    nb2 = {**nb_template, "cells": [
        make_cell("markdown", "# Part 2: RF 模型训练及评估\n\n请先运行 `01_dependencies_and_data.ipynb`。"),
        make_cell("code", load_rf),
    ] + part2_cells}
    with open("02_rf.ipynb", "w", encoding="utf-8") as f:
        json.dump(nb2, f, ensure_ascii=False, indent=1)
    print("Wrote 02_rf.ipynb")

    # ----- 03: GBDT -----
    load_gbdt = load_cell_sklearn("Part 3: GBDT.", ["from sklearn.ensemble import GradientBoostingRegressor\n"])
    nb3 = {**nb_template, "cells": [
        make_cell("markdown", "# Part 3: GBDT 模型训练及评估\n\n请先运行 `01_dependencies_and_data.ipynb`。"),
        make_cell("code", load_gbdt),
    ] + part3_cells}
    with open("03_gbdt.ipynb", "w", encoding="utf-8") as f:
        json.dump(nb3, f, ensure_ascii=False, indent=1)
    print("Wrote 03_gbdt.ipynb")

    # ----- 04: ANN -----
    load_ann = load_cell_sklearn("Part 4: ANN.", ["from sklearn.neural_network import MLPRegressor\n"])
    nb4 = {**nb_template, "cells": [
        make_cell("markdown", "# Part 4: ANN 模型训练及评估\n\n请先运行 `01_dependencies_and_data.ipynb`。"),
        make_cell("code", load_ann),
    ] + part4_cells}
    with open("04_ann.ipynb", "w", encoding="utf-8") as f:
        json.dump(nb4, f, ensure_ascii=False, indent=1)
    print("Wrote 04_ann.ipynb")

    # 05 BP: append cell to save test predictions for 07
    save_bp_cell = [
        "# Save test predictions for 07_compare\n",
        'save_pkl(os.path.join(outputs_dir, "bp_test_pred.pkl"), y_test_pred_of_bp)\n',
    ]
    part5_with_save = part5_cells + [make_cell("code", save_bp_cell)]

    # ----- 05: BP -----
    nb5 = {**nb_template, "cells": [
        make_cell("markdown", "# Part 5: BP 模型训练及评估\n\n请先运行 `01_dependencies_and_data.ipynb`。"),
        make_cell("code", load_cell_torch("Part 5: BP.")),
    ] + part5_with_save}
    with open("05_bp.ipynb", "w", encoding="utf-8") as f:
        json.dump(nb5, f, ensure_ascii=False, indent=1)
    print("Wrote 05_bp.ipynb")

    # 06 LSTM: append cell to save test predictions for 07
    save_lstm_cell = [
        "# Save test predictions for 07_compare\n",
        'save_pkl(os.path.join(outputs_dir, "lstm_test_pred.pkl"), y_test_pred_of_lstm)\n',
    ]
    part6_with_save = part6_cells + [make_cell("code", save_lstm_cell)]

    # ----- 06: LSTM -----
    nb6 = {**nb_template, "cells": [
        make_cell("markdown", "# Part 6: LSTM 模型训练及评估\n\n请先运行 `01_dependencies_and_data.ipynb`。"),
        make_cell("code", load_cell_torch("Part 6: LSTM.")),
    ] + part6_with_save}
    with open("06_lstm.ipynb", "w", encoding="utf-8") as f:
        json.dump(nb6, f, ensure_ascii=False, indent=1)
    print("Wrote 06_lstm.ipynb")

    # ----- 07: 模型对比 (load all models and compute predictions) -----
    load_compare = [
        '"""Part 7: 模型对比. Run 01 first; run 02-06 to generate all models and predictions."""\n',
        "import os\n",
        "import dill\n",
        "import numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "from sklearn.metrics import r2_score\n",
        "\n",
        'root_dir = "C:\\\\ML4GM"\n',
        "outputs_dir = os.path.join(root_dir, \"models\")\n",
        "\n",
        "def load_pkl(filepath):\n",
        "    with open(filepath, \"rb\") as fr:\n",
        "        return dill.load(fr)\n",
        "\n",
        "_data = load_pkl(os.path.join(outputs_dir, \"preprocessed_data.pkl\"))\n",
        "X_train = _data[\"X_train\"]\n",
        "X_test = _data[\"X_test\"]\n",
        "y_train = _data[\"y_train\"]\n",
        "y_test = _data[\"y_test\"]\n",
        "X_scaler = _data[\"X_scaler\"]\n",
        "y_scaler = _data[\"y_scaler\"]\n",
        "\n",
        "# Common y_test in original scale (same for all models)\n",
        "y_test_real = y_scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(-1)\n",
        "y_test_of_rf = y_test_of_gbdt = y_test_of_ann = y_test_of_bp = y_test_of_lstm = y_test_real.tolist()\n",
        "\n",
        "# Load RF, GBDT, ANN and get test predictions\n",
        "rf_model = load_pkl(os.path.join(outputs_dir, \"rf_model.pkl\"))\n",
        "gbdt_model = load_pkl(os.path.join(outputs_dir, \"gbdt_model.pkl\"))\n",
        "ann_model = load_pkl(os.path.join(outputs_dir, \"ann_model.pkl\"))\n",
        "y_test_pred_of_rf = y_scaler.inverse_transform(rf_model.predict(X_test).reshape(-1, 1)).reshape(-1).tolist()\n",
        "y_test_pred_of_gbdt = y_scaler.inverse_transform(gbdt_model.predict(X_test).reshape(-1, 1)).reshape(-1).tolist()\n",
        "y_test_pred_of_ann = y_scaler.inverse_transform(ann_model.predict(X_test).reshape(-1, 1)).reshape(-1).tolist()\n",
        "\n",
        "# Load BP and LSTM test predictions (saved by 05 and 06)\n",
        "y_test_pred_of_bp = load_pkl(os.path.join(outputs_dir, \"bp_test_pred.pkl\"))\n",
        "y_test_pred_of_lstm = load_pkl(os.path.join(outputs_dir, \"lstm_test_pred.pkl\"))\n",
    ]
    nb7 = {**nb_template, "cells": [
        make_cell("markdown", "# Part 7: 模型对比\n\n请先运行 `01_dependencies_and_data.ipynb`，并完成 `02_rf`、`03_gbdt`、`04_ann`、`05_bp`、`06_lstm` 以生成各模型与预测结果。"),
        make_cell("code", load_compare),
    ] + part7_cells}
    with open("07_compare.ipynb", "w", encoding="utf-8") as f:
        json.dump(nb7, f, ensure_ascii=False, indent=1)
    print("Wrote 07_compare.ipynb")

    # Remove old combined DL notebook if present
    import os as _os
    if _os.path.isfile("04_dl_and_compare.ipynb"):
        _os.remove("04_dl_and_compare.ipynb")
        print("Removed 04_dl_and_compare.ipynb")

    print("Section indices:", section_at)

if __name__ == "__main__":
    main()
