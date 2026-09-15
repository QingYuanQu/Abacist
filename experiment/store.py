"""store.py — 学习单元三表 CSV 读写（无 schema，原始字符串行）。

表结构（按概念域拆分，id 关联；三表均冗余 name 列便于人读）：
    material.csv  学什么（Material 字段）
    method.csv    怎么学（Method 字段）
    report.csv    学得怎么样（最优 Record 全字段 + passed）
"""

import csv
import os
import tempfile

from config import RECORD_FIELDS

MATERIAL_COLS = [
    "id", "name", "type", "ops", "repeat", "start", "end",
    "allow_negative", "input_format", "parse", "eval", "sample",
    "forbidden_patterns", "split", "abacus_bead", "abacus_order", "abacus_snapshot",
    "abacus_style", "env_steps", "source", "variant",
]
METHOD_COLS = ["id", "name", "repeat_factor", "epochs", "batch_size", "learning_rate", "shuffle"]
REPORT_COLS = ["id", "name"] + RECORD_FIELDS + ["passed"]


class _CsvTable:
    """CSV 原子读写基类（原始字符串行，类型转换交给调用方）。"""

    def __init__(self, path: str):
        self.path = path

    def all(self) -> list[dict]:
        """返回原始字符串行；文件不存在返回空列表。"""
        if not os.path.isfile(self.path):
            return []
        with open(self.path, "r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))

    def _save(self, fieldnames: list[str], rows: list[dict]):
        """原子写入：临时文件 + os.replace。"""
        dirname = os.path.dirname(self.path) or "."
        fd, tmp = tempfile.mkstemp(dir=dirname, suffix=".csv")
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                w.writeheader()
                w.writerows(rows)
            os.replace(tmp, self.path)
        except Exception:
            if os.path.isfile(tmp):
                os.remove(tmp)
            raise

    def write_header(self, fieldnames: list[str]):
        """写入仅表头的空表。"""
        self._save(fieldnames, [])


class MaterialTable(_CsvTable):
    """material.csv —— 学什么。"""


class MethodTable(_CsvTable):
    """method.csv —— 怎么学。"""


class ReportTable(_CsvTable):
    """report.csv —— 最优 Record + passed。"""

    def save_result(self, trial_id: int, trial_name: str, record, passed: bool):
        """写回单课最优 Record 及通过判定（无则追加行）。"""
        row = {"id": str(trial_id), "name": trial_name,
               **record.to_csv_row(), "passed": "1" if passed else "0"}
        rows = self.all()
        for r in rows:
            if r["id"].strip() == str(trial_id):
                r.update(row)
                break
        else:
            rows.append(row)
        self._save(REPORT_COLS, rows)

    def clear_results(self, trial_ids=None):
        """清空结果列（保留 id/name 行）。trial_ids=None 表示全部。返回清除行数。"""
        keys = set(map(str, trial_ids)) if trial_ids else None
        keep = {"id", "name"}
        rows = self.all()
        count = 0
        for r in rows:
            if keys is None or r["id"].strip() in keys:
                for c in REPORT_COLS:
                    if c not in keep and c in r:
                        r[c] = ""
                count += 1
        if count:
            self._save(REPORT_COLS, rows)
        return count
