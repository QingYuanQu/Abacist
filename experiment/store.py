"""store.py — 实验结果表（report.csv）读写。

配置只有两个来源：`config.yaml`（唯一输入，见 experiment/schema.py）和磁盘产物。
本模块只负责**结果**这一侧：每行 = 一个 trial 的最优 Record + 判定口径 + passed。

写约定（路线 B / 单写方）：report.csv 只由"完整 train+eval"运行写出（runner 路径），
每个 trial 一行、重复 save 整行覆盖，无非空合并。独立评估（eval.py）只打印、不持久化，
避免把配置改后的 what-if 重评分覆盖进正式结果表。CSV 便于人直接观察（设计约定）。
"""

import csv
import os

from config import RECORD_FIELDS, Record

# id/name 是身份，acc_mode/pass_threshold 让每行自带判定口径
# （旧格式只有 passed 而无口径，同一文件里 0.15 判过、0.26 判不过，无法自证）
REPORT_COLS = ["id", "name"] + RECORD_FIELDS + ["acc_mode", "pass_threshold", "passed"]


class ReportMismatch(Exception):
    """report.csv 与当前 config.yaml 不一致 —— 旧结果不可复用。"""


class ReportTable:
    """report.csv —— 每个 trial 的最优 Record + 判定口径 + passed。"""

    def __init__(self, path: str):
        self.path = path

    # ---------- 读 ----------

    def all(self) -> list[dict]:
        """返回原始字符串行；文件不存在返回空列表。"""
        if not os.path.isfile(self.path):
            return []
        with open(self.path, "r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))

    def load(self, names: list[str]) -> dict[int, tuple[Record | None, bool | None]]:
        """按 id 读取每个 trial 的结果，并校验身份指纹。

        返回 {trial_id: (record, passed)}；缺行/空行的 trial 不出现在结果里。
        """
        out: dict[int, tuple[Record | None, bool | None]] = {}
        for row in self.all():
            raw_id = (row.get("id") or "").strip()
            if not raw_id:
                continue
            try:
                tid = int(raw_id)
            except ValueError:
                raise ReportMismatch(f"{self.path}: id={raw_id!r} 不是整数") from None
            if not 0 <= tid < len(names):
                raise ReportMismatch(
                    f"{self.path}: 存在 id={tid} 的行，但当前只有 {len(names)} 个 trial")
            row_name = (row.get("name") or "").strip()
            if row_name != names[tid]:
                raise ReportMismatch(
                    f"{self.path}: trial id={tid} 的 name={row_name!r} 与配置 {names[tid]!r} 不一致；"
                    f"trials 已被重排/改名/增删，请用 --reset-eval 清理结果后重跑")
            passed_s = (row.get("passed") or "").strip()
            out[tid] = (Record.from_csv_row(row), (passed_s == "1") if passed_s else None)
        return out

    # ---------- 写 ----------

    def save_result(self, trial_id: int, trial_name: str, record: Record | None,
                    passed: bool, acc_mode: str, pass_threshold: float) -> None:
        """写回单 trial 最优 Record 及判定口径（无则追加行）。整行覆盖。"""
        row = {"id": str(trial_id), "name": trial_name,
               **(record.to_csv_row() if record is not None else {}),
               "acc_mode": acc_mode,
               "pass_threshold": f"{pass_threshold:g}",
               "passed": "1" if passed else "0"}
        rows = self.all()
        for r in rows:
            if (r.get("id") or "").strip() == str(trial_id):
                r.update(row)          # 整行覆盖：只由完整 train+eval 运行写出
                break
        else:
            rows.append(row)
        self._write(rows)

    def clear_results(self, trial_ids=None) -> int:
        """清空结果列（保留 id/name 行）。trial_ids=None 表示全部。返回清除行数。"""
        keys = set(map(str, trial_ids)) if trial_ids else None
        rows = self.all()
        count = 0
        for r in rows:
            if keys is None or (r.get("id") or "").strip() in keys:
                for c in REPORT_COLS:
                    if c not in ("id", "name"):
                        r[c] = ""
                count += 1
        if count:
            self._write(rows)
        return count

    def write_header(self) -> None:
        """写入仅表头的空表。"""
        self._write([])

    def _write(self, rows: list[dict]) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=REPORT_COLS, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
