"""构造式生成器 gen_values 的正确性与性能自检。

1. 正确性：gen_values 成功的样本必须通过 check_tree，叶值全在 1..9；
2. 速度：每档 300 棵随机树，统计单棵最大耗时（旧版在 n>=13 的 ÷ 密集树上
   最坏可达小时级，是 2026-08-29 整夜卡死的根因）；
3. 最坏情况：全 ÷ 树（昨夜卡死的同款对象），应当毫秒级出结果（多数不可行）。
"""
import random
import sys
import time

sys.path.insert(0, r'd:\Abacist\parse')
import dataset_generator as dg

rng = random.Random(7)

print("== correctness & speed: 300 random trees per n ==")
bad = 0
t_all = 0.0
for n in range(2, 21):
    ok = fail = 0
    tmax = 0.0
    for _ in range(300):
        ts = dg.sample_tree(n, rng)
        ops = [rng.choice(dg.OPS) for _ in range(n - 1)]
        tree = dg.assign_ops(dg.assign_leaves(ts), ops)
        t0 = time.perf_counter()
        vals = dg.gen_values(tree, rng)
        tmax = max(tmax, time.perf_counter() - t0)
        if vals is None:
            fail += 1
            continue
        ok += 1
        if not dg.check_tree(tree, vals):
            bad += 1
            print(f"  VIOLATION(check_tree): n={n}")
        if not all(1 <= v <= 9 for v in vals.values()):
            bad += 1
            print(f"  VIOLATION(leaf range): n={n}")
    t_all += tmax
    print(f"  n={n:2d}: ok={ok:3d} infeasible={fail:3d} max_time={tmax*1000:7.2f}ms")
print(f"  violations: {bad} (must be 0)")

print("\n== worst case: all-division trees ==")
if dg.USE_DIV:
    for n in [10, 14, 18, 20]:
        ts = dg.sample_tree(n, rng)
        tree = dg.assign_ops(dg.assign_leaves(ts), ['÷'] * (n - 1))
        t0 = time.perf_counter()
        r = dg.gen_values(tree, rng)
        dt = (time.perf_counter() - t0) * 1000
        print(f"  n={n:2d} all-÷: {dt:8.2f}ms -> {'OK' if r else 'INFEASIBLE (fast)'}")
else:
    print("  ÷ 已禁用（USE_DIV=False），跳过；+−× 下该最坏情况不存在。")
