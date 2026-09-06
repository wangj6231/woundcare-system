# ============================================================
# audit_duplicates.py — Phase 6.5-B：Duplicate Group Audit
#
# 目標：對 206 個重複 MD5 hash 做完整展開
#   ① 每個 hash group 含哪些檔案、哪些 class
#   ② 模擬 5-Fold CV 分配，每張圖在哪個 Fold 的 Train or Val
#   ③ 判斷是否有「同一 hash group 同時出現在 Train 和 Val」
#   ④ 生成 JSON 報告 + 論文用結論
#
# PASS 條件：
#   所有 duplicate group 內，圖片都集中在「同一 Fold 的同一角色（Train/Val）」
#   → 不影響 K-Fold 結果的有效性
#
# FAIL 條件：
#   任何 duplicate group 出現 Train + Val 跨角色分配
#   → Cross-fold content leakage，98.34% 結果有偏差
#
# 使用方式：
#   python experiments/scripts/audit_duplicates.py
# ============================================================
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

BASE_DIR    = Path(__file__).parent.parent.parent
DATASET     = BASE_DIR / 'yolo_wound_cls_dataset_v3'
STATS_DIR   = Path(__file__).parent.parent / 'results' / 'statistics'
CLASS_NAMES = ['Abrasions', 'Bruises', 'Burns', 'Cut',
               'Ingrown_nails', 'Laceration', 'Stab_wound']
DEV_SPLITS  = ['train', 'val']
SEED = 42
K    = 5


def file_md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


def collect_dev_set():
    """回傳 [(Path, cls_idx, split_name), ...]"""
    items = []
    for split in DEV_SPLITS:
        for cls_idx, cls_name in enumerate(CLASS_NAMES):
            cls_dir = DATASET / split / cls_name
            if not cls_dir.exists():
                continue
            for img in list(cls_dir.glob('*.[jp][pn]g')) + list(cls_dir.glob('*.jpeg')):
                items.append((img, cls_idx, split))
    return items


def simulate_kfold(items):
    """
    模擬 StratifiedKFold，回傳 {img_path_str: {fold: 'train'/'val'}}
    """
    all_paths  = [str(p) for p, _, _ in items]
    all_labels = [lbl for _, lbl, _ in items]
    all_p_obj  = [p for p, _, _ in items]

    np_paths  = np.array(all_paths)
    np_labels = np.array(all_labels)

    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=SEED)
    assignment = defaultdict(dict)  # path_str → {fold_idx: 'train'/'val'}

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(np_paths, np_labels), 1):
        for i in train_idx:
            assignment[all_paths[i]][fold_idx] = 'train'
        for i in val_idx:
            assignment[all_paths[i]][fold_idx] = 'val'

    return assignment


def run_duplicate_group_audit():
    print(f"\n{'='*62}")
    print(f"  Phase 6.5-B — Duplicate Group Audit")
    print(f"  Dataset: {DATASET}")
    print(f"{'='*62}")

    # ── Step 1: 計算所有 MD5 ──────────────────────────────────
    print(f"\n  Collecting development set images...")
    items = collect_dev_set()
    total = len(items)
    print(f"  Total: {total} images  |  Computing MD5...")

    hash_to_files = defaultdict(list)
    for img_path, cls_idx, split in items:
        md5 = file_md5(img_path)
        hash_to_files[md5].append({
            'path':      str(img_path),
            'filename':  img_path.name,
            'class':     CLASS_NAMES[cls_idx],
            'cls_idx':   cls_idx,
            'orig_split': split,
        })

    all_hashes       = len(hash_to_files)
    duplicate_groups = {h: files for h, files in hash_to_files.items() if len(files) > 1}
    unique_groups    = {h: files for h, files in hash_to_files.items() if len(files) == 1}

    total_dup_files = sum(len(v) for v in duplicate_groups.values())

    print(f"  Total unique hashes:      {all_hashes}")
    print(f"  Unique (no duplicate):    {len(unique_groups)}")
    print(f"  Duplicate groups:         {len(duplicate_groups)}")
    print(f"  Files in duplicate groups:{total_dup_files}")

    # ── Step 2: 模擬 K-Fold 分配 ─────────────────────────────
    print(f"\n  Simulating {K}-Fold assignments (seed={SEED})...")
    assignment = simulate_kfold(items)

    # ── Step 3: 分析每個 duplicate group ─────────────────────
    print(f"\n{'─'*62}")
    print(f"  Checking {len(duplicate_groups)} duplicate groups for cross-fold leakage...")
    print(f"{'─'*62}")

    leakage_groups   = []   # 有跨 Train/Val 的 group
    clean_groups     = []   # 無 leakage 的 group
    class_dup_count  = defaultdict(int)

    for md5_hash, files in duplicate_groups.items():
        # 記錄每個 class 的 duplicate 數量
        for f in files:
            class_dup_count[f['class']] += 1

        # 取得每張圖的 Fold 分配
        group_assignments = []
        for f in files:
            path_str = f['path']
            folds = assignment.get(path_str, {})
            for fold_idx, role in folds.items():
                group_assignments.append({
                    'file':     f['filename'],
                    'class':    f['class'],
                    'fold':     fold_idx,
                    'role':     role,
                    'path':     path_str,
                })

        # 檢查：同一 fold 下是否有 Train + Val 同時存在
        has_leakage = False
        leakage_details = []
        for fold_idx in range(1, K+1):
            roles_in_fold = {a['role'] for a in group_assignments if a['fold'] == fold_idx}
            if 'train' in roles_in_fold and 'val' in roles_in_fold:
                has_leakage = True
                leakage_details.append({
                    'fold':   fold_idx,
                    'files':  [a['file'] for a in group_assignments if a['fold'] == fold_idx],
                    'roles':  list(roles_in_fold),
                })

        entry = {
            'md5':        md5_hash[:12] + '...',
            'num_files':  len(files),
            'classes':    list({f['class'] for f in files}),
            'filenames':  [f['filename'] for f in files],
            'leakage':    has_leakage,
            'leakage_details': leakage_details,
            'assignments': group_assignments,
        }

        if has_leakage:
            leakage_groups.append(entry)
        else:
            clean_groups.append(entry)

    # ── Step 4: 印出結果 ──────────────────────────────────────
    print(f"\n  Duplicate groups with leakage:   {len(leakage_groups)}")
    print(f"  Duplicate groups clean (no leak): {len(clean_groups)}")

    if leakage_groups:
        print(f"\n  {'='*58}")
        print(f"  ❌  CROSS-FOLD CONTENT LEAKAGE DETECTED")
        print(f"  {'='*58}")
        print(f"\n  Top leakage groups (first 10):")
        for g in leakage_groups[:10]:
            print(f"\n  Hash: {g['md5']}")
            print(f"  Files ({g['num_files']}): {g['filenames']}")
            print(f"  Classes: {g['classes']}")
            for ld in g['leakage_details']:
                print(f"  → Fold {ld['fold']}: {ld['roles']} — same content in Train AND Val")
    else:
        print(f"\n  {'='*58}")
        print(f"  ✅  NO CROSS-FOLD LEAKAGE DETECTED")
        print(f"  {'='*58}")
        print(f"\n  All {len(duplicate_groups)} duplicate groups have their copies")
        print(f"  confined to the same role (either Train or Val) within each fold.")
        print(f"  The K-Fold CV results (98.34 ± 1.21%) are clean.")

    # ── Step 5: Class-level summary ──────────────────────────
    print(f"\n  Duplicate image count by class:")
    for cls_name in CLASS_NAMES:
        cnt = class_dup_count.get(cls_name, 0)
        bar = '█' * (cnt // 5 + 1) if cnt > 0 else ''
        print(f"    {cls_name:<22}: {cnt:>4}  {bar}")

    # ── Step 6: 說明 duplicate 的來源 ────────────────────────
    print(f"""
  ─────────────────────────────────────────────────────────
  ℹ️  Why are there {len(duplicate_groups)} duplicate groups?
  
  Likely sources:
  ① Oversampling strategy: rare classes (e.g., Stab_wound)
    may have been duplicated to balance class distribution.
  ② Web-scraping artifacts: same image downloaded twice
    under different filenames.
  ③ Dataset curation: different sources contained the
    same image.

  As long as ALL copies of the same image stay in the same
  Fold's Train set (or all in Val), there is NO leakage.
  The concern is only when copies end up in BOTH Train and
  Val of the same fold.
  ─────────────────────────────────────────────────────────""")

    # ── Step 7: 儲存 JSON ─────────────────────────────────────
    STATS_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        'total_images':           total,
        'total_unique_hashes':    all_hashes,
        'duplicate_groups':       len(duplicate_groups),
        'files_in_dup_groups':    total_dup_files,
        'leakage_group_count':    len(leakage_groups),
        'clean_group_count':      len(clean_groups),
        'cross_fold_leakage':     len(leakage_groups) > 0,
        'verdict':                'FAIL' if leakage_groups else 'PASS',
        'k_folds':                K,
        'seed':                   SEED,
        'class_duplicate_counts': dict(class_dup_count),
        'leakage_groups':         leakage_groups[:20],  # 只存前 20 筆避免檔案過大
    }
    out_path = STATS_DIR / 'phase65b_duplicate_group_audit.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # ── Step 8: 最終裁定 ─────────────────────────────────────
    print(f"\n{'='*62}")
    if len(leakage_groups) == 0:
        print(f"  ✅  PHASE 6.5-B VERDICT: PASS")
        print(f"  → 98.34 ± 1.21% can be confirmed as clean K-Fold CV result")
        print(f"  → Table 2 can use this as the official C-Arch-05 result")
        print(f"  → Proceed to Phase 7: Multi-Seed")
    else:
        print(f"  ❌  PHASE 6.5-B VERDICT: FAIL")
        print(f"  → {len(leakage_groups)} duplicate groups cross Train/Val boundary")
        print(f"  → Must use GroupKFold or deduplicate before re-running CV")
        print(f"  → 98.34% is PENDING — do not report in Table 2 yet")
    print(f"  💾 Report: {out_path}")
    print('='*62)

    return report


if __name__ == '__main__':
    run_duplicate_group_audit()
