import os, sys, json, time, argparse, threading
import concurrent.futures
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))                     # BIRD/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))    # 项目根（chat.py 所在）

from schema_linking import SchemaLinking
import chat as chat_mod


def load_json(p, default):
    if os.path.exists(p):
        try:
            return json.load(open(p, encoding="utf-8"))
        except Exception:
            return default
    return default


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_json", required=True)
    ap.add_argument("--db_root", required=True)      # 本地库根目录
    ap.add_argument("--work_dir", required=True)     # 本地工作目录（cache + 分片 + 结果）
    ap.add_argument("--out_dir", required=True)      # persistent results directory
    ap.add_argument("--model", default="glm-5.2")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--max_rounds", type=int, default=500)   # 安全阀；限流会持续重跑到消化，通常用不到此上限
    ap.add_argument("--limit", type=int, default=0)   # >0 时只处理前 N 条（验证用）；0=全量
    a = ap.parse_args()

    cache_dir = os.path.join(a.work_dir, "cache", a.model)
    parts_dir = os.path.join(a.work_dir, "parts")
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(parts_dir, exist_ok=True)
    os.makedirs(a.out_dir, exist_ok=True)

    results_local = os.path.join(a.work_dir, f"results_{a.model}.json")
    results_out   = os.path.join(a.out_dir,  f"results_{a.model}.json")
    fail_out      = os.path.join(a.out_dir,  "failures.json")

    data = json.load(open(a.data_json, encoding="utf-8"))
    if a.limit and a.limit > 0:
        data = data[:a.limit]
        print(f"[limit] 验证模式：仅处理前 {len(data)} 条样本")
    lock = threading.Lock()

    # resume: local results first, then previous shared results, then per-question shards
    results = load_json(results_local, {})
    for k, v in load_json(results_out, {}).items():
        results.setdefault(k, v)
    for fn in os.listdir(parts_dir):
        if fn.endswith(".json"):
            results.setdefault(fn[:-5], load_json(os.path.join(parts_dir, fn), None))
    results = {k: v for k, v in results.items() if v}

    def flush(to_shared=True):
        with lock:
            snapshot = dict(results)
        json.dump(snapshot, open(results_local, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        if to_shared:
            json.dump(snapshot, open(results_out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    def worker(case):
        qid = str(case["question_id"]); db_id = case["db_id"]
        sqlite_path = os.path.join(a.db_root, db_id, f"{db_id}.sqlite")
        if not os.path.exists(sqlite_path):
            return ("err", qid, f"db not found: {sqlite_path}")
        try:
            linker = SchemaLinking(model=a.model, cache_dir=cache_dir)
            res = linker.run_schema_linking(
                case["question"], qid, evidence=case.get("evidence", ""),
                sqlite_path=sqlite_path, database_root_dir=a.db_root, enable_pruning=True)
            out = {"question_id": qid, "question": case["question"], "db_id": db_id,
                   "evidence": case.get("evidence", ""), "result": res}
            json.dump(out, open(os.path.join(parts_dir, f"{qid}.json"), "w", encoding="utf-8"),
                      ensure_ascii=False)                       # 分片落盘（崩溃安全，无需大锁）
            with lock:
                results[qid] = out
            return ("ok", qid, None)
        except Exception as e:
            return ("err", qid, f"{type(e).__name__}: {e}")

    pending = [c for c in data if str(c["question_id"]) not in results]
    print(f"[init] total={len(data)} done={len(results)} pending={len(pending)}")

    all_fail = {}
    HARD_FAIL_ROUNDS = 5      # 连续「无进展且无限流」达到此轮数 → 判定为硬错误，停止
    no_progress = 0
    rnd = 0
    while pending:
        rnd += 1
        print(f"\n===== ROUND {rnd} : {len(pending)} cases, workers={a.workers} =====")
        rl_before = chat_mod.RATE_LIMIT_COUNTER
        n_before = len(pending)
        failed = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as ex:
            futs = {ex.submit(worker, c): c for c in pending}
            cnt = 0
            for f in tqdm(concurrent.futures.as_completed(futs), total=len(futs)):
                st, qid, err = f.result()
                if st == "err":
                    failed.append(futs[f]); all_fail[qid] = err
                else:
                    all_fail.pop(qid, None)
                cnt += 1
                if cnt % 50 == 0:
                    flush()
                    print(f"[progress] round={rnd} done={cnt}/{len(futs)} "
                          f"ok_total={len(results)} rl_hits_total={chat_mod.RATE_LIMIT_COUNTER}", flush=True)
        flush()
        rl = [q for q, e in all_fail.items()
              if "##RATELIMIT##" in e or "429" in e or "rate" in e.lower() or "throttl" in e.lower() or "限流" in e]
        json.dump({"round": rnd, "failed": all_fail, "rate_limit_ids": rl,
                   "rate_limit_hits_total": chat_mod.RATE_LIMIT_COUNTER,
                   "rate_limit_hits_this_round": chat_mod.RATE_LIMIT_COUNTER - rl_before},
                  open(fail_out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"[round {rnd}] ok_total={len(results)} failed={len(failed)} rate_limit={len(rl)} "
              f"rl_hits_total={chat_mod.RATE_LIMIT_COUNTER}")
        pending = failed
        if not pending:
            break                                     # 全部成功，结束
        made_progress = len(pending) < n_before
        if len(rl) > 0 or made_progress:
            # 仍有限流 或 本轮有进展 → 继续重跑（限流一定会被消化）；限流时多退避
            no_progress = 0
            time.sleep(30 if len(rl) > 0 else 5)
        else:
            # 无进展且无任何限流 → 疑似硬错误（重跑也不会好）
            no_progress += 1
            print(f"[warn] round {rnd}: no progress & no rate-limit "
                  f"({no_progress}/{HARD_FAIL_ROUNDS}); {len(pending)} non-ratelimit failures.")
            if no_progress >= HARD_FAIL_ROUNDS:
                print(f"[stop] {len(pending)} hard-error cases can't be fixed by retrying; see failures.json.")
                break
            time.sleep(5)
        if rnd >= a.max_rounds:                        # 安全阀，防止极端死循环
            print(f"[stop] reached max_rounds={a.max_rounds}; {len(pending)} still pending. "
                  f"Rerun the SAME command to resume (断点续跑).")
            break

    flush()
    print(f"\n[done] results={len(results)}/{len(data)} -> {results_out}")
    print(f"[rate-limit total hits] {chat_mod.RATE_LIMIT_COUNTER}")


if __name__ == "__main__":
    main()
