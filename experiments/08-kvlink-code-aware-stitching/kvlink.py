"""Code-aware selective recompute for stitched per-file KV caches.

Question: when file B's KV cache was computed without file A in view and is stitched after A's cache, which of B's
tokens should be recomputed to recover the full-prefill result? CacheBlend picks tokens by KV deviation. Here we
test picking them from the code itself: the lines of B that reference symbols defined in A.

Everything comes out of one forward pass with a custom attention mask (no cache plumbing):
  A        attends causally within A                                -> A's cache (exact, it comes first)
  B        attends causally within B only                           -> B's chunk-alone cache, at B's final positions
  S'       fresh copies of the selected B tokens, attending to A, the unselected B tokens before them, and the
           S' copies before them                                    -> recomputed KV (CacheBlend semantics)
  P        the probe, the next tokens of B, attending to A, unselected B, S' and P causally
Ground truth is a plain causal pass over A + B + P. Metric: KL(truth || stitched) over P's next-token distributions.
"""

import argparse
import ast
import bisect
import json
import pathlib
import random
import subprocess
import sys
import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

NEG = torch.finfo(torch.float32).min


def module_path(repo, b_rel, node):
    """Resolve an ImportFrom in file b_rel to a repo-relative .py path, or None."""
    if node.level == 0:
        if not node.module:
            return None
        parts = node.module.split(".")
        base = pathlib.Path()
    else:
        base = pathlib.Path(b_rel).parent
        for _ in range(node.level - 1):
            base = base.parent
        parts = node.module.split(".") if node.module else []
    for cand in (base.joinpath(*parts).with_suffix(".py"), base.joinpath(*parts, "__init__.py")):
        if parts and (repo / cand).is_file():
            return str(cand)
    return None


def find_pairs(repos):
    pairs = []
    for repo in map(pathlib.Path, repos):
        files = subprocess.run(["git", "-C", str(repo), "ls-files", "*.py"], capture_output=True, text=True).stdout.split()
        for b in files:
            try:
                tree = ast.parse((repo / b).read_text())
            except Exception:
                continue
            for n in ast.walk(tree):
                if isinstance(n, ast.ImportFrom):
                    a = module_path(repo, b, n)
                    names = [x.asname or x.name for x in n.names if x.name != "*"]
                    if a and a != b and names and not a.endswith("__init__.py"):
                        pairs.append((str(repo), a, b, names))
    return pairs


def ref_lines(b_src, names):
    """1-based lines of B that use any imported name (the import statement itself excluded)."""
    names = set(names)
    lines = set()
    for n in ast.walk(ast.parse(b_src)):
        if isinstance(n, ast.Name) and n.id in names:
            lines.add(n.lineno)
        elif isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id in names:
            lines.add(n.lineno)
    return lines


def ref_tokens(b_src, names, offsets, head_len):
    """Token indices (within B) covering the identifier occurrences of the imported names."""
    names = set(names)
    line_starts = [0]
    for line in b_src.splitlines(keepends=True):
        line_starts.append(line_starts[-1] + len(line))
    spans = []
    for n in ast.walk(ast.parse(b_src)):
        if isinstance(n, ast.Name) and n.id in names:
            s = line_starts[n.lineno - 1] + n.col_offset
            spans.append((s, s + len(n.id)))
    sel = set()
    for i, (s, e) in enumerate(offsets):
        s, e = s - head_len, e - head_len
        if any(s < se and e > ss for ss, se in spans):
            sel.add(i)
    return sel


@torch.no_grad()
def logits_for(model, ids, pos, mask, keep):
    # only the last `keep` rows: full logits are n x 152k floats, about 0.6 GB per 1k tokens
    out = model(input_ids=ids[None], position_ids=pos[None], attention_mask=mask[None, None], logits_to_keep=keep)
    return out.logits[0].float()


def stitched_mask(nA, nB, sel, nP):
    """Attention mask for [A | B | S' | P] as described in the module docstring."""
    sel = sorted(sel)
    nS = len(sel)
    n = nA + nB + nS + nP
    allow = torch.zeros(n, n, dtype=torch.bool)
    # positions in the original sequence, used for causality
    posB = torch.arange(nA, nA + nB)
    posS = torch.tensor([nA + i for i in sel], dtype=torch.long)
    is_sel = torch.zeros(nB, dtype=torch.bool)
    is_sel[sel] = True
    a, b, s, p = 0, nA, nA + nB, nA + nB + nS
    allow[a:b, a:b] = torch.ones(nA, nA, dtype=torch.bool).tril()
    allow[b:s, b:s] = torch.ones(nB, nB, dtype=torch.bool).tril()
    if nS:
        allow[s:p, a:b] = True
        allow[s:p, b:s] = (posB[None, :] < posS[:, None]) & ~is_sel[None, :]
        allow[s:p, s:p] = torch.ones(nS, nS, dtype=torch.bool).tril()
    allow[p:, a:b] = True
    allow[p:, b:s] = ~is_sel[None, :]
    allow[p:, s:p] = True
    allow[p:, p:] = torch.ones(nP, nP, dtype=torch.bool).tril()
    pos = torch.cat([torch.arange(nA), posB, posS, torch.arange(nA + nB, nA + nB + nP)])
    mask = torch.zeros(n, n)
    mask.masked_fill_(~allow, NEG)
    return mask, pos


def kld(truth, pred):
    lt, lp = F.log_softmax(truth, -1), F.log_softmax(pred, -1)
    k = (lt.exp() * (lt - lp)).sum(-1)
    top = (truth.argmax(-1) == pred.argmax(-1)).float()
    return k.mean().item(), top.mean().item()


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--repos", nargs="+", required=True)
    ap.add_argument("--pairs", type=int, default=12)
    ap.add_argument("--max-a", type=int, default=1500)
    ap.add_argument("--max-b", type=int, default=1500)
    ap.add_argument("--min-b", type=int, default=400)
    ap.add_argument("--probe", type=int, default=256)
    ap.add_argument("--min-refs", type=int, default=3, help="minimum reference lines of B inside the cached part")
    ap.add_argument("--check", action="store_true", help="only verify the masked pass against a plain causal pass")
    ap.add_argument("--out", default="results.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.set_num_threads(8)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32, attn_implementation="eager").eval()
    nL = model.config.num_hidden_layers

    def enc(text):
        e = tok(text, return_offsets_mapping=True, add_special_tokens=False)
        return torch.tensor(e["input_ids"]), e["offset_mapping"]

    if args.check:
        ids, _ = enc(pathlib.Path(__file__).read_text()[:6000])
        n = len(ids)
        nP = n - 2 * (n // 3)
        plain = model(input_ids=ids[None], logits_to_keep=nP).logits[0].float()
        # degenerate stitch: B = everything after A with nothing selected must differ; a causal mask through the
        # same 4D path must match exactly
        mask = torch.zeros(n, n).masked_fill_(~torch.ones(n, n, dtype=torch.bool).tril(), NEG)
        same = logits_for(model, ids, torch.arange(n), mask, nP)
        print("4D causal vs plain: max |dlogit| =", (plain - same).abs().max().item())
        # full recompute through the stitch path (every B token selected) must also match the plain pass on P
        nA, nB = n // 3, n // 3
        sel = list(range(nB))
        m, pos = stitched_mask(nA, nB, sel, nP)
        x = torch.cat([ids[:nA], ids[nA:nA + nB], ids[nA:nA + nB], ids[nA + nB:]])
        st = logits_for(model, x, pos, m, nP)
        print("all-selected stitch vs plain on P: max |dlogit| =", (plain - st).abs().max().item())
        return

    rng = random.Random(args.seed)
    pairs = find_pairs(args.repos)
    rng.shuffle(pairs)
    print(f"{len(pairs)} candidate pairs", flush=True)

    done = 0
    out = open(args.out, "a")
    for repo, a_rel, b_rel, names in pairs:
        if done >= args.pairs:
            break
        repo = pathlib.Path(repo)
        a_src, b_src = (repo / a_rel).read_text(), (repo / b_rel).read_text()
        a_text = f"# file: {a_rel}\n{a_src}\n"
        b_head = f"# file: {b_rel}\n"
        idsA, _ = enc(a_text)
        idsBall, offB = enc(b_head + b_src)
        nA = min(len(idsA), args.max_a)
        idsA = idsA[-nA:]  # keep the end of A when truncating: the definitions closest to B
        if len(idsBall) < args.min_b + args.probe:
            continue
        nB = min(len(idsBall) - args.probe, args.max_b)
        idsB, idsP = idsBall[:nB], idsBall[nB:nB + args.probe]
        offB = offB[:nB]

        # lines of B (within the cached part) that reference A's names, as token sets
        lines = ref_lines(b_src, names)
        hl = len(b_head)
        line_of = []
        starts = [0]
        for ln in b_src.splitlines(keepends=True):
            starts.append(starts[-1] + len(ln))
        for s, e in offB:
            line_of.append(bisect.bisect_right(starts, max(s - hl, 0)))
        sel_lines = {i for i, ln in enumerate(line_of) if ln in lines and offB[i][0] >= hl}
        sel_ident = ref_tokens(b_src, names, offB, hl)
        n_ref_lines = len({line_of[i] for i in sel_lines})
        if n_ref_lines < args.min_refs:
            continue

        t0 = time.time()
        full = torch.cat([idsA, idsB, idsP])
        nP = len(idsP)
        out_full = model(input_ids=full[None], use_cache=True, logits_to_keep=nP)

        # CacheBlend-style selector: value deviation at layer 1 between the full pass and B's chunk-alone pass
        alone = model(input_ids=idsB[None], position_ids=torch.arange(nA, nA + nB)[None], use_cache=True, logits_to_keep=1)
        def layer_v(cache, l):
            return cache.layers[l].values if hasattr(cache, "layers") else cache[l][1]
        v_full = layer_v(out_full.past_key_values, 1)[0, :, nA:nA + nB]
        v_alone = layer_v(alone.past_key_values, 1)[0, :, :nB]
        dev1 = (v_full - v_alone).float().pow(2).sum((0, 2))
        dev_all = torch.zeros(nB)
        for l in range(nL):
            dev_all += (layer_v(out_full.past_key_values, l)[0, :, nA:nA + nB] - layer_v(alone.past_key_values, l)[0, :, :nB]).float().pow(2).sum((0, 2)) / \
                       layer_v(out_full.past_key_values, l)[0, :, nA:nA + nB].float().pow(2).sum((0, 2)).mean()

        def stitch(sel):
            m, pos = stitched_mask(nA, nB, sel, nP)
            x = torch.cat([idsA, idsB, idsB[sorted(sel)] if sel else idsB[:0], idsP])
            return logits_for(model, x, pos, m, nP)

        def topk(score, k):
            return set(torch.topk(score, k).indices.tolist()) if k else set()

        k_lines = len(sel_lines)
        k15 = int(0.15 * nB)
        # head: the chunk's first tokens act as an attention sink when B is cached alone
        sink = set(range(min(4, nB)))
        sel = {
            "none": set(),
            "random@lines": set(rng.sample(range(nB), k_lines)),
            "cacheblend@lines": topk(dev1, k_lines),
            "oracle@lines": topk(dev_all, k_lines),
            "code_lines": sel_lines,
            "code_ident@lines": sel_ident | set(rng.sample(sorted(set(range(nB)) - sel_ident), max(0, k_lines - len(sel_ident)))),
            "code_ident_only": sel_ident,
            "code_lines+sink": sel_lines | sink,
            "sink_only": sink,
            "sink+random@lines": sink | set(rng.sample(range(4, nB), max(0, k_lines - 4))),
            "sink+cacheblend@lines": sink | set(i for i in torch.topk(dev1, k_lines).indices.tolist() if i >= 4),
            "cacheblend@15%": topk(dev1, k15),
            "oracle@15%": topk(dev_all, k15),
        }
        # P's own positions, each predicting the next token (the same rows stitch() returns)
        truth_p = out_full.logits[0].float()
        row = {"repo": repo.name, "a": a_rel, "b": b_rel, "nA": nA, "nB": nB, "nP": nP, "ref_lines": n_ref_lines,
               "k_lines": k_lines, "k_ident": len(sel_ident),
               # where CacheBlend's picks fall: share within B's first 16 tokens, and how far into B the rest sit
               "cb_first16": sum(1 for i in topk(dev1, k_lines) if i < 16) / max(1, k_lines),
               "cb_overlap_lines": len(topk(dev1, k_lines) & sel_lines) / max(1, k_lines)}
        for name, s in sel.items():
            kl, top = kld(truth_p, stitch(s))
            row[name] = {"k": len(s), "frac": len(s) / nB, "kld": kl, "top1": top}
        row["secs"] = time.time() - t0
        out.write(json.dumps(row) + "\n")
        out.flush()
        done += 1
        print(f"[{done}] {repo.name}:{b_rel} <- {a_rel} nA={nA} nB={nB} lines={k_lines} ({k_lines/nB:.1%}) "
              + " ".join(f"{k}={v['kld']:.4f}" for k, v in row.items() if isinstance(v, dict))
              + f" {row['secs']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
