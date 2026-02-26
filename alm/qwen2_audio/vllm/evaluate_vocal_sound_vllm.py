import argparse
import json
import random
import time
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import numpy as np
import requests
from tqdm import tqdm
from transformers import AutoProcessor
from transformers.pipelines.audio_utils import ffmpeg_read

from vllm import LLM, SamplingParams
from sklearn.metrics import accuracy_score


ds_collections = {
    "vocalsound": {"path": "data/vsc/vocalsound_eval.jsonl"},
}


class AudioDataset:
    def __init__(self, jsonl_path: Path):
        self.jsonl_path = jsonl_path
        self.datas = jsonl_path.read_text(encoding="utf-8").splitlines()

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx: int) -> Dict:
        data = json.loads(self.datas[idx])
        return {
            "audio": data["audio"],
            "prompt": data["prompt"],
            "gt": data["gt"],
            "source": data.get("source", "unknown"),
        }


def read_audio_bytes(audio_path: str, base_dir: Optional[Path] = None) -> bytes:
    if audio_path.startswith(("http://", "https://")):
        return requests.get(audio_path, timeout=30).content
    p = Path(audio_path)
    if base_dir is not None:
        p = base_dir / p
    return p.read_bytes()


def load_audio_array(
    audio_path: str,
    sampling_rate: int,
    base_dir: Optional[Path] = None,
    max_seconds: float = 30.0,
) -> Optional[np.ndarray]:
    raw = read_audio_bytes(audio_path, base_dir=base_dir)
    audio = ffmpeg_read(raw, sampling_rate=sampling_rate)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=0)
    audio = audio.reshape(-1)
    if audio.shape[0] > int(max_seconds * sampling_rate):
        return None
    return audio


def build_vllm_prompt(processor: AutoProcessor, text_prompt: str) -> str:
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "audio"},
                {"type": "text", "text": text_prompt},
            ],
        }
    ]
    return processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=False,
        add_audio_id=True,
    )


def batched_indices(n: int, bs: int):
    for i in range(0, n, bs):
        yield range(i, min(i + bs, n))


def pad_audio_list(audio_list: List[Tuple[np.ndarray, int]]) -> List[Tuple[np.ndarray, int]]:
    max_len = max(a.shape[0] for a, _ in audio_list)
    out = []
    for a, sr in audio_list:
        if a.shape[0] < max_len:
            a = np.pad(a, (0, max_len - a.shape[0]))
        out.append((np.ascontiguousarray(a, dtype=np.float32), sr))
    return out


def contiguous_audio_list(audio_list: List[Tuple[np.ndarray, int]]) -> List[Tuple[np.ndarray, int]]:
    return [(np.ascontiguousarray(a, dtype=np.float32), sr) for a, sr in audio_list]


def append_lines(path: Path, lines: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


def norm_label(s: str) -> str:
    return s.strip().lower()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/cx8k/fs100/jies_data/llm/weights/Qwen/Qwen2-Audio-7B")
    parser.add_argument("--dataset", type=str, default="vocalsound")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--tp", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=256)

    parser.add_argument("--audio-root", type=str, default="")
    parser.add_argument("--pad-audio", action="store_true")
    parser.add_argument("--max-seconds", type=float, default=30.0)

    parser.add_argument("--out-dir", type=str, default="results_vsc")
    parser.add_argument("--summary-file", type=str, default="summary_all.txt")

    args = parser.parse_args()
    random.seed(args.seed)

    base_dir = Path("/home/jies/code/extra/model/Qwen2-Audio-main")
    jsonl_path = (base_dir / ds_collections[args.dataset]["path"]).resolve()
    audio_root = (base_dir / args.audio_root).resolve()

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    sr = processor.feature_extractor.sampling_rate

    llm = LLM(
        model=args.model_path,
        trust_remote_code=True,
        tensor_parallel_size=args.tp,
        max_model_len=args.max_model_len,
        limit_mm_per_prompt={"audio": 5},
        enforce_eager=True,
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.max_tokens,
        min_tokens=1,
    )

    dataset = AudioDataset(jsonl_path)

    gts: List[str] = []
    sources: List[str] = []
    rets: List[str] = []
    audio_paths: List[str] = []

    skipped = 0
    total_audio_sec = 0.0
    total_infer_sec = 0.0

    for idxs in tqdm(list(batched_indices(len(dataset), args.batch_size)), desc="batches"):
        batch = [dataset[i] for i in idxs]

        audio_list: List[Tuple[np.ndarray, int]] = []
        kept_items: List[Dict] = []

        for item in batch:
            a = load_audio_array(
                item["audio"],
                sampling_rate=sr,
                base_dir=audio_root,
                max_seconds=args.max_seconds,
            )
            if a is None:
                skipped += 1
                continue

            audio_list.append((a, sr))
            total_audio_sec += a.shape[0] / sr
            kept_items.append(item)

        if not kept_items:
            continue

        if args.pad_audio and len(audio_list) > 1:
            audio_list = pad_audio_list(audio_list)
        else:
            audio_list = contiguous_audio_list(audio_list)

        inputs = []
        for item, (a, _sr) in zip(kept_items, audio_list):
            # prompt = build_vllm_prompt(processor, item["prompt"])
            prompt = f"<|audio_bos|><|AUDIO|><|audio_eos|>{item['prompt']}"
            inputs.append(
                {
                    "prompt": prompt,
                    "multi_modal_data": {"audio": [(a, _sr)]},
                }
            )
            gts.append(item["gt"])
            sources.append(item["source"])
            audio_paths.append(item["audio"])

        t0 = time.perf_counter()
        outputs = llm.generate(inputs, sampling_params=sampling_params)
        total_infer_sec += (time.perf_counter() - t0)

        for out in outputs:
            rets.append(out.outputs[0].text.lstrip())

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = time.strftime("%y%m%d%H%M%S", time.localtime())
    results_path = out_dir / f"{args.dataset}_{ts}.json"

    summary_path = Path(args.summary_file)
    if not summary_path.is_absolute():
        summary_path = out_dir / summary_path

    header = [
        "",
        "=" * 80,
        f"DATASET: {args.dataset}",
        f"TIME: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}",
        f"MODEL: {args.model_path}",
        f"RESULTS_JSON: {results_path}",
        f"TOTAL: {len(dataset)}  USED: {len(rets)}  SKIPPED: {skipped}",
        "-" * 80,
    ]
    append_lines(summary_path, header)

    print(f"Evaluating {args.dataset} ... total={len(dataset)} used={len(rets)} skipped={skipped}")

    if total_audio_sec > 0:
        rtf = total_infer_sec / total_audio_sec
        speed = total_audio_sec / total_infer_sec if total_infer_sec > 0 else float("inf")
        rtf_line = (
            f"RTF: {rtf:.4f} (infer_sec={total_infer_sec:.2f}, "
            f"audio_sec={total_audio_sec:.2f}, speed={speed:.2f}x)"
        )
        print(rtf_line)
        append_lines(summary_path, [rtf_line])

    results = [
        {"gt": gt, "response": resp, "source": src, "audio_path": ap}
        for gt, resp, src, ap in zip(gts, rets, sources, audio_paths)
    ]
    results_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved ->", str(results_path))

    results_by_source: Dict[str, List[Dict]] = {}
    for item in results:
        results_by_source.setdefault(item["source"], []).append(item)

    acc_lines: List[str] = []
    for src, items in results_by_source.items():
        refs = [norm_label(x["gt"]) for x in items]
        hyps = [norm_label(x["response"]) for x in items]
        acc = accuracy_score(refs, hyps)
        line = f"{src} ACC_score: {acc:.6f} {len(items)}"
        print(line)
        acc_lines.append(line)

    if acc_lines:
        append_lines(summary_path, acc_lines)

    print("summary ->", str(summary_path))


if __name__ == "__main__":
    main()