
import argparse
import json
import random
import time
import re
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import requests
import numpy as np
import editdistance as ed
from tqdm import tqdm

from transformers import AutoProcessor
from transformers.pipelines.audio_utils import ffmpeg_read
from whisper_normalizer.english import EnglishTextNormalizer
from whisper_normalizer.basic import BasicTextNormalizer
from cn_tn import TextNorm
import zhconv
from evaluate_tokenizer import EvaluationTokenizer

from vllm import LLM, SamplingParams


PUNCS = "!,.?;:"
MAX_SECONDS = 30.0

english_normalizer = EnglishTextNormalizer()
basic_normalizer = BasicTextNormalizer()
chinese_normalizer = TextNorm(
    to_banjiao=False,
    to_upper=False,
    to_lower=False,
    remove_fillers=False,
    remove_erhua=False,
    check_chars=False,
    remove_space=False,
    cc_mode="",
)

ds_collections = {
    "librispeech": {"path": "data/asr/librispeech_eval.jsonl", "language": "en"},
    "aishell2": {"path": "data/asr/aishell2_eval.jsonl", "language": "zh"},
    "cv15_en": {"path": "data/asr/cv15_asr_en_eval.jsonl", "language": "en"},
    "cv15_zh": {"path": "data/asr/cv15_asr_zh_eval.jsonl", "language": "zh"},
    "cv15_yue": {"path": "data/asr/cv15_asr_yue_eval.jsonl", "language": "yue"},
    "cv15_fr": {"path": "data/asr/cv15_asr_fr_eval.jsonl", "language": "fr"},
    "fleurs_zh": {"path": "data/asr/fleurs_asr_zh_eval.jsonl", "language": "zh"},
}


class AudioDataset:
    """jsonl 每行: {"audio":..., "prompt":..., "gt":..., "source":...}"""

    def __init__(self, jsonl_path: Path):
        self.jsonl_path = jsonl_path
        self.datas = jsonl_path.read_text(encoding="utf-8").splitlines()

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx: int):
        data = json.loads(self.datas[idx].strip())
        return {
            "audio": data["audio"], 
            "prompt": data["prompt"],
            "gt": data["gt"],
            "source": data.get("source", "unknown"),
        }


def read_audio_bytes(audio_path: str, asr_root: Path) -> bytes:
    if audio_path.startswith(("http://", "https://")):
        return requests.get(audio_path, timeout=30).content
    p = asr_root / Path(audio_path)
    return p.read_bytes()


def load_audio_array(
    audio_path: str,
    asr_root: Path,
    sampling_rate: int,
    max_seconds: float = MAX_SECONDS,
) -> Optional[Tuple[np.ndarray, int]]:
    """Return (audio_1d_float32, sr). If longer than max_seconds -> None."""
    audio = ffmpeg_read(read_audio_bytes(audio_path, asr_root), sampling_rate=sampling_rate)
    audio = np.asarray(audio, dtype=np.float32)

    if audio.ndim == 2:
        audio = audio.mean(axis=0)

    audio = audio.reshape(-1)
    max_samples = int(max_seconds * sampling_rate)
    if audio.shape[0] > max_samples:
        return None
    return audio, sampling_rate


def remove_sp(text: str, language: str) -> str:
    """Remove special tokens & normalize whitespace/punctuation for fair WER."""
    t = re.sub(r"<\|.*?\|>", " ", text)
    t = re.sub(r"\s+", " ", t)
    t = re.sub(f" ?([{PUNCS}])", r"\1", t)
    t = t.lstrip(" ")
    if language == "zh":
        t = re.sub(r"\s+", "", t)
    return t


def compute_wer(refs: List[str], hyps: List[str], language: str) -> float:
    distance = 0
    ref_length = 0

    tokenizer = EvaluationTokenizer(
        tokenizer_type="none",
        lowercase=True,
        punctuation_removal=True,
        character_tokenization=False,
    )

    for ref, pred in zip(refs, hyps):
        if language == "yue":
            ref = zhconv.convert(ref, "zh-cn")
            pred = zhconv.convert(pred, "zh-cn")

        if language == "en":
            ref = english_normalizer(ref)
            pred = english_normalizer(pred)
        elif language == "zh":
            ref = chinese_normalizer(ref)
            pred = chinese_normalizer(pred)
        else:
            ref = basic_normalizer(ref)
            pred = basic_normalizer(pred)

        ref_items = tokenizer.tokenize(ref).split()
        pred_items = tokenizer.tokenize(pred).split()

        # zh/yue often compute char-level distance
        if language in ["zh", "yue"]:
            ref_items = list("".join(ref_items))
            pred_items = list("".join(pred_items))

        distance += ed.eval(ref_items, pred_items)
        ref_length += len(ref_items)

    return distance / max(ref_length, 1)


def batched_indices(n: int, bs: int):
    for i in range(0, n, bs):
        yield range(i, min(i + bs, n))


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="librispeech")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tp", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--asr-root", type=str, default="")
    parser.add_argument("--model-path", type=str, default="/cx8k/fs100/jies_data/llm/weights/Qwen/Qwen2-Audio-7B",)
    parser.add_argument("--pad-audio", action="store_true", default=True)
    parser.add_argument("--out-dir", type=str, default="results_asr")
    parser.add_argument("--summary-file", type=str, default="summary_all.txt",)

    args = parser.parse_args()
    random.seed(args.seed)

    base_dir = Path("/home/jies/code/extra/model/Qwen2-Audio-main")
    asr_root = (base_dir / args.asr_root).resolve()

    if args.dataset not in ds_collections:
        raise ValueError(f"Unknown dataset: {args.dataset}. Available: {list(ds_collections.keys())}")

    jsonl_path = (base_dir / ds_collections[args.dataset]["path"]).resolve()
    lan = ds_collections[args.dataset]["language"]

    processor = AutoProcessor.from_pretrained(args.model_path)
    sr = processor.feature_extractor.sampling_rate

    llm = LLM(
        model=args.model_path,
        trust_remote_code=True,
        tensor_parallel_size=args.tp,
        max_model_len=args.max_model_len,
        enforce_eager=True,  # 添加这行，禁用 CUDA Graph 和 Flash Attention
        dtype="bfloat16",    # 明确指定数据类型
    )

    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=0.01,
        max_tokens=args.max_tokens,
        repetition_penalty=1.1,
        top_k=0,
    )

    dataset = AudioDataset(jsonl_path)

    gts, sources, rets, audio_paths = [], [], [], []

    skipped = 0
    skipped_by_source: Dict[str, int] = {}

    total_audio_seconds = 0.0
    total_generate_seconds = 0.0
    audio_seconds_by_source: Dict[str, float] = {}
    generate_seconds_by_source: Dict[str, float] = {}

    for idxs in tqdm(list(batched_indices(len(dataset), args.batch_size))):
        batch = [dataset[i] for i in idxs]

        audio_list = []
        valid_meta = []

        for item in batch:
            ret = load_audio_array(item["audio"], asr_root, sampling_rate=sr)
            if ret is None:
                skipped += 1
                src = item.get("source", "unknown")
                skipped_by_source[src] = skipped_by_source.get(src, 0) + 1
                continue

            a, _sr = ret
            audio_list.append((a, _sr))
            valid_meta.append(item)

        if not audio_list:
            continue

        if args.pad_audio and len(audio_list) > 1:
            max_len = max(a.shape[0] for a, _ in audio_list)
            audio_list = [
                (np.pad(a, (0, max_len - a.shape[0])) if a.shape[0] < max_len else a, _sr)
                for a, _sr in audio_list
            ]

        audio_list = [(np.ascontiguousarray(a, dtype=np.float32), _sr) for a, _sr in audio_list]

        batch_audio_secs = [(a.shape[0] / float(_sr)) for (a, _sr) in audio_list]
        batch_total_audio_secs = float(sum(batch_audio_secs))

        inputs = []
        batch_sources = []
        for item, (a, _sr), asec in zip(valid_meta, audio_list, batch_audio_secs):
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
            batch_sources.append(item["source"])
            audio_paths.append(item["audio"])

        t0 = time.perf_counter()
        outputs = llm.generate(inputs, sampling_params=sampling_params)
        gen_time = time.perf_counter() - t0

        total_generate_seconds += gen_time
        total_audio_seconds += batch_total_audio_secs

        if batch_total_audio_secs > 0:
            for src, asec in zip(batch_sources, batch_audio_secs):
                share = gen_time * (asec / batch_total_audio_secs)
                audio_seconds_by_source[src] = audio_seconds_by_source.get(src, 0.0) + asec
                generate_seconds_by_source[src] = generate_seconds_by_source.get(src, 0.0) + share

        for out in outputs:
            rets.append(out.outputs[0].text)

    print(f"Skipped long audios (> {MAX_SECONDS}s): {skipped}")
    if skipped_by_source:
        print("Skipped by source:", skipped_by_source)

    print(f"Evaluating {args.dataset} ...")

    results = []
    for gt, response, source, audio_path in zip(gts, rets, sources, audio_paths):
        results.append(
            {"gt": gt, "response": response, "source": source, "audio_path": audio_path}
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = time.strftime("%y%m%d%H%M%S", time.localtime())
    results_file = out_dir / f"{args.dataset}_{ts}.json"
    payload = {
        "dataset": args.dataset,
        "model_path": args.model_path,
        "max_seconds": MAX_SECONDS,
        "skipped_long_audios": skipped,
        "skipped_by_source": skipped_by_source,
        "results": results,
    }
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    results_by_source: Dict[str, List[dict]] = {}
    for item in results:
        results_by_source.setdefault(item["source"], []).append(item)

    lines: List[str] = []

    for source in results_by_source:
        refs, hyps = [], []
        for r in results_by_source[source]:
            refs.append(remove_sp(r["gt"], lan))
            hyps.append(remove_sp(r["response"], lan))
        wer = compute_wer(refs, hyps, lan)

        line = f"source: {source}  cnt: {len(refs)}  wer: {wer:.4f}"
        print(line)
        lines.append(line)

    overall_rtf = total_generate_seconds / max(total_audio_seconds, 1e-9)

    line = f"Total audio seconds: {total_audio_seconds:.2f}s"
    print(line); lines.append(line)

    line = f"Total generate time: {total_generate_seconds:.2f}s"
    print(line); lines.append(line)

    line = f"Overall RTF: {overall_rtf:.4f} (lower is faster)"
    print(line); lines.append(line)

    if audio_seconds_by_source:
        line = "RTF by source:"
        print(line); lines.append(line)

        for src in sorted(audio_seconds_by_source.keys()):
            asec = audio_seconds_by_source[src]
            gsec = generate_seconds_by_source.get(src, 0.0)
            rtf = gsec / max(asec, 1e-9)

            line = f"  source: {src:<15} audio_s: {asec:>8.2f}  gen_s: {gsec:>8.2f}  RTF: {rtf:.4f}"
            print(line)
            lines.append(line)

    summary_path = Path(args.summary_file)
    if not summary_path.is_absolute():
        summary_path = out_dir / summary_path

    header = [
        "",
        "=" * 80,
        f"DATASET: {args.dataset}",
        f"TIME: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}",
        f"MODEL: {args.model_path}",
        f"RESULTS_JSON: {results_file}",
        f"MAX_SECONDS: {MAX_SECONDS}",
        f"SKIPPED_LONG: {skipped}",
        f"SKIPPED_BY_SOURCE: {skipped_by_source}",
        "-" * 80,
    ]
    block = "\n".join(header + lines) + "\n"

    with open(summary_path, "a", encoding="utf-8") as f:
        f.write(block)

    print(f"[Summary appended to] {summary_path}")


if __name__ == "__main__":
    main()