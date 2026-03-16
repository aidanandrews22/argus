# Implementing ARGUS from scratch

# ROI: region of interest
# VLM: visual language model
# MMLM: multi-modal large language model
# CoT: chain of thought

import os
import re
import json
import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches

from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


# default: Load the model on the available device(s)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

def generate(messages, max_new_tokens=256):
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to("cuda")
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids_trimmed = generated_ids[0][len(inputs.input_ids[0]):]
    return processor.decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()

def parse_bounding_box_json(text):
    text = text.replace("```json", "").replace("```", "").strip()
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            data = [data]
        return data
    except json.JSONDecodeError:
        # would be nice to have a fallback here
        raise ValueError(f"Invalid JSON: {text}")

def normalize_bounding_box(image, bounding_box):
    width, height = image.size

    processed_image = processor.image_processor(image)
    t, h_patches, w_patches = processed_image["image_grid_thw"][0]
    input_height, input_width = int(h_patches)*14, int(w_patches)*14

    x1, y1, x2, y2 = bounding_box
    x1 = int(x1 / input_width * width)
    y1 = int(y1 / input_height * height)
    x2 = int(x2 / input_width * width)
    y2 = int(y2 / input_height * height)

    return (x1, y1, x2, y2)

def get_padded_roi_bbox(image, bounding_box, expand=0.2):
    width, height = image.size
    x1, y1, x2, y2 = bounding_box
    bw, bh = x2 - x1, y2 - y1
    x1, y1 = max(0, x1 - int(bw * expand)), max(0, y1 - int(bh * expand))
    x2, y2 = min(width, x2 + int(bw * expand)), min(height, y2 + int(bh * expand))
    # square padding
    side = max(x2-x1, y2-y1)
    cx, cy = (x1+x2)//2, (y1+y2)//2
    x1, y1 = max(0, cx - side//2), max(0, cy - side//2)
    x2, y2 = min(width, x1 + side), min(height, y1 + side)
    return (x1, y1, x2, y2)

def crop_roi(image, bounding_box):
    x1, y1, x2, y2 = get_padded_roi_bbox(image, bounding_box)
    crop = image.crop((x1, y1, x2, y2))

    width, height = crop.size
    side = max(width, height)
    padded = Image.new("RGB", (side, side), (0, 0, 0))
    padded.paste(crop, ((side - width)//2, (side - height)//2))

    target = min(image.size)
    if side < target:
        padded = padded.resize((target, target), Image.Resampling.LANCZOS)

    return padded

def plot_image_with_bboxes(processor, image, bboxes, save_path=None, expand=0.2):
    image = Image.open(image) if isinstance(image, str) else image
    _, ax = plt.subplots(1)
    ax.imshow(image)

    if isinstance(bboxes, list) and all(isinstance(bbox, dict) for bbox in bboxes):
        colors = plt.cm.rainbow(np.linspace(0, 1, len(bboxes)))

        for i, (bbox, color) in enumerate(zip(bboxes, colors)):
            label = bbox.get("label", "") or f"ROI {i + 1}"
            x_min, y_min, x_max, y_max = bbox.get("bbox_2d")

            orig = normalize_bounding_box(image, (x_min, y_min, x_max, y_max))
            x_min_norm, y_min_norm, x_max_norm, y_max_norm = orig
            w_orig = x_max_norm - x_min_norm
            h_orig = y_max_norm - y_min_norm

            rect_orig = patches.Rectangle(
                (x_min_norm, y_min_norm),
                w_orig,
                h_orig,
                linewidth=2,
                edgecolor=color,
                facecolor="none",
                label="Original ROI",
            )
            ax.add_patch(rect_orig)
            ax.text(
                x_min_norm,
                y_min_norm - 4,
                f"{label} (original)",
                color=color,
                fontsize=9,
                fontweight="bold",
                bbox=dict(facecolor="white", edgecolor=color, alpha=0.8),
            )

            padded = get_padded_roi_bbox(image, orig, expand)
            px1, py1, px2, py2 = padded
            rect_pad = patches.Rectangle(
                (px1, py1),
                px2 - px1,
                py2 - py1,
                linewidth=1.5,
                edgecolor=color,
                facecolor="none",
                linestyle="--",
                label="Padded ROI",
            )
            ax.add_patch(rect_pad)
            ax.text(
                px1,
                py1 - 4,
                f"{label} (padded)",
                color=color,
                fontsize=8,
                bbox=dict(facecolor="white", edgecolor=color, alpha=0.6),
            )

    plt.axis("off")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close()


def argus(image_path: str, question: str, annotated_save_path: str | None = None, roi_save_path: str | None = None) -> str:
    img = Image.open(image_path)

    # Step 1: ROI sampling -- ask the VLM to locate relevant regions using bounding box annotations
    grounding_message = [
        {"role": "system", "content": "You are a helpful assistant that can output the bounding box coordinates of the relevant regions in the image. You specialize in giving rich regions of interest that encompass the **entire** relevant region of the image. You must ensure it is large enough to encapsulate everything relevant, but minimize non-relevant content (ie. dont make it too large)."},
        {"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": f"Output the bounding box coordinates of the single most relevant region in the image. Report the bbox coordinates in json format. With respect to this question: {question}"}
        ]}
    ]
    raw_response = generate(grounding_message)
    print(f"Raw response: {raw_response}")

    bounding_box_json = parse_bounding_box_json(raw_response)
    if not bounding_box_json:
        raise ValueError(f"No bounding box found in the response: {raw_response}")

    raw_box = bounding_box_json[0]["bbox_2d"]
    label = bounding_box_json[0].get("label", "")
    bounding_box = normalize_bounding_box(img, raw_box)
    print(f"Normalized bounding box: {bounding_box}")
    print(f"Label: {label}")

    # Step 2: Crop the ROI with padding
    cropped_roi = crop_roi(img, bounding_box)

    if annotated_save_path:
        plot_image_with_bboxes(processor, img, bounding_box_json, save_path=annotated_save_path)
    if roi_save_path:
        cropped_roi.save(roi_save_path)

    # Step 3: Re-engage the VLM -- send both the cropped ROI + original image and the question back to the VLM
    reasoning_message = [
        {"role": "system", "content": "You are a helpful assistant that can reason about the image and the question."},
        {"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "image", "image": cropped_roi},
                {"type": "text", "text": f"Image 1 is the full scene. Image 2 is a zoomed crop of '{label}'. Using both images, answer the following question: {question}"}
        ]}
    ]
    answer = generate(reasoning_message)
    print(f"Enriched response: {answer}")
    return answer

def baseline(image_path: str, question: str) -> str:
    img = Image.open(image_path)

    base_message = [{"role": "user", "content": [
        {"type": "image", "image": img},
        {"type": "text", "text": question}
    ]}]
    return generate(base_message).strip()

def evaluate_against_visual_cot(dataset_path: str = "./Visual-CoT/viscot_363k.json", image_root_path: str = "./Visual-CoT", output_dir: str = "./eval_results", n: int = 50):
    os.makedirs(output_dir, exist_ok=True)
    with open(dataset_path) as f:
        data = json.load(f)

    results: list[dict] = []
    correct_argus, correct_baseline, total, skipped = 0, 0, 0, 0

    for sample in data:
        if total >= n:
            break

        conversations = sample["conversations"]
        question = conversations[0]["value"].replace("<image>\n", "").split("Please provide the bounding box")[0].strip()
        ground_truth = conversations[3]["value"].strip()

        # Dataset paths use cot/ but actual directory is cot_image_data/
        image_rel = sample["image"][0].replace("cot/", "cot_image_data/", 1)
        image_path = f"{image_root_path}/{image_rel}"
        try:
            image = Image.open(image_path)
        except FileNotFoundError:
            skipped += 1
            print(f"[SKIP] Image not found: {image_path}")
            continue

        annotated_path = f"{output_dir}/{total:04d}_annotated.png"
        try:
            argus_answer = argus(image_path, question, annotated_save_path=annotated_path).strip()
        except (ValueError, KeyError) as e:
            skipped += 1
            print(f"[SKIP] Argus failed: {e}")
            continue

        base_message = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": question}
        ]}]
        base_answer = generate(base_message).strip()

        answer_ok = compare_answers(argus_answer, ground_truth)
        baseline_ok = compare_answers(base_answer, ground_truth)
        correct_argus += (answer_ok >= 0.7)
        correct_baseline += (baseline_ok >= 0.7)
        total += 1

        result = {
            "index": total,
            "image_path": image_path,
            "annotated_path": annotated_path,
            "question": question,
            "ground_truth": ground_truth,
            "argus_answer": argus_answer,
            "baseline_answer": base_answer,
            "argus_correct": str(answer_ok),
            "baseline_correct": str(baseline_ok),
        }
        results.append(result)

        with open(f"{output_dir}/results.json", "w") as f:
            json.dump(results, f, indent=2)

        print(f"[{total}] I: {image_path}\n  Q: {question}\n  ARGUS: {argus_answer}\n  BASE:  {base_answer}\n  GT:    {ground_truth}\n  argus={answer_ok} base={baseline_ok}")

    print(f"\nResults: {total} evaluated, {skipped} skipped")
    print(f"Argus:    {correct_argus}/{total} ({correct_argus/total*100:.2f}%)")
    print(f"Baseline: {correct_baseline}/{total} ({correct_baseline/total*100:.2f}%)")

def evaluate_against_my_benchmark(dataset_path: str = "./benchmark/benchmark.json", image_root_path: str = "./benchmark", output_dir: str = "./benchmark_eval_results", n: int = 50):
    os.makedirs(output_dir, exist_ok=True)
    with open(dataset_path) as f:
        data = json.load(f)

    results: list[dict] = []
    correct_argus, correct_baseline, total, skipped = 0, 0, 0, 0

    for sample in data:
        if total >= n:
            break

        question = sample["question"]
        ground_truth = sample["ground_truth"]
        image_path = f"{image_root_path}/{sample['image_path']}"
        try:
            image = Image.open(image_path)
        except FileNotFoundError:
            skipped += 1
            print(f"[SKIP] Image not found: {image_path}")
            continue

        annotated_path = f"{output_dir}/{total:04d}_annotated.png"
        roi_path = f"{output_dir}/{total:04d}_roi.png"
        try:
            argus_answer = argus(image_path, question, annotated_save_path=annotated_path, roi_save_path=roi_path).strip()
        except (ValueError, KeyError) as e:
            skipped += 1
            print(f"[SKIP] Argus failed: {e}")
            continue

        base_answer = baseline(image_path, question)

        argus_sim = compare_answers(argus_answer, ground_truth)
        baseline_sim = compare_answers(base_answer, ground_truth)
        correct_argus += (argus_sim >= 0.7)
        correct_baseline += (baseline_sim >= 0.7)
        total += 1

        result = {
            "index": total,
            "image_path": image_path,
            "annotated_path": annotated_path,
            "roi_path": roi_path,
            "question": question,
            "ground_truth": ground_truth,
            "argus_answer": argus_answer,
            "baseline_answer": base_answer,
            "argus_similarity": str(argus_sim),
            "baseline_similarity": str(baseline_sim),
        }
        results.append(result)

        with open(f"{output_dir}/results.json", "w") as f:
            json.dump(results, f, indent=2)

        print(f"[{total}] I: {image_path}\n  Q: {question}\n  ARGUS: {argus_answer}\n  BASE:  {base_answer}\n  GT:    {ground_truth}\n  argus_sim={argus_sim:.3f} base_sim={baseline_sim:.3f}")

    print(f"\nResults: {total} evaluated, {skipped} skipped")
    print(f"Argus:    {correct_argus}/{total} ({correct_argus/total*100:.2f}%)")
    print(f"Baseline: {correct_baseline}/{total} ({correct_baseline/total*100:.2f}%)")


def compare_answers(answer1, answer2):
    embeddings_1 = embedding_model.encode(answer1)
    embeddings_2 = embedding_model.encode(answer2)
    return np.dot(embeddings_1, embeddings_2) / (np.linalg.norm(embeddings_1) * np.linalg.norm(embeddings_2))

def main():
    # image_path = "assets/image.png"
    # question = "what color are the majority of people on the padio wearing?"
    # a = argus(image_path, question)
    # print(f"ARGUS: {a}")
    # b = baseline(image_path, question)
    # print(f"BASELINE: {b}")
    # evaluate_against_visual_cot()
    evaluate_against_my_benchmark()
main()