"""Build a small, explicitly unconfirmed RGB review pack, not training truth."""

import csv
import hashlib
import html
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main():
    destination = ROOT / "runs/stage1/public_trial_review"
    destination.mkdir(parents=True, exist_ok=False)
    annotation = json.loads((ROOT / "data/raw/coco2017/annotations/person_keypoints_val2017.json").read_text())
    images = {item["id"]: item for item in annotation["images"]}
    licenses = {item["id"]: item for item in annotation["licenses"]}
    crop_list = json.loads((ROOT / "runs/stage1/coco_chair_review/crops.json").read_text())["candidates"]
    crops = {item["annotation_id"]: item for item in crop_list}
    proposals = [
        ("coco_051610", 51610, None, "upright", "side", "Seated at a laptop; trunk and head approximately upright. Child, floor seating; limited domain match."),
        ("coco_368212", 368212, None, "upright", "front", "Front-facing seated child; no clear head or trunk lean. Child domain limitation."),
        ("coco_504074_a484032", 504074, 484032, "upright", "front", "Seated adult with near-upright trunk and head; arms rest in lap."),
        ("coco_177489_a184170", 177489, 184170, "upright", "oblique", "Seated adult at table, trunk near upright; head turned toward camera. Reviewer must decide whether the turn warrants exclusion."),
        ("coco_076547_a188853", 76547, 188853, "head_down", "side", "Seated adult looks downward at table; trunk comparatively upright. Mild hunch makes the head-down/compound boundary worth human review."),
        ("coco_455937_a192465", 455937, 192465, "forward_lean", "side", "Seated on sofa, trunk inclined forward and elbows near knees; face directed ahead."),
        ("coco_122046_a469277", 122046, 469277, None, "oblique", "Exclude proposal: downward/sideways head orientation and possible trunk lean coexist; cannot isolate a pure class."),
        ("coco_454404_a424966", 454404, 424966, None, "front", "Exclude proposal: raised knee, head tilt and downward gaze; not evidence of pure torso side lean."),
        ("coco_017899_a209675", 17899, 209675, None, "oblique", "Exclude proposal: standing at a table, despite chair association; not seated."),
        ("coco_061418_a232029", 61418, 232029, None, "side", "Exclude proposal: very strong forward bend together with lowered head; compound pose."),
    ]
    records = []
    for sample_id, image_id, annotation_id, label, view, note in proposals:
        image = images[image_id]
        crop = crops[annotation_id] if annotation_id is not None else None
        original_path = ROOT / "data/raw/coco2017/images/val2017" / image["file_name"]
        image_path = Path(crop["crop_path"]) if crop else original_path
        record = {
            "sample_id": sample_id,
            "image_path": str(image_path.relative_to(ROOT)),
            "original_path": str(original_path.relative_to(ROOT)),
            "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
            "image_id": image_id,
            "annotation_id": annotation_id,
            "original_metadata": image,
            "license": licenses[image["license"]],
            "crop_rectangle": crop["crop_rectangle"] if crop else None,
            "proposed_label": label,
            "proposed_view": view,
            "proposed_action": "review" if label else "exclude",
            "reason": note,
            "reviewer": "assistant visual proposal",
            "human_confirmed": False,
            "human_label": None,
            "group_id": "coco_image_{:012d}".format(image_id),
        }
        records.append(record)
    # A separately sourced adult example supplements the oblique pose check.
    fall_image = ROOT / "runs/stage1/dataset_audit/rgb_fall_sit_forward.jpg"
    records.append({
        "sample_id": "fall_author_forward_trial",
        "image_path": str(fall_image.relative_to(ROOT)),
        "original_path": str(fall_image.relative_to(ROOT)),
        "image_sha256": hashlib.sha256(fall_image.read_bytes()).hexdigest(),
        "original_source": "https://www.kaggle.com/datasets/patrickogbuitepu/posture-monitor-and-fall-detection",
        "original_file": "train/pose/sit-lean-forward.jpg",
        "original_split": "train",
        "license": {"name": "CC BY 4.0 (author declaration)", "url": "https://creativecommons.org/licenses/by/4.0/"},
        "proposed_label": "forward_lean", "proposed_view": "oblique",
        "proposed_action": "review", "reason": "Seated adult leans trunk forward with head approximately facing ahead. Camera is approximately oblique; angle not calibrated.",
        "reviewer": "assistant visual proposal", "human_confirmed": False, "human_label": None,
        "subject_id": "fall_author_single_participant",
        "group_id": "fall_author_single_participant",
    })
    report = {
        "scope": "Public RGB visual proposals and pose feasibility; no confirmed training or evaluation truth",
        "review_status": "Assistant proposals only; human confirmation pending",
        "human_confirmed_count": 0,
        "source_contact_sheets": ["runs/stage1/coco_chair_review/contact_{}.jpg".format(i) for i in range(1, 5)],
        "contact_sheet_candidates_seen": 73,
        "note": "The 11 listed examples are a selected review pack, not a count or label distribution of all 73 chair candidates. Geometric filters did not assign these proposals. No clear pure side_lean was confirmed. AI reinspection is not the independent human 20 percent review.",
        "records": records,
    }
    (destination / "review.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    manifest_path = ROOT / "data/processed/stage1_public_views_pilot.csv"
    fields = ["sample_id", "image_path", "label", "split", "subject_id", "video_id", "group_id", "view", "source", "timestamp_sec"]
    with manifest_path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            if record["proposed_label"] is None:
                continue
            image_path = ROOT / record["image_path"]
            writer.writerow({
                "sample_id": record["sample_id"],
                "image_path": os.path.relpath(image_path, manifest_path.parent),
                "label": record["proposed_label"], "split": "val",
                "subject_id": record.get("subject_id", ""),
                "group_id": record["group_id"], "view": record["proposed_view"],
                "source": "unconfirmed_public_pose_feasibility_only",
            })
    review_companion = {
        "confirmed_postures": False, "human_confirmed_count": 0,
        "purpose": "Real pose extraction and visibility only; do not train or score a classifier on these proposal labels",
        "review_file": str((destination / "review.json").relative_to(ROOT)),
        "local_split": "All locally reserved as val for feasibility; not a new train/val/test experiment. COCO original val retained; author fall image originally train, local subject reserved only for feasibility.",
    }
    manifest_path.with_suffix(".review.json").write_text(json.dumps(review_companion, ensure_ascii=False, indent=2) + "\n")
    cards = []
    for record in records:
        src = os.path.relpath(ROOT / record["image_path"], destination)
        original = os.path.relpath(ROOT / record["original_path"], destination)
        title = record["proposed_label"] or "建议剔除"
        cards.append('<article><a href="{original}"><img src="{src}" alt="{sample}"></a><div><h2>{sample}</h2><p><b>{title}</b> · {view}</p><p>{reason}</p><p class="small">{license}</p><p>人工确认：待完成</p></div></article>'.format(
            original=html.escape(original, quote=True), src=html.escape(src, quote=True),
            sample=html.escape(record["sample_id"]), title=html.escape(title),
            view=html.escape(record["proposed_view"]), reason=html.escape(record["reason"]),
            license=html.escape(record["license"]["name"])))
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>公开坐姿试标复核</title>
<style>body{font:16px/1.6 system-ui;margin:24px auto;max-width:1100px;padding:0 20px;background:#f7f7f4;color:#222}header{background:#fff2c9;padding:20px;border-radius:10px}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:18px;margin-top:24px}article{background:white;border:1px solid #ddd;border-radius:10px;overflow:hidden}img{width:100%;height:320px;object-fit:contain;background:#eee}article div{padding:14px}h2{font-size:16px;overflow-wrap:anywhere}.small{font-size:13px;color:#555}</style>
<header><h1>公开坐姿试标复核</h1><p>以下是助手的视觉提案，尚无人工确认，不能直接用于训练或计算分类精度。点击图片可看原图。</p><p>7 个待复核提案：端正 4、低头 1、前倾 2；侧倾仍缺少清晰样例。另列 4 个建议剔除例。复核时记录样本 ID、同意/修改/剔除及原因，正式标注须另外保留复核人和日期。</p></header><main>''' + "\n".join(cards) + "</main></html>"
    (destination / "review.html").write_text(page, encoding="utf-8")
    print(json.dumps({"review": str(destination / "review.html"), "manifest": str(manifest_path), "proposal_count": 7, "excluded_examples": 4}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
