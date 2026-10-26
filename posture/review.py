"""Local human annotation, with auditable revisions and development-only exports.

This module uses only the Python standard library. It never generates posture
labels and never modifies the project's training manifest.
"""

import copy
import csv
from datetime import datetime, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import os
from pathlib import Path
import re
import secrets
import tempfile
import threading
from urllib.parse import urlsplit

from .data import CLASS_NAMES, MANIFEST_FIELDS


LABEL_SCHEMAS = {
    "posture_v1": CLASS_NAMES,
    "posture_v2": CLASS_NAMES + ("back_lean",),
}
VIEWS = ("front", "oblique", "side", "unknown")
REASONS = ("uncertain", "not_seated", "compound", "occluded", "multiple_people", "other")
SCREENING_DECISIONS = ("keep_seated", "uncertain", "exclude_not_seated", "exclude_unsuitable")
DEV_NOTICE = (
    "All candidates and linked image/video/identity groups are development-only. "
    "Never assign them to a final test set. CSV split is intentionally unassigned. "
    "An annotator name and confirmation record a human declaration, not verified identity."
)
MAX_BODY_BYTES = 32768
IMAGE_TYPES = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _text(value, field, limit=200):
    if not isinstance(value, str) or len(value) > limit:
        raise ValueError("{} must be text of at most {} characters".format(field, limit))
    if any(ord(char) < 32 and char not in "\n\t" for char in value):
        raise ValueError("{} contains control characters".format(field))
    return value.strip()


class ReviewStore:
    """One candidate pool, one revision log; mismatched pools are never merged."""

    def __init__(self, candidates_path, output_path, screening_path=None, label_schema="posture_v1"):
        if label_schema not in LABEL_SCHEMAS:
            raise ValueError("Unknown posture label schema")
        self.label_schema = label_schema
        self.class_names = LABEL_SCHEMAS[label_schema]
        self.candidates_path = Path(candidates_path).expanduser().resolve()
        self.output_path = Path(output_path).expanduser().resolve()
        if self.candidates_path == self.output_path:
            raise ValueError("Review output must not overwrite the candidate file")
        raw = self.candidates_path.read_bytes()
        self.pool_sha256 = hashlib.sha256(raw).hexdigest()
        payload = json.loads(raw.decode("utf-8-sig"))
        entries = payload.get("entries")
        if not isinstance(entries, list):
            raise ValueError("Candidate file requires an entries array")
        self.candidates = []
        self.image_paths = {}
        ids = set()
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError("Candidate entries must be objects")
            if entry.get("status") != "downloaded":
                continue
            candidate = copy.deepcopy(entry)
            image = candidate.get("image", {})
            if not isinstance(image, dict):
                raise ValueError("Candidate image metadata must be an object")
            image_id = candidate.get("image_id", image.get("id"))
            annotation_id = candidate.get("annotation_id")
            sample_id = candidate.get("sample_id")
            if not sample_id and image_id is not None:
                sample_id = "coco_{}_a{}".format(image_id, annotation_id if annotation_id is not None else "whole")
            sample_id = _text(sample_id, "sample_id")
            if not sample_id or sample_id in ids:
                raise ValueError("Candidate sample_id is missing or duplicated: {}".format(sample_id))
            ids.add(sample_id)
            original = candidate.get("original_path") or candidate.get("local_path")
            crop = candidate.get("crop_path") or original
            if not original or not crop:
                raise ValueError("Candidate requires original_path/local_path and an image")
            resolved = {}
            for kind, value in (("original", original), ("crop", crop)):
                path = Path(value).expanduser()
                if not path.is_absolute():
                    path = self.candidates_path.parent / path
                path = path.resolve()
                if path.suffix.lower() not in IMAGE_TYPES or not path.is_file():
                    raise ValueError("Candidate image missing or unsupported: {}".format(path))
                resolved[kind] = path
            if self.output_path in resolved.values():
                raise ValueError("Review output must not overwrite a candidate image")
            candidate.update(
                sample_id=sample_id,
                image_id=image_id,
                original_path=str(resolved["original"]),
                crop_path=str(resolved["crop"]),
                source=candidate.get("source") or "coco",
                group_id=candidate.get("group_id") or "coco_image_{}".format(image_id if image_id is not None else sample_id),
                dev_only=True,
            )
            self.image_paths[len(self.candidates)] = resolved
            self.candidates.append(candidate)
        if not self.candidates:
            raise ValueError("Candidate pool has no successfully downloaded images")
        self.by_id = {item["sample_id"]: item for item in self.candidates}
        # Screening is read-only prioritization, kept outside the persisted human
        # review state so existing pool bindings and revision histories stay exact.
        self.screening = self._load_screening(screening_path)
        self.lock = threading.RLock()
        self._disk_sha256 = None
        self.state = {
            "schema_version": 1,
            "candidate_file": str(self.candidates_path),
            "candidate_pool_sha256": self.pool_sha256,
            "created_at": _now(),
            "updated_at": None,
            "dev_only": True,
            "notice": DEV_NOTICE,
            "candidates": copy.deepcopy(self.candidates),
            "records": {},
        }
        if self.label_schema == "posture_v2":
            self.state["label_schema_version"] = self.label_schema
        if self.output_path.exists():
            existing = self.output_path.read_bytes()
            state = json.loads(existing.decode("utf-8"))
            if state.get("label_schema_version", "posture_v1") != self.label_schema:
                raise ValueError("Existing review uses a different label schema; keep it unchanged and choose a new output to re-review every image under posture_v2")
            if state.get("schema_version") != 1 or state.get("candidate_pool_sha256") != self.pool_sha256:
                raise ValueError("Existing review belongs to a different candidate pool; choose a new output")
            saved_ids = [item.get("sample_id") for item in state.get("candidates", [])]
            if saved_ids != [item["sample_id"] for item in self.candidates] or state.get("dev_only") is not True:
                raise ValueError("Existing review has incompatible candidates or development policy")
            if not isinstance(state.get("records"), dict) or set(state["records"]).difference(self.by_id):
                raise ValueError("Existing review has invalid records")
            for sample_id, record in state["records"].items():
                history = record.get("history") if isinstance(record, dict) else None
                if not isinstance(history, list) or not history or record.get("latest") != history[-1]:
                    raise ValueError("Existing review has inconsistent revision history")
                for number, revision in enumerate(history, 1):
                    if self.label_schema == "posture_v2" and revision.get("label_schema_version") != self.label_schema:
                        raise ValueError("V2 review requires fresh V2 human decisions; old decisions cannot be relabelled as migrated")
                    self._validate_revision(revision)
                    if revision.get("revision") != number or not revision.get("saved_at"):
                        raise ValueError("Existing review has invalid revision numbering")
            self.state = state
            self._disk_sha256 = hashlib.sha256(existing).hexdigest()

    def _load_screening(self, screening_path):
        if screening_path is None:
            path = self.candidates_path.parent / "screening.json"
            if not path.exists():
                return None
        else:
            path = Path(screening_path).expanduser().resolve()
        raw = path.read_bytes()
        data = json.loads(raw.decode("utf-8-sig"))
        if not isinstance(data, dict) or data.get("schema_version") != 1:
            raise ValueError("Screening requires schema_version 1")
        if data.get("reviewer") != "assistant_visual_prescreen":
            raise ValueError("Screening reviewer must be assistant_visual_prescreen")
        if data.get("source_pool_sha256") != self.pool_sha256:
            raise ValueError("Screening source_pool_sha256 does not match this candidate pool")
        records = data.get("records")
        if not isinstance(records, list):
            raise ValueError("Screening requires a records array")
        seen = set()
        cleaned = []
        for record in records:
            if not isinstance(record, dict):
                raise ValueError("Screening records must be objects")
            if set(record).difference({"sample_id", "decision", "reason", "page"}):
                raise ValueError("Screening records may not contain posture labels or human annotations")
            sample_id = _text(record.get("sample_id"), "screening sample_id")
            if sample_id not in self.by_id or sample_id in seen:
                raise ValueError("Screening sample_id is unknown or duplicated: {}".format(sample_id))
            if record.get("decision") not in SCREENING_DECISIONS:
                raise ValueError("Invalid screening decision")
            reason = _text(record.get("reason", ""), "screening reason", 4000)
            page = record.get("page")
            if not isinstance(page, (str, int)) or isinstance(page, bool):
                raise ValueError("Screening page must be a page number or text reference")
            if isinstance(page, str):
                page = _text(page, "screening page", 500)
            seen.add(sample_id)
            cleaned.append({"sample_id": sample_id, "decision": record["decision"], "reason": reason, "page": page})
        return {
            "schema_version": 1,
            "reviewer": "assistant_visual_prescreen",
            "source_pool_sha256": self.pool_sha256,
            "screening_file": str(path.resolve()),
            "screening_sha256": hashlib.sha256(raw).hexdigest(),
            "read_only": True,
            "notice": "AI seating prescreen only; not human confirmation or a four-class posture label.",
            "records": cleaned,
        }

    def _validate_revision(self, value):
        if not isinstance(value, dict):
            raise ValueError("Annotation must be an object")
        status = value.get("status")
        if status not in ("accepted", "excluded", "pending"):
            raise ValueError("Invalid annotation status")
        label = value.get("label", "")
        if status == "accepted":
            if label not in self.class_names:
                raise ValueError("Accepted annotation requires a label from {}".format(self.label_schema))
        elif label:
            raise ValueError("Excluded or pending annotation must have no class label")
        if value.get("view") not in VIEWS:
            raise ValueError("Invalid camera view")
        if not _text(value.get("annotator"), "annotator", 100):
            raise ValueError("A human annotator name is required")
        if value.get("human_confirmed") is not True:
            raise ValueError("Explicit human confirmation is required")
        _text(value.get("note", ""), "note", 4000)
        reason = value.get("reason", "")
        if reason not in ("",) + REASONS:
            raise ValueError("Invalid exclusion/review reason")
        if status == "excluded" and not reason:
            raise ValueError("Excluded annotation requires a reason")
        if status == "accepted" and reason:
            raise ValueError("Accepted annotation must not include an exclusion reason")
        if value.get("split", "") not in ("",):
            raise ValueError("Pilot annotations must remain unassigned and development-only")

    def save(self, value):
        if not isinstance(value, dict):
            raise ValueError("Annotation must be an object")
        self._validate_revision(value)
        sample_id = _text(value.get("sample_id"), "sample_id")
        if sample_id not in self.by_id:
            raise ValueError("Unknown sample_id")
        expected = value.get("expected_revision")
        if not isinstance(expected, int) or isinstance(expected, bool) or expected < 0:
            raise ValueError("expected_revision must be a nonnegative integer")
        with self.lock:
            record = self.state["records"].get(sample_id, {"latest": None, "history": []})
            previous_revision = record["latest"]["revision"] if record["latest"] else 0
            if expected != previous_revision:
                raise ValueError("Annotation changed in another tab; reload before saving")
            revision = {
                key: value.get(key, "")
                for key in ("status", "label", "view", "annotator", "note", "reason")
            }
            revision["annotator"] = revision["annotator"].strip()
            revision.update(human_confirmed=True, annotator_type="human_declared", revision=previous_revision + 1, saved_at=_now())
            if self.label_schema == "posture_v2":
                revision["label_schema_version"] = self.label_schema
            updated = copy.deepcopy(self.state)
            updated["updated_at"] = revision["saved_at"]
            updated["records"][sample_id] = {
                "latest": revision,
                "history": copy.deepcopy(record["history"]) + [revision],
            }
            self._persist(updated)
            self.state = updated
            return copy.deepcopy(revision)

    def _persist(self, state):
        if self.output_path.exists():
            if self._disk_sha256 is None or hashlib.sha256(self.output_path.read_bytes()).hexdigest() != self._disk_sha256:
                raise ValueError("Review file changed externally; reload instead of overwriting it")
        elif self._disk_sha256 is not None:
            raise ValueError("Review file was removed externally; refusing silent replacement")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        data = _json_bytes(state)
        descriptor, name = tempfile.mkstemp(prefix=".review-", suffix=".tmp", dir=str(self.output_path.parent))
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(name, self.output_path)
            self._disk_sha256 = hashlib.sha256(data).hexdigest()
        finally:
            if os.path.exists(name):
                os.unlink(name)

    def snapshot(self):
        with self.lock:
            state = copy.deepcopy(self.state)
            if self.screening is not None:
                screening = copy.deepcopy(self.screening)
                decisions = {item["sample_id"]: item["decision"] for item in screening["records"]}
                seated = [item["sample_id"] for item in self.candidates if decisions.get(item["sample_id"]) == "keep_seated"]
                unreviewed = [sample_id for sample_id in seated if sample_id not in state["records"]]
                uncertain = [item["sample_id"] for item in self.candidates if decisions.get(item["sample_id"]) == "uncertain" and item["sample_id"] not in state["records"]]
                disagreement = [
                    item["sample_id"] for item in self.candidates
                    if decisions.get(item["sample_id"]) in ("uncertain", "exclude_not_seated", "exclude_unsuitable")
                    and state["records"].get(item["sample_id"], {}).get("latest", {}).get("status") == "accepted"
                ]
                screening["seated_sample_ids"] = seated
                screening["unreviewed_seated_sample_ids"] = unreviewed
                screening["unreviewed_uncertain_sample_ids"] = uncertain
                screening["accepted_disagreement_sample_ids"] = disagreement
                screening["counts"] = {
                    "original_candidates": len(self.candidates),
                    "screened": len(screening["records"]),
                    "keep_seated": len(seated),
                    "unreviewed_seated": len(unreviewed),
                    "accepted_disagreement": len(disagreement),
                }
                state["screening"] = screening
            return state

    def export_csv(self):
        """Only confirmed samples; V2 is versioned separately from training V1."""
        state = self.snapshot()
        stream = io.StringIO(newline="")
        fields = MANIFEST_FIELDS + (("label_schema_version",) if self.label_schema == "posture_v2" else ())
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for candidate in self.candidates:
            latest = state["records"].get(candidate["sample_id"], {}).get("latest", {})
            if latest.get("status") != "accepted" or latest.get("human_confirmed") is not True or latest.get("label") not in self.class_names:
                continue
            row = dict.fromkeys(MANIFEST_FIELDS, "")
            row.update(
                sample_id=candidate["sample_id"], image_path=candidate["crop_path"],
                label=latest["label"], group_id=candidate["group_id"],
                view=latest["view"], source=candidate["source"],
                subject_id=candidate.get("subject_id", ""), video_id=candidate.get("video_id", ""),
                timestamp_sec=candidate.get("timestamp_sec", ""),
            )
            if self.label_schema == "posture_v2":
                row["label_schema_version"] = self.label_schema
            writer.writerow(row)
        return stream.getvalue().encode("utf-8-sig")


HTML = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>坐姿候选人工标注</title>
<style>
[hidden]{display:none!important}
.image-stage{position:relative}.image-stage a,.image-stage img{display:block}.target-overlay{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}.target-overlay rect{fill:none;vector-effect:non-scaling-stroke;stroke-linejoin:round}.target-overlay .target-halo{stroke:white;stroke-width:5px}.target-overlay .target-box{stroke:#e52d35;stroke-width:2.5px}.target-note{display:block;font-size:12px;font-weight:400;color:#9e272c;margin-top:3px}
:root{color-scheme:light}*{box-sizing:border-box}body{font:15px/1.5 system-ui,"Microsoft YaHei",sans-serif;margin:0;background:#f4f6f8;color:#182536}header,main{max-width:1440px;margin:auto;padding:20px 28px}header{padding-bottom:8px}h1{font-size:25px;margin:0 0 6px}p{margin:6px 0}.muted{color:#556879;font-size:13px}.toolbar,.row,.choices{display:flex;gap:10px;align-items:center;flex-wrap:wrap}.toolbar{margin:16px 0}.toolbar label{display:flex;align-items:center;gap:8px}button,select,input,textarea{font:inherit;border:1px solid #b9c5d0;border-radius:6px;padding:9px 12px;background:white;color:inherit}button{cursor:pointer}button:hover{background:#eaf1f7}button:disabled{opacity:.5;cursor:default}.primary{background:#17699c;color:white;border-color:#17699c}.primary:hover{background:#10537e}input[type=radio],input[type=checkbox]{accent-color:#17699c}.layout{display:grid;grid-template-columns:minmax(0,1fr) 370px;gap:20px}.panel{background:white;border:1px solid #d4dde5;border-radius:10px;padding:18px}.images{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:14px}figure{margin:0}figcaption{font-weight:600;margin-bottom:8px}figure img{width:100%;height:430px;object-fit:contain;background:#eef1f4;border-radius:6px}h2{font-size:17px;margin:0 0 12px}.choices{flex-direction:column;align-items:stretch}.choice{display:flex;gap:8px;align-items:center;padding:9px;border:1px solid #dbe2e8;border-radius:6px}.choice:has(input:checked){background:#e6f2fa;border-color:#17699c}.field{display:block;margin:14px 0 5px}.full{width:100%}textarea{resize:vertical;min-height:72px}.stats{display:flex;flex-wrap:wrap;gap:8px}.stat{background:#e5edf3;padding:5px 10px;border-radius:5px}.notice{border-left:3px solid #ab7027;padding-left:12px;color:#765022}.error{color:#b12e2e}.ok{color:#17683c}#meta{overflow-wrap:anywhere;margin-top:12px}#message{min-height:24px;margin-top:12px}footer{max-width:1440px;margin:0 auto;padding:12px 28px 24px;color:#556879;font-size:13px}@media(max-width:1000px){.layout{grid-template-columns:1fr}.images img{height:330px}}@media(max-width:600px){header,main{padding:15px}.images{grid-template-columns:1fr}.images img{height:300px}}
</style></head><body>
<header><h1 id="reviewTitle">坐姿候选人工标注</h1><p><strong>四类仅用于坐着的人；站立请选择剔除。</strong>再观察头部与躯干。类别没有默认选项；不确定时请保留待定。</p>
<p class="notice">本批全部是开发用试标素材，原图及关联组不得进入最终测试。保存是你的人工确认，AI 提案不能代替人工标注。</p>
<p id="screeningNotice" class="notice" hidden>已按是否坐姿做 AI 视觉初筛，可能存在误判；它只调整优先审核队列，不提供四类坐姿标签。四类仍由你逐张确认，已有人工标注全部保留，可从“已标注”或“全部原始候选”回看。</p>
<p id="disagreementNotice" class="notice" hidden></p>
<p id="sourceLabelNotice" class="notice" hidden></p>
<div class="toolbar"><label>筛选 <select id="filter"><option value="screened_unreviewed" hidden disabled>待标坐姿（已初筛）</option><option value="screened_all" hidden disabled>初筛坐姿（全部）</option><option value="screened_uncertain" hidden disabled>初筛待定（未人工标注）</option><option value="screened_disagreement" hidden disabled>已标待回看</option><option value="all">全部原始候选</option><option value="unreviewed">全部未标注</option><option value="reviewed">已标注（全部）</option><option value="pending">人工待定</option></select></label><button id="prev">← 上一张</button><span id="position"></span><button id="next">下一张 →</button><button id="csv">导出已确认 CSV</button><button id="json">导出完整审计 JSON</button></div><div id="stats" class="stats"></div></header>
<main><div id="empty" class="panel" hidden>当前筛选下没有样本。可以切换筛选继续查看。</div><div class="layout" id="workspace"><section class="panel"><h2 id="sample"></h2><div class="images"><figure><figcaption>原始图像<span class="target-note" id="originalTargetNote"></span></figcaption><div class="image-stage"><a id="originalLink" target="_blank" rel="noopener"><img id="original" alt="候选所在原图"></a><svg id="originalOverlay" class="target-overlay" preserveAspectRatio="xMidYMid meet" aria-hidden="true" hidden><rect class="target-halo"></rect><rect class="target-box"></rect></svg></div></figure><figure><figcaption><span id="cropCaption">目标人物裁剪</span><span class="target-note" id="cropTargetNote"></span></figcaption><div class="image-stage"><a id="cropLink" target="_blank" rel="noopener"><img id="crop" alt="目标人物裁剪"></a><svg id="cropOverlay" class="target-overlay" preserveAspectRatio="xMidYMid meet" aria-hidden="true" hidden><rect class="target-halo"></rect><rect class="target-box"></rect></svg></div></figure></div><p class="muted" id="targetGuidance">请只标注红框内的目标人物，结合原图上下文判断。红框仅在审核页面叠加，不写入图像；点击图片打开的是无框原文件。纯侧面或复合姿势看不清时不要强行归类。</p><div id="meta" class="muted"></div></section>
<section class="panel"><h2>人工判断</h2><div id="choices" class="choices">
<label class="choice"><input type="radio" name="decision" value="upright">1　端正：头与躯干自然直立</label>
<label class="choice"><input type="radio" name="decision" value="head_down">2　低头：躯干基本端正，头颈下弯</label>
<label class="choice"><input type="radio" name="decision" value="forward_lean">3　上身前倾：躯干整体向前</label>
<label class="choice"><input type="radio" name="decision" value="side_lean">4　侧倾：躯干向左或右倾斜</label>
<label class="choice"><input type="radio" name="decision" value="excluded">X　剔除：不进入四类数据</label>
<label class="choice"><input type="radio" name="decision" value="pending">U　待定：保留后续复核</label></div>
<label class="field" for="reason">剔除或待定原因</label><select class="full" id="reason"><option value="">请选择原因</option><option value="uncertain">无法判断</option><option value="not_seated">非坐姿</option><option value="compound">复合姿势</option><option value="occluded">遮挡严重</option><option value="multiple_people">多人干扰</option><option value="other">其他（请备注）</option></select>
<label class="field" for="view">摄像视角</label><select class="full" id="view"><option value="unknown">未知 / 无法确认</option><option value="front">正面</option><option value="oblique">斜前方</option><option value="side">侧面</option></select>
<label class="field" for="annotator">标注人姓名或固定代号（必填）</label><input id="annotator" class="full" maxlength="100" autocomplete="off" placeholder="填写实际人工标注者">
<label class="field" for="note">备注</label><textarea id="note" class="full" maxlength="4000" placeholder="记录边界情况、歧义或修改原因"></textarea>
<label class="field"><input id="human" type="checkbox"> 我已亲自查看图像并作出以上判断</label>
<div class="row" style="margin-top:16px"><button id="save" class="primary">保存并下一张</button><button id="saveStay">仅保存</button></div><div id="message" aria-live="polite"></div><p id="history" class="muted"></p></section></div></main>
<footer>快捷键：1–4 选择类别，X 剔除，U 待定，←/→ 切换；输入文字时快捷键不生效。未保存切换会提醒。CSV 只包含已确认的四类记录，split 留空；完整审计 JSON 保留全部候选、修订与开发用途限制，请与 CSV 一起保存。</footer>
<script>
"use strict";
const $=id=>document.getElementById(id), classNames={upright:"端正",head_down:"低头",forward_lean:"前倾",side_lean:"侧倾"};
let state=null,filtered=[],current=null,dirty=false,token="",saving=false,filterValue="all",filterInitialized=false;
const inputs=[...document.querySelectorAll('input[name="decision"]')];
function msg(text,error=false){$("message").textContent=text;$("message").className=error?"error":"ok";}
function latest(id){return state.records[id]?.latest||null;}
function prepareFilters(){const enabled=!!state.screening;for(const option of $("filter").options){if(option.value.startsWith("screened_")){option.disabled=!enabled;option.hidden=!enabled;}}$("screeningNotice").hidden=!enabled;const conflicts=state.screening?.counts.accepted_disagreement||0;$("disagreementNotice").hidden=!conflicts;$("disagreementNotice").textContent=`坐姿初筛与 ${conflicts} 条已有标注存在分歧，请从“已标待回看”检查；原人工结果未修改，也不会因 AI 初筛而自动排除导出。`;if(!filterInitialized){filterValue=enabled?"screened_unreviewed":"all";$("filter").value=filterValue;filterInitialized=true;}}
function applyFilter(){const f=$("filter").value,screening=state.screening||{},queues={screened_unreviewed:screening.unreviewed_seated_sample_ids||[],screened_all:screening.seated_sample_ids||[],screened_uncertain:screening.unreviewed_uncertain_sample_ids||[],screened_disagreement:screening.accepted_disagreement_sample_ids||[]},queue=Object.hasOwn(queues,f)?new Set(queues[f]):null;filtered=state.candidates.map((c,i)=>i).filter(i=>{const id=state.candidates[i].sample_id,r=latest(id);return queue?queue.has(id):f==="all"||(f==="unreviewed"&&!r)||(f==="reviewed"&&!!r)||(f==="pending"&&r?.status==="pending");});}
function stats(){const counts={upright:0,head_down:0,forward_lean:0,side_lean:0,excluded:0,pending:0},records=Object.values(state.records);for(const r of records){const v=r.latest;if(v.status==="accepted")counts[v.label]++;else counts[v.status]++;}const screening=state.screening?.counts,summary=[`原始总数 ${state.candidates.length}`];if(screening)summary.push(`本次初筛通过 ${screening.keep_seated}`,`待标坐姿 ${screening.unreviewed_seated}`,`已标待回看 ${screening.accepted_disagreement}`);summary.push(`已人工记录 ${records.length}`,`全部未标 ${state.candidates.length-records.length}`,...Object.entries(classNames).map(([key,name])=>`已确认${name} ${counts[key]}`),`人工剔除 ${counts.excluded}`,`人工待定 ${counts.pending}`);$("stats").replaceChildren();for(const text of summary){const s=document.createElement("span");s.className="stat";s.textContent=text;$("stats").appendChild(s);}}
function showSourceContext(candidate){const imported=candidate.source==="classmate_provided";const unboxed=!Array.isArray(candidate.bbox)||candidate.bbox.length!==4;$("reviewTitle").textContent=imported?"同学数据人工复核":"坐姿候选人工标注";document.title=imported?"同学数据人工复核":"坐姿候选人工标注";$("sourceLabelNotice").hidden=!imported;const label=String(candidate.source_label||"未记录"),name=classNames[label];$("sourceLabelNotice").textContent=imported?`同学原始标签：${name?name+"（"+label+"）":label}。来自同学人工标注，本次需你逐张确认；原标签不自动作为复核结果。`:"";$("cropCaption").textContent=(imported||unboxed)?"原图参考（待确认目标人物）":"目标人物裁剪";$("crop").alt=(imported||unboxed)?"原图参考，目标人物待确认":"目标人物裁剪";$("targetGuidance").textContent=(imported||unboxed)?"这批暂未确认目标人物框，左右均展示原图。多人图请在备注中说明你判断的目标人物；目标不明确时请选择待定。点击图片可打开原尺寸查看。":"请只标注红框内的目标人物，结合原图上下文判断。红框仅在审核页面叠加，不写入图像；点击图片打开的是无框原文件。纯侧面或复合姿势看不清时不要强行归类。";return imported;}
function showImage(kind,index,candidate){const imported=showSourceContext(candidate),img=$(kind),overlay=$(kind+"Overlay"),note=$(kind+"TargetNote"),url=`/image/${index}/${kind}`;const box=kind==="original"?candidate.bbox:candidate.target_bbox_in_crop;const valid=Array.isArray(box)&&box.length===4&&box.every(Number.isFinite)&&box[2]>0&&box[3]>0;overlay.setAttribute("hidden", "");note.textContent=valid?"红框内为本次标注目标":imported?"尚无确认目标框，请结合原图和备注确认人物":"此记录未提供目标框，请在备注中说明人物；不明确时选待定";const draw=()=>{if(current!==index||!valid||!img.naturalWidth||!img.naturalHeight)return;overlay.setAttribute("viewBox",`0 0 ${img.naturalWidth} ${img.naturalHeight}`);for(const rect of overlay.querySelectorAll("rect")){["x","y","width","height"].forEach((attribute,i)=>rect.setAttribute(attribute,String(box[i])));}overlay.removeAttribute("hidden");};img.onload=draw;img.onerror=()=>{if(current===index){overlay.setAttribute("hidden", "");note.textContent="图像加载失败，请刷新后重试";}};img.src=url;$(kind+"Link").href=url;if(img.complete)draw();}
function show(index){current=index;dirty=false;msg("");const exists=index!==undefined&&index!==null;$("empty").hidden=exists;$("workspace").hidden=!exists;if(!exists){$("position").textContent="0 / 0";$("prev").disabled=true;$("next").disabled=true;return;}const c=state.candidates[index],r=latest(c.sample_id),pos=filtered.indexOf(index);$("position").textContent=`${pos+1} / ${filtered.length}`;$("prev").disabled=pos<=0;$("next").disabled=pos>=filtered.length-1;$("sample").textContent=c.sample_id;for(const kind of ["original","crop"])showImage(kind,index,c);const license=typeof c.license==="object"?(c.license?.name||JSON.stringify(c.license)):(c.license||"未记录");$("meta").textContent=`来源：${c.source}　原始划分：${c.original_split||"未记录"}　图像：${c.image_id??"未知"}　人物：${c.annotation_id??"整图"}\n分组：${c.group_id}　许可：${license}`;for(const input of inputs)input.checked=r?input.value===(r.status==="accepted"?r.label:r.status):false;$("view").value=r?.view||"unknown";$("reason").value=r?.reason||"";$("note").value=r?.note||"";$("human").checked=false;$("history").textContent=r?`已保存第 ${r.revision} 版 · ${r.annotator} · ${r.saved_at}`:"尚未标注";}
function mayLeave(){return !dirty||window.confirm("当前修改尚未保存，确定切换并放弃修改？");}
function move(delta){if(!mayLeave())return;const p=filtered.indexOf(current)+delta;if(p>=0&&p<filtered.length)show(filtered[p]);}
async function refresh(){const response=await fetch("/api/state",{cache:"no-store"});if(!response.ok)throw Error("读取候选失败");state=await response.json();token=state.csrf_token;prepareFilters();stats();applyFilter();}
async function save(advance){if(saving||current===null)return;const choice=inputs.find(i=>i.checked)?.value;if(!choice){msg("请选择类别、剔除或待定。",true);return;}if(!$("annotator").value.trim()||!$("human").checked){msg("请填写标注人，并确认亲自查看了图像。",true);return;}const index=current,oldPosition=filtered.indexOf(index),c=state.candidates[index],r=latest(c.sample_id),accepted=Object.hasOwn(classNames,choice);const payload={sample_id:c.sample_id,status:accepted?"accepted":choice,label:accepted?choice:"",view:$("view").value,reason:accepted?"":$("reason").value,annotator:$("annotator").value.trim(),note:$("note").value,human_confirmed:true,expected_revision:r?.revision||0};saving=true;$("save").disabled=true;$("saveStay").disabled=true;try{const response=await fetch("/api/review",{method:"POST",headers:{"Content-Type":"application/json","X-Review-Token":token},body:JSON.stringify(payload)});const result=await response.json();if(!response.ok)throw Error(result.error||"保存失败");dirty=false;await refresh();const same=filtered.indexOf(index);const next=advance?(same>=0?filtered[Math.min(same+1,filtered.length-1)]:filtered[Math.min(oldPosition,filtered.length-1)]):(same>=0?index:filtered[0]);show(next);msg("已保存，修订历史已保留。");}catch(error){msg(error.message,true);}finally{saving=false;$("save").disabled=false;$("saveStay").disabled=false;}}
for(const element of [...inputs,$("reason"),$("view"),$("note"),$("human")])element.addEventListener("change",()=>dirty=true);$("note").addEventListener("input",()=>dirty=true);$("annotator").addEventListener("input",()=>dirty=true);
$("prev").onclick=()=>move(-1);$("next").onclick=()=>move(1);$("save").onclick=()=>save(true);$("saveStay").onclick=()=>save(false);$("filter").onchange=()=>{if(!mayLeave()){$("filter").value=filterValue;return;}filterValue=$("filter").value;applyFilter();show(filtered[0]);};
$("csv").onclick=()=>{if(!mayLeave())return;window.location.href="/api/export.csv";};$("json").onclick=()=>{if(!mayLeave())return;window.location.href="/api/export.json";};
document.addEventListener("keydown",event=>{const target=event.target;const typing=["TEXTAREA","SELECT"].includes(target.tagName)||(target.tagName==="INPUT"&&!["radio","checkbox"].includes(target.type));if(event.ctrlKey||event.altKey||event.metaKey||typing)return;const mapping={"1":"upright","2":"head_down","3":"forward_lean","4":"side_lean","x":"excluded","u":"pending"},choice=mapping[event.key.toLowerCase()];if(choice){inputs.find(i=>i.value===choice).checked=true;dirty=true;event.preventDefault();}else if(event.key==="ArrowLeft"){move(-1);event.preventDefault();}else if(event.key==="ArrowRight"){move(1);event.preventDefault();}});
window.addEventListener("beforeunload",event=>{if(dirty){event.preventDefault();event.returnValue="";}});
refresh().then(()=>show(filtered[0])).catch(error=>msg(error.message,true));
</script></body></html>'''


def _review_html(store):
    html = HTML
    if store.label_schema == "posture_v2":
        html = html.replace("四类", "五类").replace("1–4", "1–5")
        html = html.replace('value="excluded">X', 'value="back_lean">5　后仰：躯干明显向后倾，接触椅背本身不算后仰</label>\n<label class="choice"><input type="radio" name="decision" value="excluded">X')
        html = html.replace('side_lean:"侧倾"}', 'side_lean:"侧倾",back_lean:"后仰"}')
        html = html.replace('side_lean:0,excluded:0', 'side_lean:0,back_lean:0,excluded:0')
        html = html.replace('"4":"side_lean","x"', '"4":"side_lean","5":"back_lean","x"')
        html = html.replace("再观察头部与躯干。", "先判断躯干方向，再判断头部；躯干有明显倾斜时优先选躯干类别。")
        notice = "posture_v2 五类新复核：旧四类决策未复制为新确认，每张图需重新查看。训练流程仍为四类，当前导出不可直接作为五类训练输入。"
    else:
        notice = "posture_v1 四类历史标准；五类 posture_v2 必须使用新的输出文件逐图重新复核。"
    return html.replace('<p class="notice">', '<p class="notice">' + notice + '</p>\n<p class="notice">', 1)


def create_server(store, port=8765):
    """Bind exclusively to loopback; every image route maps to a known file."""
    token = secrets.token_urlsafe(32)

    class Handler(BaseHTTPRequestHandler):
        server_version = "PostureReview/1.0"

        def log_message(self, format, *args):
            pass

        def _valid_host(self):
            return self.headers.get("Host") == "127.0.0.1:{}".format(self.server.server_port)

        def _send(self, status, body, content_type="application/json; charset=utf-8", filename=None):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Content-Security-Policy", "default-src 'none'; img-src 'self'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; connect-src 'self'; base-uri 'none'; frame-ancestors 'none'; form-action 'none'")
            if filename:
                self.send_header("Content-Disposition", 'attachment; filename="{}"'.format(filename))
            self.end_headers()
            self.wfile.write(body)

        def _error(self, status, message):
            self._send(status, _json_bytes({"error": message}))

        def do_GET(self):
            if not self._valid_host():
                self._error(403, "Invalid Host header")
                return
            path = urlsplit(self.path).path
            if path == "/":
                self._send(200, _review_html(store).encode("utf-8"), "text/html; charset=utf-8")
            elif path == "/api/state":
                result = store.snapshot()
                result["csrf_token"] = token
                self._send(200, _json_bytes(result))
            elif path == "/api/export.json":
                self._send(200, _json_bytes(store.snapshot()), filename="review.dev-only.json")
            elif path == "/api/export.csv":
                self._send(200, store.export_csv(), "text/csv; charset=utf-8", "review.accepted.dev-only.csv")
            else:
                match = re.fullmatch(r"/image/(\d+)/(original|crop)", path)
                if not match:
                    self._error(404, "Unknown endpoint")
                    return
                image_path = store.image_paths.get(int(match.group(1)), {}).get(match.group(2))
                if image_path is None or not image_path.is_file() or image_path.resolve() != image_path:
                    self._error(404, "Unknown or unavailable image")
                    return
                self._send(200, image_path.read_bytes(), IMAGE_TYPES[image_path.suffix.lower()])

        def do_POST(self):
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self._error(400, "Invalid Content-Length")
                return
            if not 0 < length <= MAX_BODY_BYTES or self.headers.get("Transfer-Encoding"):
                # A small oversize request must receive its 413 on Windows too.
                # Drain only a bounded, declared body; never read arbitrary sizes.
                if 0 < length <= 2 * MAX_BODY_BYTES and not self.headers.get("Transfer-Encoding"):
                    try:
                        self.connection.settimeout(2)
                        self.rfile.read(length)
                    except OSError:
                        pass
                self._error(413, "Request body is missing or too large")
                return
            try:
                # Drain only a bounded body before returning a rejection. Closing
                # with unread request bytes can reset Windows TCP connections.
                self.connection.settimeout(5)
                raw = self.rfile.read(length)
                if len(raw) != length:
                    raise ValueError("Incomplete request body")
            except (OSError, ValueError):
                self._error(400, "Cannot read complete request body")
                return
            origin = "http://127.0.0.1:{}".format(self.server.server_port)
            if not self._valid_host() or self.headers.get("Origin") != origin or not secrets.compare_digest(self.headers.get("X-Review-Token", ""), token):
                self._error(403, "Only same-origin, confirmed review requests are accepted")
                return
            if urlsplit(self.path).path != "/api/review":
                self._error(404, "Unknown endpoint")
                return
            if self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower() != "application/json":
                self._error(415, "Content-Type must be application/json")
                return
            try:
                value = json.loads(raw.decode("utf-8"))
                revision = store.save(value)
            except (ValueError, UnicodeError) as error:
                self._error(400, str(error))
                return
            except OSError:
                self._error(500, "Cannot persist review; no saved confirmation was returned")
                return
            self._send(200, _json_bytes({"saved": True, "revision": revision}))

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)
