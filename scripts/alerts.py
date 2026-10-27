"""Step 4: write a Bangla alert for every ward at yellow or above, plus an
English rendering of the template (alert_en) so the dashboard can show
non-Bangla readers what residents receive.

Default: deterministic templates built only from model outputs and the
guidance file, so no number is ever invented. With --llm and an
ANTHROPIC_API_KEY, orange and red alerts are reworded by an LLM that sees the
same facts and guidance and is told not to add anything new (simple RAG:
the retrieved context is the guidance lines that match the ward's drivers
and level).

Also writes one CAP 1.2 XML file per orange/red ward to docs/data/cap/.
"""
import argparse
import json
import os
from datetime import datetime, timezone
from xml.sax.saxutils import escape

import geopandas as gpd
import requests

import config

BN_DIGITS = str.maketrans("0123456789", "০১২৩৪৫৬৭৮৯")
BN_MONTHS = ["জানুয়ারি", "ফেব্রুয়ারি", "মার্চ", "এপ্রিল", "মে", "জুন", "জুলাই",
             "আগস্ট", "সেপ্টেম্বর", "অক্টোবর", "নভেম্বর", "ডিসেম্বর"]


def bn(x):
    return str(x).translate(BN_DIGITS)


def bn_time(iso):
    """'2026-10-19T21:00' -> '১৯ অক্টোবর, রাত ৯টা'"""
    t = datetime.fromisoformat(iso) if isinstance(iso, str) else iso  # readers may parse it already
    h = t.hour
    part = ("ভোর" if 4 <= h < 6 else "সকাল" if 6 <= h < 12 else
            "দুপুর" if 12 <= h < 15 else "বিকেল" if 15 <= h < 18 else
            "সন্ধ্যা" if 18 <= h < 20 else "রাত")
    h12 = h % 12 or 12
    return f"{bn(t.day)} {BN_MONTHS[t.month - 1]}, {part} {bn(h12)}টা"


def as_list(v):
    """GeoJSON readers may hand back the drivers field as text or as a list."""
    if isinstance(v, str):
        return json.loads(v)
    return [str(x) for x in list(v)] if v is not None else []


def template_alert(w, guide):
    drivers = as_list(w["drivers"])
    # keep an SMS short: two reasons, three actions
    reasons = [guide["drivers"][d] for d in drivers if d in guide["drivers"]][:2]
    acts = guide["actions"].get(w["level"], [])[:3]
    lines = [
        f"{w['name']}: {w['level_bn']}",
        f"আগামী ৭২ ঘণ্টায় জলাবদ্ধতার সম্ভাবনা প্রায় {bn(round(w['prob'] * 100))}%। "
        f"সবচেয়ে বেশি বৃষ্টি {bn_time(w['peak_time'])} নাগাদ "
        f"(২৪ ঘণ্টায় প্রায় {bn(round(w['p24_max']))} মিমি)।",
    ]
    if reasons:
        lines.append("কারণ: " + "; ".join(reasons) + "।")
    if acts:
        lines.append("করণীয়: " + " | ".join(acts))
    return "\n".join(lines)


def en_time(iso):
    """'2026-10-19T21:00' -> '19 Oct, 9 pm'"""
    t = datetime.fromisoformat(iso) if isinstance(iso, str) else iso
    return f"{t.day} {t:%b}, {t.hour % 12 or 12} {'am' if t.hour < 12 else 'pm'}"


def template_alert_en(w, guide):
    """English rendering of template_alert for the dashboard; same facts, same order."""
    drivers = as_list(w["drivers"])
    reasons = [guide["drivers_en"][d] for d in drivers if d in guide["drivers_en"]][:2]
    acts = guide["actions_en"].get(w["level"], [])[:3]
    lines = [
        f"{w['name']}: {w['level'].capitalize()}",
        f"Chance of waterlogging in the next 72 hours: about {round(w['prob'] * 100)}%. "
        f"Heaviest rain around {en_time(w['peak_time'])} "
        f"(about {round(w['p24_max'])} mm in 24 hours).",
    ]
    if reasons:
        lines.append("Why: " + "; ".join(reasons) + ".")
    if acts:
        lines.append("What to do: " + " | ".join(acts))
    return "\n".join(lines)


def llm_rewrite(text, w, guide):
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return None
    context = {"level": w["level"],
               "actions": guide["actions"].get(w["level"], []),
               "reasons": [guide["drivers"][d] for d in as_list(w["drivers"])
                           if d in guide["drivers"]]}
    prompt = (
        "You write public flood alerts in Bangla for residents of Dhaka. "
        "Rewrite the draft alert below so it reads naturally and calmly for an SMS, "
        "under 300 characters if possible. Use ONLY the facts in the draft and the "
        "guidance context. Do not add, change or round any number, time or place. "
        "Keep Bengali digits. Return only the alert text.\n\n"
        f"GUIDANCE CONTEXT:\n{json.dumps(context, ensure_ascii=False)}\n\n"
        f"DRAFT:\n{text}")
    r = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={"x-api-key": key, "anthropic-version": "2023-06-01",
                 "content-type": "application/json"},
        json={"model": config.LLM_MODEL, "max_tokens": 400,
              "messages": [{"role": "user", "content": prompt}]},
        timeout=60)
    r.raise_for_status()
    return r.json()["content"][0]["text"].strip()


def cap_xml(w, text, sent):
    severity = {"red": "Severe", "orange": "Moderate"}.get(w["level"], "Minor")
    poly = w.geometry if w.geometry.geom_type == "Polygon" else max(
        w.geometry.geoms, key=lambda p: p.area)
    coords = " ".join(f"{y:.5f},{x:.5f}" for x, y in poly.exterior.coords)
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<alert xmlns="urn:oasis:names:tc:emergency:cap:1.2">
  <identifier>bonnaalert-{w['ward_id']}-{sent.replace(':', '').replace('-', '')}</identifier>
  <sender>bonnaalert</sender>
  <sent>{sent}</sent>
  <status>Exercise</status>
  <msgType>Alert</msgType>
  <scope>Public</scope>
  <info>
    <language>bn-BD</language>
    <category>Met</category>
    <event>Urban flooding / waterlogging</event>
    <urgency>Expected</urgency>
    <severity>{severity}</severity>
    <certainty>{"Likely" if w["prob"] >= 0.5 else "Possible"}</certainty>
    <headline>{escape(w['name'] + ': ' + w['level_bn'])}</headline>
    <description>{escape(text)}</description>
    <area>
      <areaDesc>{escape(w['name'])}, Dhaka</areaDesc>
      <polygon>{coords}</polygon>
    </area>
  </info>
</alert>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", action="store_true",
                    help="reword orange/red alerts with an LLM (needs ANTHROPIC_API_KEY)")
    args = ap.parse_args()

    guide = json.loads(config.GUIDANCE.read_text(encoding="utf-8"))
    g = gpd.read_file(config.WARDS_SCORED)
    raw = json.loads(config.WARDS_SCORED.read_text(encoding="utf-8"))
    # newer pyogrio drops list-valued fields ("unsupported OGR type: 5"), so take drivers from the raw file
    raw_drivers = {f["properties"]["ward_id"]: f["properties"].get("drivers") for f in raw["features"]}
    g["drivers"] = g["ward_id"].map(raw_drivers)
    sent = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")
    cap_dir = config.WEB_DATA / "cap"
    cap_dir.mkdir(parents=True, exist_ok=True)
    for f in cap_dir.glob("*.xml"):
        f.unlink()

    texts, texts_en, n_llm = [], [], 0
    for _, w in g.iterrows():
        if w["level"] == "green":
            texts.append("")
            texts_en.append("")
            continue
        text = template_alert(w, guide)
        texts_en.append(template_alert_en(w, guide))
        if args.llm and w["level"] in ("orange", "red"):
            try:
                better = llm_rewrite(text, w, guide)
                if better:
                    text, n_llm = better, n_llm + 1
            except Exception as e:  # keep the template if the API fails
                print(f"LLM failed for {w['ward_id']}: {e}")
        texts.append(text)
        if w["level"] in ("orange", "red"):
            (cap_dir / f"{w['ward_id']}.xml").write_text(cap_xml(w, text, sent), encoding="utf-8")

    # patch the raw GeoJSON so other fields keep the exact form score.py wrote
    by_id = dict(zip(g["ward_id"], texts))
    en_by_id = dict(zip(g["ward_id"], texts_en))
    for f in raw["features"]:
        f["properties"]["alert_bn"] = by_id.get(f["properties"]["ward_id"], "")
        f["properties"]["alert_en"] = en_by_id.get(f["properties"]["ward_id"], "")
    config.WARDS_SCORED.write_text(json.dumps(raw, ensure_ascii=False, separators=(",", ":")),
                                   encoding="utf-8")
    n = sum(1 for t in texts if t)
    print(f"Alerts written for {n} wards ({n_llm} reworded by LLM); CAP files in {cap_dir}")
    first = next((t for t in texts if t), None)
    if first:
        print("\nExample:\n" + first)


if __name__ == "__main__":
    main()
