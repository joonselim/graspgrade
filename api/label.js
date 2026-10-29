// Labels for the soft object check (splits/softness_labels.csv), stored in Neon Postgres.
//   GET  /api/label?who=l1        items, my labels, progress; both labelers' labels once both are done
//   POST /api/label {who,id,label} save one label ("" removes it); who "agreed" only after both are done
//   GET  /api/label?export=1      the sheet as CSV, columns as in splits/softness_labels.csv

import { neon } from "@neondatabase/serverless";
import { createHash } from "crypto";
import { readFileSync } from "fs";
import { join } from "path";

const ITEMS = JSON.parse(readFileSync(join(process.cwd(), "label-media", "items.json"), "utf8"));
const IDS = new Set(ITEMS.map((i) => i.id));
const LABELERS = ["l1", "l2"];
const VALUES = new Set(["rigid", "squeezable", "soft"]);
const SALT = "graspgrade-label-v1";

let ready = null;
function db() {
  const sql = neon(process.env.DATABASE_URL || process.env.POSTGRES_URL);
  if (!ready) {
    ready = sql`create table if not exists labels (
      item text not null, labeler text not null, label text not null,
      updated_at timestamptz not null default now(), primary key (item, labeler))`;
  }
  return ready.then(() => sql);
}

function authorized(req) {
  const password = process.env.LABEL_PASSWORD;
  if (!password) return false;
  const want = createHash("sha256").update(SALT + password).digest("hex");
  const cookie = (req.headers.cookie || "").split(/;\s*/).find((c) => c.startsWith("gg_label="));
  return Boolean(cookie) && cookie.slice("gg_label=".length) === want;
}

async function state(sql) {
  const rows = await sql`select item, labeler, label from labels`;
  const by = { l1: {}, l2: {}, agreed: {} };
  for (const r of rows) if (by[r.labeler] && IDS.has(r.item)) by[r.labeler][r.item] = r.label;
  const counts = { l1: Object.keys(by.l1).length, l2: Object.keys(by.l2).length };
  const reveal = counts.l1 === ITEMS.length && counts.l2 === ITEMS.length;
  return { by, counts, reveal };
}

function csv(by) {
  const head = "dataset,object,tune,held_out,labeler_1,labeler_2,agreed";
  const q = (s) => (/[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s);
  const lines = ITEMS.map((i) => {
    const a = by.l1[i.id] || "", b = by.l2[i.id] || "";
    const agreed = by.agreed[i.id] || (a && a === b ? a : "");
    return [i.dataset, q(i.object), i.tune, i.held_out, a, b, agreed].join(",");
  });
  return [head, ...lines].join("\n") + "\n";
}

export default async function handler(req, res) {
  res.setHeader("cache-control", "no-store");
  if (!authorized(req)) return res.status(401).json({ error: "unauthorized" });
  const sql = await db();

  if (req.method === "GET") {
    const { by, counts, reveal } = await state(sql);
    if (req.query.export) {
      res.setHeader("content-type", "text/csv; charset=utf-8");
      res.setHeader("content-disposition", 'attachment; filename="softness_labels.csv"');
      return res.status(200).send(csv(by));
    }
    const who = String(req.query.who || "");
    return res.status(200).json({
      items: ITEMS,
      counts,
      reveal,
      mine: LABELERS.includes(who) ? by[who] : {},
      all: reveal ? by : null,
    });
  }

  if (req.method === "POST") {
    const body = typeof req.body === "string" ? JSON.parse(req.body || "{}") : req.body || {};
    const { who, id, label } = body;
    if (![...LABELERS, "agreed"].includes(who) || !IDS.has(id) || !(label === "" || VALUES.has(label))) {
      return res.status(400).json({ error: "bad request" });
    }
    if (who === "agreed" && !(await state(sql)).reveal) {
      return res.status(409).json({ error: "both labelers must finish first" });
    }
    if (label === "") await sql`delete from labels where item = ${id} and labeler = ${who}`;
    else await sql`insert into labels (item, labeler, label) values (${id}, ${who}, ${label})
                   on conflict (item, labeler) do update set label = excluded.label, updated_at = now()`;
    return res.status(200).json({ ok: true });
  }

  return res.status(405).json({ error: "method not allowed" });
}
