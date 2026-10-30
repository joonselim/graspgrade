// Video check jobs, stored in Neon Postgres. The page uploads a video to Vercel Blob
// (api/validate-upload.js), then this API keeps the job list. A worker on the team's
// own computer (web/worker/worker.py) takes jobs, runs the GraspGrade pipeline with the
// Meta API key it holds, and reports back. The site never holds the Meta API key.
//
// Page (cookie gg_validate, set by middleware.js after the VALIDATE_PASSWORD login):
//   GET  /api/validate                          jobs, worker heartbeat, budget
//   POST /api/validate {action:"create", url, pathname, filename, size}
//   POST /api/validate {action:"confirm", id}   start a job after its cost estimate
//   POST /api/validate {action:"cancel", id}    or remove a finished one; deletes the video
// Worker (header x-validate-token, the same hash as the cookie):
//   POST /api/validate {action:"next"}          heartbeat, and the next job to work on
//   POST /api/validate {action:"update", id, status, ...fields}

import { neon } from "@neondatabase/serverless";
import { del } from "@vercel/blob";
import { createHash } from "crypto";

const SALT = "graspgrade-validate-v1";
const BUDGET = Number(process.env.VALIDATE_BUDGET_USD || 5); // Meta API dollars for all checks together
const WORKER_FIELDS = ["status", "duration_s", "est_cost", "cost", "progress", "result", "error"];
const WORKER_STATUS = new Set(["estimated", "running", "done", "failed"]);

let ready = null;
function db() {
  const sql = neon(process.env.DATABASE_URL || process.env.POSTGRES_URL);
  if (!ready) {
    ready = sql`create table if not exists validate_jobs (
      id bigserial primary key, created_at timestamptz not null default now(),
      updated_at timestamptz not null default now(), status text not null,
      filename text, size bigint, url text, pathname text,
      duration_s real, est_cost real, cost real, progress text, result jsonb, error text)`
      .then(() => sql`create table if not exists validate_worker (
        id int primary key, seen_at timestamptz not null, host text)`);
  }
  return ready.then(() => sql);
}

function expected() {
  const password = process.env.VALIDATE_PASSWORD;
  return password ? createHash("sha256").update(SALT + password).digest("hex") : null;
}

function isPage(req) {
  const want = expected();
  const cookie = (req.headers.cookie || "").split(/;\s*/).find((c) => c.startsWith("gg_validate="));
  return Boolean(want && cookie) && cookie.slice("gg_validate=".length) === want;
}

function isWorker(req) {
  const want = expected();
  return Boolean(want) && req.headers["x-validate-token"] === want;
}

async function budget(sql) {
  const [r] = await sql`select
      coalesce(sum(cost), 0)::float as used,
      coalesce(sum(est_cost) filter (where status in ('confirmed', 'running') and cost is null), 0)::float as reserved
    from validate_jobs`;
  return { limit: BUDGET, used: r.used, reserved: r.reserved };
}

async function state(sql) {
  const jobs = await sql`select id, created_at, updated_at, status, filename, size, url, duration_s,
      est_cost, cost, progress, result, error from validate_jobs
    where status <> 'removed' order by id desc limit 20`;
  const [w] = await sql`select seen_at, host from validate_worker where id = 1`;
  return { jobs, worker: w || null, budget: await budget(sql), now: new Date().toISOString() };
}

async function dropVideo(url) {
  if (!url) return;
  try { await del(url); } catch (e) { /* already gone */ }
}

export default async function handler(req, res) {
  res.setHeader("cache-control", "no-store");
  const page = isPage(req), worker = !page && isWorker(req);
  if (!page && !worker) return res.status(401).json({ error: "unauthorized" });
  const sql = await db();

  if (req.method === "GET") {
    if (!page) return res.status(401).json({ error: "unauthorized" });
    return res.json(await state(sql));
  }
  if (req.method !== "POST") return res.status(405).json({ error: "method" });
  const b = req.body || {};

  if (page && b.action === "create") {
    if (!b.url || !/^https:\/\/[^/]+\.blob\.vercel-storage\.com\//.test(b.url)) return res.status(400).json({ error: "url" });
    await sql`insert into validate_jobs (status, filename, size, url, pathname)
      values ('queued', ${String(b.filename || "").slice(0, 200)}, ${Number(b.size) || null}, ${b.url}, ${b.pathname || null})`;
    return res.json(await state(sql));
  }

  if (page && b.action === "confirm") {
    const [job] = await sql`select id, status, est_cost from validate_jobs where id = ${b.id}`;
    if (!job || job.status !== "estimated") return res.status(409).json({ error: "not waiting for confirmation" });
    const m = await budget(sql);
    if (m.used + m.reserved + (job.est_cost || 0) > m.limit) {
      return res.status(409).json({ error: "budget", budget: m });
    }
    await sql`update validate_jobs set status = 'confirmed', updated_at = now() where id = ${b.id} and status = 'estimated'`;
    return res.json(await state(sql));
  }

  if (page && b.action === "cancel") {
    const [job] = await sql`select id, status, url from validate_jobs where id = ${b.id}`;
    if (!job) return res.status(404).json({ error: "no job" });
    if (job.status === "running") return res.status(409).json({ error: "running" });
    await dropVideo(job.url);
    await sql`update validate_jobs set status = 'removed', url = null, updated_at = now() where id = ${b.id}`;
    return res.json(await state(sql));
  }

  if (worker && b.action === "next") {
    await sql`insert into validate_worker (id, seen_at, host) values (1, now(), ${String(b.host || "").slice(0, 80)})
      on conflict (id) do update set seen_at = now(), host = excluded.host`;
    // A queued video is probed and priced first; a confirmed one is run.
    const [job] = await sql`update validate_jobs set
        status = case status when 'queued' then 'probing' else 'running' end, updated_at = now()
      where id = (select id from validate_jobs where status in ('queued', 'confirmed')
                  order by id limit 1 for update skip locked)
      returning id, status, url, filename, est_cost`;
    return res.json({ job: job || null, budget: await budget(sql) });
  }

  if (worker && b.action === "update") {
    const set = {};
    for (const k of WORKER_FIELDS) if (k in b) set[k] = b[k];
    if (set.status && !WORKER_STATUS.has(set.status)) return res.status(400).json({ error: "status" });
    const [job] = await sql`select id from validate_jobs where id = ${b.id}`;
    if (!job) return res.status(404).json({ error: "no job" });
    await sql`update validate_jobs set
        status = coalesce(${set.status ?? null}, status),
        duration_s = coalesce(${set.duration_s ?? null}, duration_s),
        est_cost = coalesce(${set.est_cost ?? null}, est_cost),
        cost = coalesce(${set.cost ?? null}, cost),
        progress = coalesce(${set.progress ?? null}, progress),
        result = coalesce(${set.result ? JSON.stringify(set.result) : null}::jsonb, result),
        error = coalesce(${set.error ?? null}, error),
        updated_at = now()
      where id = ${b.id}`;
    return res.json({ ok: true });
  }

  return res.status(400).json({ error: "action" });
}
