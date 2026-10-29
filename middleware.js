// Password gate for the labeling page, its media and its API.
// The password lives in the LABEL_PASSWORD environment variable, never in the repository.

export const config = { matcher: ["/label", "/label.html", "/label-media/:path*", "/api/label"] };

const SALT = "graspgrade-label-v1";

async function token(password) {
  const d = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(SALT + password));
  return [...new Uint8Array(d)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

function loginPage(wrong) {
  const html = `<!DOCTYPE html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><meta name="robots" content="noindex, nofollow">
<meta name="color-scheme" content="light only"><title>GraspGrade 라벨링</title>
<style>
body{margin:0;min-height:100vh;display:grid;place-items:center;background:#eef0f3;color:#1a1d23;
font-family:-apple-system,BlinkMacSystemFont,"Apple SD Gothic Neo","Malgun Gothic","Noto Sans KR",sans-serif}
form{background:#fff;border:1px solid #dde1e7;padding:28px;width:min(320px,calc(100vw - 32px));box-sizing:border-box}
h1{font-size:18px;margin:0 0 14px}input{width:100%;box-sizing:border-box;font:inherit;padding:9px 10px;border:1px solid #c7cdd6;border-radius:4px}
button{margin-top:10px;width:100%;font:inherit;font-weight:600;padding:9px;border:0;border-radius:4px;background:#2350c9;color:#fff;cursor:pointer}
p{font-size:13px;color:#d0342c;margin:10px 0 0}
</style></head><body><form method="post" action="/label">
<h1>GraspGrade 라벨링</h1><input type="password" name="password" placeholder="비밀번호" autofocus required>
<button type="submit">들어가기</button>${wrong ? "<p>비밀번호가 맞지 않음</p>" : ""}</form></body></html>`;
  return new Response(html, {
    status: wrong ? 401 : 200,
    headers: { "content-type": "text/html; charset=utf-8", "cache-control": "no-store", "x-robots-tag": "noindex, nofollow" },
  });
}

export default async function middleware(request) {
  const password = process.env.LABEL_PASSWORD;
  if (!password) return new Response("LABEL_PASSWORD is not set", { status: 503 });
  const want = await token(password);
  const url = new URL(request.url);
  const cookie = (request.headers.get("cookie") || "").split(/;\s*/).find((c) => c.startsWith("gg_label="));
  if (cookie && cookie.slice("gg_label=".length) === want) return; // continue to the page, media or API

  const page = url.pathname === "/label" || url.pathname === "/label.html";
  if (!page) return new Response("Unauthorized", { status: 401, headers: { "cache-control": "no-store" } });
  if (request.method === "POST") {
    const form = await request.formData();
    if (form.get("password") === password) {
      return new Response(null, {
        status: 303,
        headers: {
          location: "/label",
          "set-cookie": `gg_label=${want}; Path=/; HttpOnly; Secure; SameSite=Lax; Max-Age=1209600`,
          "cache-control": "no-store",
        },
      });
    }
    return loginPage(true);
  }
  return loginPage(false);
}
