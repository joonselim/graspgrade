// Client upload to Vercel Blob for the video check page. The browser asks this route
// for a short-lived upload token, then sends the video straight to Blob storage, so
// videos larger than the 4.5 MB function body limit still work. Vercel Blob calls this
// route again when the upload finishes; that call is checked by its signature.

import { handleUpload } from "@vercel/blob/client";
import { createHash } from "crypto";

const SALT = "graspgrade-validate-v1";
const MAX_BYTES = 200 * 1024 * 1024;

function isPage(req) {
  const password = process.env.VALIDATE_PASSWORD;
  if (!password) return false;
  const want = createHash("sha256").update(SALT + password).digest("hex");
  const cookie = (req.headers.cookie || "").split(/;\s*/).find((c) => c.startsWith("gg_validate="));
  return Boolean(cookie) && cookie.slice("gg_validate=".length) === want;
}

export default async function handler(req, res) {
  res.setHeader("cache-control", "no-store");
  if (req.method !== "POST") return res.status(405).json({ error: "method" });
  try {
    const json = await handleUpload({
      body: req.body,
      request: req,
      onBeforeGenerateToken: async () => {
        if (!isPage(req)) throw new Error("unauthorized");
        return {
          allowedContentTypes: ["video/mp4", "video/quicktime", "video/webm", "video/x-matroska"],
          maximumSizeInBytes: MAX_BYTES,
          addRandomSuffix: true,
        };
      },
      onUploadCompleted: async () => {},
    });
    return res.json(json);
  } catch (e) {
    return res.status(400).json({ error: String(e.message || e) });
  }
}
