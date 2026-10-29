// Temporary: says which variables the functions can see (true or false only, never values).
export default function handler(req, res) {
  res.setHeader("cache-control", "no-store");
  const names = ["LABEL_PASSWORD", "DATABASE_URL", "POSTGRES_URL"];
  const seen = Object.fromEntries(names.map((n) => [n, Boolean(process.env[n])]));
  res.status(200).json({ env: process.env.VERCEL_ENV || null, ...seen });
}
