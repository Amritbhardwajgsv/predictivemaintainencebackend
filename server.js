const express = require("express");
const { spawn } = require("child_process");
const cors = require("cors");
const path = require("path");

const app = express();

// Allow requests from the React frontend (Vite default port)
app.use(cors({ origin: "http://localhost:5173" }));
app.use(express.json());

// ─── POST /api/predict ───────────────────────────────────────────────────────
app.post("/api/predict", (req, res) => {
  const input = req.body;

const required = [
  "setting1", "setting2",
  "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10",
  "s11", "s12", "s13", "s14", "s15", "s16", "s17",
  "s18", "s19",
  "s20", "s21"
];

  const missing = required.filter((k) => input[k] === undefined);
  if (missing.length > 0) {
    return res.status(400).json({ error: `Missing fields: ${missing.join(", ")}` });
  }

  const py = spawn("python", [path.join(__dirname, "predict.py")]);

  let stdout = "";
  let stderr = "";

  py.stdin.write(JSON.stringify(input));
  py.stdin.end();

  py.stdout.on("data", (d) => (stdout += d.toString()));
  py.stderr.on("data", (d) => (stderr += d.toString()));

  py.on("close", (code) => {
    if (code !== 0) {
      console.error("Python stderr:", stderr);
      return res.status(500).json({ error: "Model inference failed", detail: stderr });
    }
    try {
      return res.json(JSON.parse(stdout.trim()));
    } catch {
      return res.status(500).json({ error: "Bad output from model", raw: stdout });
    }
  });
});

// ─── GET /api/health ─────────────────────────────────────────────────────────
app.get("/api/health", (_req, res) => res.json({ status: "ok" }));

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`✈️  Backend API running → http://localhost:${PORT}`);
});