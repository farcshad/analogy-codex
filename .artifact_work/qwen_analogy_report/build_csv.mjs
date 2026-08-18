import fs from "node:fs/promises";
import path from "node:path";
import { Workbook } from "@oai/artifact-tool";

const repoRoot = path.resolve("../..");
const outputDir = path.join(repoRoot, "reports");
const conditionPath = path.join(
  repoRoot,
  "gpu_experiments/pipeline_runs/teacher-deepseek-v4-flash__student-Qwen-Qwen3-0.6B_condition_4.jsonl",
);
const baselinePath = path.join(
  repoRoot,
  "gpu_experiments/pipeline_runs/teacher-deepseek-v4-flash__student-Qwen-Qwen3-0.6B_condition_20.jsonl",
);

const parseJsonl = async (filePath) =>
  (await fs.readFile(filePath, "utf8"))
    .split(/\r?\n/)
    .filter(Boolean)
    .map((line) => JSON.parse(line));

const conditionRows = (await parseJsonl(conditionPath)).filter(
  (row) => row.record_type === "result" && !row.error && typeof row.is_correct === "boolean",
);
const baselineById = new Map(
  (await parseJsonl(baselinePath))
    .filter((row) => row.record_type === "result" && !row.error && typeof row.is_correct === "boolean")
    .map((row) => [row.id, row]),
);

// This reproduces the broad, explicit analogy-use definition used in the analysis.
const teacherOrAnalogy = /\b(?:teacher|analog(?:y|ies|ical))\b/i;
const comparisonLanguage = /\b(?:metaphor|just as|in the same way|is like|works like|similar to|comparable to|maps? (?:to|onto))\b/i;

const stopwords = new Set(
  "the and with from into like using used use that this where when while which your their each same different more less first second three two one analogy analogies analogous example system method process model rule concept effect problem question illustrates illustrated showing shows teacher explained explanation domain correct answer option based given highlights aligns represented represents corresponds similar making make made acts acting works".split(" "),
);

function sourceTitleOverlap(row) {
  const headings = [...(row.teaching_content || "").matchAll(/(?:\*\*)?Analogy\s*\d+\s*:\s*(?:\*\*)?([^\n*]+)/gim)]
    .map((match) => match[1].replace(/\([^)]*\)/g, " "))
    .join(" ");
  const titleTokens = new Set(
    (headings.match(/[A-Za-z][A-Za-z'-]+/g) || [])
      .map((word) => word.toLowerCase())
      .filter((word) => word.length >= 4 && !stopwords.has(word)),
  );
  const targetText = `${row.question_stem || ""} ${row.choices || ""} ${row.scientific_concept || ""}`.toLowerCase();
  const targetTokens = new Set(targetText.match(/[A-Za-z][A-Za-z'-]+/g) || []);
  const reasonTokens = new Set((row.reason || "").toLowerCase().match(/[A-Za-z][A-Za-z'-]+/g) || []);
  return [...titleTokens].filter((word) => !targetTokens.has(word) && reasonTokens.has(word)).sort();
}

const selected = conditionRows
  .filter((row) => teacherOrAnalogy.test(row.reason || "") || comparisonLanguage.test(row.reason || ""))
  .sort((a, b) => a.id.localeCompare(b.id));

function transition(conditionCorrect, baseline) {
  if (!baseline) return "baseline_unavailable";
  if (!baseline.is_correct && conditionCorrect) return "wrong_to_right";
  if (baseline.is_correct && !conditionCorrect) return "right_to_wrong";
  if (baseline.is_correct && conditionCorrect) return "both_right";
  return "both_wrong";
}

const headers = [
  "request_key",
  "id",
  "scientific_concept",
  "question_stem",
  "choices",
  "answer_key",
  "teaching_content_3x200",
  "qwen_prediction",
  "qwen_reason",
  "qwen_is_correct",
  "explicit_analogy_use",
  "source_title_overlap_count",
  "source_title_overlap_terms",
  "strong_mapping_evidence_ge_2_terms",
  "baseline_answer_available",
  "baseline_prediction",
  "baseline_reason",
  "baseline_is_correct",
  "outcome_transition",
  "analogy_fixed_baseline_error",
  "analogy_broke_baseline_correct",
];

const data = selected.map((row) => {
  const baseline = baselineById.get(row.id);
  const overlaps = sourceTitleOverlap(row);
  const outcome = transition(row.is_correct, baseline);
  return [
    row.request_key,
    row.id,
    row.scientific_concept || "",
    row.question_stem || "",
    row.choices || "",
    row.answer_key || "",
    row.teaching_content || "",
    row.prediction || "",
    row.reason || "",
    row.is_correct,
    true,
    overlaps.length,
    overlaps.join(" | "),
    overlaps.length >= 2,
    Boolean(baseline),
    baseline?.prediction || "",
    baseline?.reason || "",
    baseline?.is_correct ?? "",
    outcome,
    outcome === "wrong_to_right",
    outcome === "right_to_wrong",
  ];
});

if (selected.length !== 200) throw new Error(`Expected 200 selected answers; found ${selected.length}`);

// Build an in-memory spreadsheet for structural and visual QA before CSV export.
const workbook = Workbook.create();
const sheet = workbook.worksheets.add("Analogy-use answers");
sheet.getRangeByIndexes(0, 0, data.length + 1, headers.length).values = [headers, ...data];
sheet.freezePanes.freezeRows(1);
sheet.showGridLines = false;
sheet.getRangeByIndexes(0, 0, 1, headers.length).format = {
  fill: "#17365D",
  font: { bold: true, color: "#FFFFFF" },
  wrapText: true,
  rowHeight: 32,
};
sheet.getRangeByIndexes(1, 0, data.length, headers.length).format.wrapText = true;
sheet.getRange("A:U").format.autofitColumns();
for (const col of [2, 3, 4, 6, 8, 12, 16, 18]) {
  sheet.getRangeByIndexes(0, col, data.length + 1, 1).format.columnWidth = col === 6 ? 55 : 32;
}
sheet.getRangeByIndexes(1, 0, data.length, headers.length).format.rowHeight = 48;
sheet.tables.add(`A1:U${data.length + 1}`, true, "QwenAnalogyUseTable");

const inspection = await workbook.inspect({
  kind: "table",
  range: "'Analogy-use answers'!A1:U6",
  include: "values,formulas",
  tableMaxRows: 6,
  tableMaxCols: 21,
  maxChars: 6000,
});
console.log(inspection.ndjson);

const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 50 },
  summary: "final formula error scan",
});
console.log(errors.ndjson);

const preview = await workbook.render({
  sheetName: "Analogy-use answers",
  range: "A1:U8",
  scale: 1,
  format: "png",
});
await fs.writeFile(path.join(repoRoot, ".artifact_work/qwen_analogy_report/preview.png"), new Uint8Array(await preview.arrayBuffer()));

function csvCell(value) {
  if (value === null || value === undefined) return "";
  const text = typeof value === "boolean" ? (value ? "true" : "false") : String(value);
  return /[",\r\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

const csv = [headers, ...data].map((row) => row.map(csvCell).join(",")).join("\r\n") + "\r\n";
await fs.mkdir(outputDir, { recursive: true });
const csvPath = path.join(outputDir, "qwen3_0.6b_no_thinking_condition4_analogy_use_200.csv");
await fs.writeFile(csvPath, `\uFEFF${csv}`, "utf8");

const counts = data.reduce((acc, row) => {
  acc[row[18]] = (acc[row[18]] || 0) + 1;
  return acc;
}, {});
console.log(JSON.stringify({ csvPath, rows: data.length, transitions: counts }, null, 2));
