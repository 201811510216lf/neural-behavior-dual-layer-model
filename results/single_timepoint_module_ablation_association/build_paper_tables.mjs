import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const root = "D:/video/outputs/single_timepoint_module_ablation_association";
const outputPath = path.join(root, "paper_tables.xlsx");
const previewDir = path.join(root, "qa", "workbook_previews");
const fontFamily = "Arial";

function parseCsv(text) {
  const rows = [];
  let row = [];
  let field = "";
  let quoted = false;
  for (let i = 0; i < text.length; i += 1) {
    const ch = text[i];
    if (quoted) {
      if (ch === '"' && text[i + 1] === '"') {
        field += '"';
        i += 1;
      } else if (ch === '"') {
        quoted = false;
      } else {
        field += ch;
      }
    } else if (ch === '"') {
      quoted = true;
    } else if (ch === ",") {
      row.push(field);
      field = "";
    } else if (ch === "\n") {
      row.push(field.replace(/\r$/, ""));
      rows.push(row);
      row = [];
      field = "";
    } else {
      field += ch;
    }
  }
  if (field.length || row.length) {
    row.push(field.replace(/\r$/, ""));
    rows.push(row);
  }
  const filtered = rows.filter((r) => r.some((v) => v !== ""));
  if (filtered.length && filtered[0].length) filtered[0][0] = filtered[0][0].replace(/^\uFEFF/, "");
  return filtered;
}

function typedValue(value, header) {
  if (value === "") return null;
  if (value === "True") return true;
  if (value === "False") return false;
  const textHeaders = new Set([
    "condition_id", "comparison_id", "effect_id", "effect_type", "module",
    "stage", "source_stage", "source_stage_class", "architecture", "metric",
    "label_name", "behavior_label", "display_label", "interaction_sign",
    "saved_prediction_file", "data_vector_json", "module_vector_json",
    "checkpoint_path", "sha256", "recording_id", "recording_path",
  ]);
  if (textHeaders.has(header) || header.endsWith("_id") || header.endsWith("_path") || header.endsWith("_json")) {
    return value;
  }
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : value;
}

async function loadCsv(relativePath) {
  const text = await fs.readFile(path.join(root, relativePath), "utf8");
  const raw = parseCsv(text);
  const headers = raw[0];
  return [headers, ...raw.slice(1).map((row) => headers.map((h, i) => typedValue(row[i] ?? "", h)))];
}

function colName(index) {
  let n = index + 1;
  let name = "";
  while (n > 0) {
    const rem = (n - 1) % 26;
    name = String.fromCharCode(65 + rem) + name;
    n = Math.floor((n - 1) / 26);
  }
  return name;
}

function styleDataSheet(sheet, matrix, title, sourcePath, tableName, tabColor = null) {
  const rows = matrix.length;
  const cols = matrix[0].length;
  const endCol = colName(cols - 1);
  sheet.showGridLines = false;
  if (tabColor) sheet.tabColor = tabColor;
  sheet.getRange("A2").values = [[title]];
  sheet.getRange("A2").format.font = { name: fontFamily, size: 14, bold: true, color: "#172554" };
  sheet.getRange("A3").values = [[`Source: ${sourcePath}`]];
  sheet.getRange("A3").format.font = { name: fontFamily, size: 9, italic: true, color: "#64748B" };
  sheet.getRangeByIndexes(4, 0, rows, cols).values = matrix;
  const header = sheet.getRange(`A5:${endCol}5`);
  header.format = {
    fill: "#1E3A5F",
    font: { name: fontFamily, size: 10, bold: true, color: "#FFFFFF" },
    horizontalAlignment: "center",
    verticalAlignment: "center",
  };
  const body = sheet.getRangeByIndexes(5, 0, Math.max(rows - 1, 1), cols);
  body.format.font = { name: fontFamily, size: 10, color: "#111827" };
  body.format.verticalAlignment = "center";
  body.format.rowHeight = 18;
  sheet.getRangeByIndexes(4, 0, rows, cols).format.borders = {
    insideHorizontal: { style: "thin", color: "#E2E8F0" },
    bottom: { style: "thin", color: "#CBD5E1" },
  };
  const headers = matrix[0];
  for (let c = 0; c < cols; c += 1) {
    const h = String(headers[c] ?? "");
    let maxLen = h.length;
    for (let r = 1; r < Math.min(rows, 80); r += 1) {
      maxLen = Math.max(maxLen, String(matrix[r][c] ?? "").length);
    }
    let width = Math.max(9, Math.min(24, maxLen + 2));
    if (h.includes("path") || h.includes("json")) width = 36;
    if (h === "y") width = 40;
    if (h.includes("inference")) width = 48;
    if (h.includes("condition_id") || h.includes("comparison_id")) width = 23;
    sheet.getRange(`${colName(c)}:${colName(c)}`).format.columnWidth = width;
  }
  const table = sheet.tables.add(`A5:${endCol}${rows + 4}`, true, tableName);
  table.style = "TableStyleMedium2";
  table.showBandedColumns = false;
  table.showFilterButton = true;
  sheet.freezePanes.freezeRows(5);
  sheet.freezePanes.freezeColumns(Math.min(2, cols));
}

const workbook = Workbook.create();
const summary = workbook.worksheets.add("Summary");
summary.showGridLines = false;
summary.tabColor = "#172554";
summary.getRange("A2").values = [["单时间点神经数据消融 × 模块消融：论文汇总"]];
summary.getRange("A2").format.font = { name: fontFamily, size: 15, bold: true, color: "#172554" };
summary.getRange("A3").values = [["Frozen-model inference only；未重新训练、未重新划分；legacy 5 s / 10 ms / 500 bins / 250-bin step。"]];
summary.getRange("A3").format.font = { name: fontFamily, size: 10, italic: true, color: "#475569" };

const summaryRows = [
  ["项目", "结果", "口径 / 解释"],
  ["显著时间点", 24, "Summary_P_long；window_ms=10, p<0.05, label 1..5, time [-1,1]"],
  ["结构模型", 4, "Full、NoConv、NoLSTM、NoAttention；coherent 3-fold checkpoints"],
  ["阶段", 2, "Run1 / Run2 stage-wise，保持原 test indices"],
  ["条件级预测行", 119300, "clean + point mask + module + double ablation"],
  ["配对效应行", 204003, "相同 OOF 样本上的 ΔPtrue / Δlogit-margin 等"],
  ["最强单点 ΔRecall", -6.313, "tp24_L5_p0p70 → L5，percentage points"],
  ["最强模块 ΔRecall", -14.810, "NoConv → L4，percentage points"],
  ["最大 |Recall-vector Spearman|", 0.821, "tp01_L1_m0p69 × NoConv"],
  ["最大 CW Jaccard", 0.065, "tp05_L2_p0p06 × NoLSTM"],
  ["最大样本级 |Spearman|", 0.154, "tp12_L4_m0p56 × NoConv；Run2；ΔPtrue"],
  ["最强 double interaction", 4.261, "tp19_L5_p0p08 × NoAttention × L5，percentage points"],
  ["高重叠 mask 对", 52, "Run1 非对角 Jaccard ≥ 0.90；解读单点独立性时必须谨慎"],
];
summary.getRangeByIndexes(5, 0, summaryRows.length, 3).values = summaryRows;
summary.getRange("A6:C6").format = {
  fill: "#1E3A5F",
  font: { name: fontFamily, size: 10, bold: true, color: "#FFFFFF" },
  horizontalAlignment: "center",
};
summary.getRange("A7:C18").format.font = { name: fontFamily, size: 10, color: "#111827" };
summary.getRange("A6:C18").format.borders = {
  insideHorizontal: { style: "thin", color: "#E2E8F0" },
  bottom: { style: "thin", color: "#CBD5E1" },
};
summary.getRange("A:A").format.columnWidth = 32;
summary.getRange("B:B").format.columnWidth = 50;
summary.getRange("C:C").format.columnWidth = 56;
summary.getRange("B7:B11").format.numberFormat = "0";
summary.getRange("B12:B17").format.numberFormat = "0.000";
summary.getRange("B18").format.numberFormat = "0";
summary.getRange("A20").values = [["解释边界"]];
summary.getRange("A20").format.font = { name: fontFamily, size: 12, bold: true, color: "#172554" };
summary.getRange("A21:C23").values = [
  ["Correlation / similarity", "描述两个效应模式的相似性，不等于模块介导。", "详见 A–E 表"],
  ["Double-ablation interaction", "difference-in-differences；反映在冻结模型中的非加性依赖。", "详见 F Interaction"],
  ["Causal biological interpretation", "本分析不能单独支持神经生物学因果结论；重叠窗口也不是独立生物重复。", "CI 使用 recording-level/block bootstrap（能恢复 recording 时）"],
];
summary.getRange("A21:C23").format.font = { name: fontFamily, size: 10, color: "#111827" };
summary.getRange("A21:A23").format.font = { name: fontFamily, size: 10, bold: true, color: "#7C2D12" };
summary.getRange("A21:C23").format.wrapText = true;
summary.getRange("A21:C23").format.rowHeight = 58;

const specs = [
  ["Timepoints", "source_data/significant_timepoints_24.csv", "24 significant timepoints", "TimepointsTable", "#0F766E"],
  ["A RecallSimilarity", "tables/A_recall_vector_similarity.csv", "A. 5-behavior ΔRecall vector similarity", "RecallSimilarityTable", "#2563EB"],
  ["B SampleSpearman", "tables/B_sample_effect_spearman.csv", "B. OOF sample-level ΔPtrue / Δlogit-margin Spearman", "SampleSpearmanTable", "#2563EB"],
  ["C SetAssociation", "tables/C_damage_recovery_set_association.csv", "C. CW/WC set association", "SetAssociationTable", "#2563EB"],
  ["D ErrorDelta", "tables/D_canonical_behavior_delta_fp_fn.csv", "D. One-vs-rest ΔFP / ΔFN", "ErrorDeltaTable", "#2563EB"],
  ["E ConfusionSimilarity", "tables/E_confusion_delta_offdiagonal_similarity.csv", "E. Off-diagonal confusion-delta similarity", "ConfusionSimilarityTable", "#2563EB"],
  ["F Interaction", "tables/F_double_ablation_interaction_recall.csv", "F. Double-ablation interaction in per-class recall", "InteractionTable", "#7C3AED"],
  ["G TtestAssociation", "tables/G_ttest_vs_data_ablation_correlations.csv", "G. Exploratory t-stat / group-direction association", "TtestAssociationTable", "#7C3AED"],
  ["ConditionMetrics", "tables/condition_metrics_stagewise.csv", "Condition-level stage-wise metrics", "ConditionMetricsTable", "#64748B"],
];

for (const [name, rel, title, tableName, color] of specs) {
  const sheet = workbook.worksheets.add(name);
  const matrix = await loadCsv(rel);
  styleDataSheet(sheet, matrix, title, rel, tableName, color);
}

const qa = workbook.worksheets.add("QA");
qa.showGridLines = false;
qa.tabColor = "#B45309";
qa.getRange("A2").values = [["Quality assurance and reproducibility checks"]];
qa.getRange("A2").format.font = { name: fontFamily, size: 14, bold: true, color: "#172554" };
const qaSpecs = [
  ["Frozen Full baseline reproduction", "qa/clean_baseline_reproduction.csv"],
  ["Coherent structural metric reproduction", "qa/structural_clean_metric_reproduction.csv"],
  ["Fold test-index alignment", "qa/fold_test_index_alignment.csv"],
  ["Dataset and mask equivalence", "qa/dataset_and_mask_equivalence.csv"],
  ["Timepoint mask coverage", "qa/timepoint_mask_coverage.csv"],
];
let startRow = 5;
for (const [sectionTitle, rel] of qaSpecs) {
  const matrix = await loadCsv(rel);
  const cols = matrix[0].length;
  const endCol = colName(cols - 1);
  qa.getRange(`A${startRow}`).values = [[sectionTitle]];
  qa.getRange(`A${startRow}`).format.font = { name: fontFamily, size: 11, bold: true, color: "#92400E" };
  qa.getRangeByIndexes(startRow, 0, matrix.length, cols).values = matrix;
  qa.getRange(`A${startRow + 1}:${endCol}${startRow + 1}`).format = {
    fill: "#78350F",
    font: { name: fontFamily, size: 10, bold: true, color: "#FFFFFF" },
    horizontalAlignment: "center",
  };
  qa.getRangeByIndexes(startRow + 1, 0, matrix.length, cols).format.font = { name: fontFamily, size: 10, color: "#111827" };
  qa.getRangeByIndexes(startRow + 1, 0, matrix.length, cols).format.borders = {
    insideHorizontal: { style: "thin", color: "#E2E8F0" },
    bottom: { style: "thin", color: "#CBD5E1" },
  };
  startRow += matrix.length + 3;
}
qa.getUsedRange().format.verticalAlignment = "center";
qa.getUsedRange().format.autofitColumns();
qa.getRange("A:A").format.columnWidth = 28;
qa.getRange("B:B").format.columnWidth = 22;
qa.getRange("C:C").format.columnWidth = 42;
qa.getRange("D:F").format.columnWidth = 34;
qa.getRange("G:G").format.columnWidth = 22;
qa.getRange("H:H").format.columnWidth = 42;
qa.getRange("I:J").format.columnWidth = 22;
qa.freezePanes.freezeRows(6);

workbook.recalculate();
const inspection = await workbook.inspect({
  kind: "workbook,sheet,table",
  maxChars: 12000,
  tableMaxRows: 4,
  tableMaxCols: 8,
  tableMaxCellChars: 80,
});
await fs.mkdir(previewDir, { recursive: true });
const previewNames = ["Summary", ...specs.map((s) => s[0]), "QA"];
for (const sheetName of previewNames) {
  const preview = await workbook.render({ sheetName, autoCrop: "all", scale: 0.75, format: "png" });
  const safeName = sheetName.replace(/[^A-Za-z0-9]+/g, "_");
  await fs.writeFile(path.join(previewDir, `${safeName}.png`), new Uint8Array(await preview.arrayBuffer()));
}
const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputPath);
console.log(JSON.stringify({ outputPath, previewDir, inspection: inspection.ndjson }, null, 2));
