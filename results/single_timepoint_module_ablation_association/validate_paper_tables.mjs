import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const workbookPath = "D:/video/outputs/single_timepoint_module_ablation_association/paper_tables.xlsx";
const sheetNames = [
  "Summary", "Timepoints", "A RecallSimilarity", "B SampleSpearman",
  "C SetAssociation", "D ErrorDelta", "E ConfusionSimilarity",
  "F Interaction", "G TtestAssociation", "ConditionMetrics", "QA",
];
const input = await FileBlob.load(workbookPath);
const workbook = await SpreadsheetFile.importXlsx(input);
workbook.recalculate();
const errors = [];
const sheets = [];
for (const name of sheetNames) {
  const sheet = workbook.worksheets.getItem(name);
  const used = sheet.getUsedRange();
  const values = used.values;
  for (let r = 0; r < values.length; r += 1) {
    for (let c = 0; c < values[r].length; c += 1) {
      const value = values[r][c];
      if (typeof value === "string" && /^#(REF!|DIV\/0!|VALUE!|NAME\?|N\/A|NUM!|NULL!)/.test(value)) {
        errors.push({ sheet: name, row: r + 1, col: c + 1, value });
      }
    }
  }
  sheets.push({ name, rows: values.length, cols: values[0]?.length ?? 0 });
}
const keyChecks = {
  summaryTitle: workbook.worksheets.getItem("Summary").getRange("A2").values[0][0],
  timepointCount: workbook.worksheets.getItem("Timepoints").getRange("A6:A29").values.filter((r) => r[0] !== null).length,
  firstHeader: workbook.worksheets.getItem("Timepoints").getRange("A5").values[0][0],
  ttestRowCount: workbook.worksheets.getItem("G TtestAssociation").getRange("A6:A11").values.filter((r) => r[0] !== null).length,
};
console.log(JSON.stringify({ workbookPath, sheets, keyChecks, formulaErrors: errors }, null, 2));
