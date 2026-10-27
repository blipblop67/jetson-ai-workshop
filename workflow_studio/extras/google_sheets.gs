/**
 * AI Workflow Studio -> Google Sheets
 * -----------------------------------
 * 1. Open your Google Sheet > Extensions > Apps Script. Delete what's there and paste this file.
 * 2. Click Deploy > New deployment > type "Web app".
 *      Execute as: Me        Who has access: Anyone
 * 3. Click Deploy, allow access, and copy the "Web app URL" (ends with /exec).
 * 4. Paste that URL into the Google Sheets block in the studio.
 * Each message from the Jetson adds one row.
 */
function doPost(e) {
  var sheet = SpreadsheetApp.getActiveSpreadsheet().getActiveSheet();
  var d = JSON.parse(e.postData.contents);
  if (sheet.getLastRow() === 0) {
    sheet.appendRow(["Time", "Workflow", "Reason", "Total objects", "Counts", "Label", "Metrics"]);
  }
  sheet.appendRow([d.time, d.workflow, d.reason, d.total, d.counts, d.label || "", d.metrics]);
  return ContentService.createTextOutput("ok");
}
