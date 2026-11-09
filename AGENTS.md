# Reports

- Write all reports in LaTeX and save their `.tex` sources in `docs/reports/`. Name each source `YYYY-MM-DD-report-topic.tex`.
- Keep at most the two newest reports in `docs/reports/`. Use the report date first and the source modification time to break ties. Remove older reports and files used only by them.
- Save report assets under `docs/reports/`.
- Generate or compile PDF versions only when the user explicitly requests them.
- Keep only final result CSVs in `outputs/`. Keep intermediate run files in the local run-state directory.
- When a report must be one page, use a one-page LaTeX layout. Verify the compiled page count only when PDF generation is requested.
