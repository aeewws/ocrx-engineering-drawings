# Pipelines

## Smart Mode

`ocrsmart` is the default practical preset for day-to-day drawing work.

- lower DPI and width targets
- faster folder throughput
- good enough for most title blocks, labels, and mixed drawing pages

## Best Mode

`ocrbest` pushes a bit further on detail retention.

- higher DPI and width targets
- more forgiving on hard pages
- better fit for dense notes, fine labels, and mixed CAD-exported scans

## PDF Route

1. inspect the PDF for native vector text
2. extract text directly when possible
3. OCR only the pages or regions that still need visual recognition

## CAD Route

1. load DXF directly or convert DWG to DXF with ODA
2. extract text-like entities from the drawing model
3. render a clean image for OCR fallback
4. merge direct text with OCR text conservatively

## Why The Split Matters

Engineering drawing quality depends less on one global OCR knob and more on routing the right format into the right extraction path. Native parsing and CAD direct extraction remove large classes of OCR errors before model tuning even starts.
