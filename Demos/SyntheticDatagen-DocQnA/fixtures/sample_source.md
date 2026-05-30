# COBOL — a brief reference

COBOL (Common Business-Oriented Language) is a compiled English-like computer programming language designed for business use. It was first standardized in 1968 and remains widely used in mainframe applications for finance, payroll, and government systems.

## Program structure

A COBOL program has four divisions, in this order:

1. **IDENTIFICATION DIVISION** — program metadata (name, author, date written)
2. **ENVIRONMENT DIVISION** — runtime environment (input/output assignments, file control)
3. **DATA DIVISION** — data declarations (WORKING-STORAGE, FILE, LINKAGE sections)
4. **PROCEDURE DIVISION** — executable statements (paragraphs, sentences)

## Data types

COBOL declares fields with PICTURE clauses:

- `PIC X(20)` — 20-character alphanumeric string
- `PIC 9(5)` — 5-digit numeric (unsigned)
- `PIC S9(7)V99` — signed 7-digit integer with 2-digit decimal
- `PIC A(10)` — 10-character alphabetic

Group items aggregate related fields under a single name with level numbers (01 for record, 05/10 for subfields).

## File I/O

File access is declared in the ENVIRONMENT DIVISION's INPUT-OUTPUT SECTION and used via `OPEN`, `READ`, `WRITE`, `REWRITE`, `CLOSE` statements. Organizations: SEQUENTIAL (line-by-line), INDEXED (keyed), or RELATIVE (record-number).

## Common idioms

- `PERFORM` for loops and subroutine calls
- `EVALUATE` for multi-way branching (like switch/case)
- `COMPUTE` for arithmetic expressions
- `INSPECT` for character-level string manipulation
- `STRING` / `UNSTRING` for concatenation and parsing

## Modern usage

COBOL persists in 2020s production systems because business logic written decades ago still works and rewrites are risky. Modern COBOL compilers (GnuCOBOL, IBM Enterprise COBOL) support web interop, JSON parsing, and integration with Java and .NET.

