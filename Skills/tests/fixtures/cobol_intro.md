# COBOL: A Brief Reference

This document is a small, self-contained reference for the COBOL programming
language. It is included as a synthetic test fixture for the Foundry Data
Generation API's `SimpleQnA` recipe — the goal is for a teacher model to be
able to generate a small batch (~15) of meaningful question / answer pairs
from this content.

The original content here is released to the public domain (CC0).

## What is COBOL?

COBOL (Common Business-Oriented Language) is a compiled English-like
programming language designed for business data processing. It was first
standardized in 1968 and remains widely used in mainframe applications for
finance, payroll, insurance, and government systems. Estimates of the
amount of COBOL code in active production worldwide range from 200 to
800 billion lines, depending on the source.

The language was conceived in 1959 by a committee that included Grace
Hopper, who is sometimes called "the mother of COBOL" — though her actual
role was as a senior advisor whose prior compiler work (FLOW-MATIC)
heavily influenced the design.

## Program Structure

Every COBOL program contains four required divisions, in this order:

1. **IDENTIFICATION DIVISION** — program metadata. Required entry:
   `PROGRAM-ID`. Optional entries include `AUTHOR`, `INSTALLATION`,
   `DATE-WRITTEN`, `DATE-COMPILED`.

2. **ENVIRONMENT DIVISION** — runtime environment description. Two
   sections: `CONFIGURATION SECTION` (source-computer, object-computer)
   and `INPUT-OUTPUT SECTION` (file-control, i-o-control).

3. **DATA DIVISION** — data declarations. Sections:
   - `FILE SECTION` — record descriptions for files declared in
     `INPUT-OUTPUT SECTION`.
   - `WORKING-STORAGE SECTION` — variables initialized once at program
     start and retained across PERFORM invocations.
   - `LOCAL-STORAGE SECTION` — variables re-initialized on each
     entry (since COBOL-85).
   - `LINKAGE SECTION` — parameters passed in from a calling program.

4. **PROCEDURE DIVISION** — executable statements organized into
   paragraphs and sections. This is where the program logic lives.

## Data Types and PICTURE Clauses

COBOL declares fields with `PICTURE` (or `PIC`) clauses. The clause
describes both the type and the storage layout. Common forms:

- `PIC X(20)` — 20-character alphanumeric string
- `PIC 9(5)` — 5-digit unsigned numeric
- `PIC S9(7)V99` — signed 7-digit integer with 2-digit decimal (the `V`
  is an implied decimal point, not stored)
- `PIC A(10)` — 10-character alphabetic (letters only; rarely used in
  practice since modern code uses `X`)
- `PIC $$$,$$9.99` — edited numeric, displays with leading-dollar-sign
  suppression and comma insertion

Group items aggregate related fields under a single name with level
numbers (01 for the record, 05/10/15 for subfields). Special level
numbers: 66 (RENAMES), 77 (independent elementary, no subordinates),
88 (condition-name).

## File I/O

File access is configured in the `ENVIRONMENT DIVISION`'s
`INPUT-OUTPUT SECTION` (`FILE-CONTROL` paragraph), declared in the
`DATA DIVISION` (`FILE SECTION`), and used in the `PROCEDURE DIVISION`
via the verbs `OPEN`, `READ`, `WRITE`, `REWRITE`, `DELETE`, and
`CLOSE`. Three file organizations are supported:

- **SEQUENTIAL** — records accessed line by line in the order written.
- **INDEXED** — records keyed by one or more indexed fields, accessible
  by key or sequentially.
- **RELATIVE** — records accessed by their position number (1, 2, 3, ...).

The `OPEN` verb takes one of four modes: `INPUT`, `OUTPUT`, `I-O`, or
`EXTEND`. Reading past end-of-file raises the `AT END` condition.

## Common Procedural Idioms

- **PERFORM** — call a paragraph or section. `PERFORM ... THRU` calls a
  range. `PERFORM ... VARYING ... UNTIL` is COBOL's primary loop
  construct.
- **EVALUATE** — multi-way branching (analogous to switch/case in C).
  Supports condition expressions and the `ALSO` keyword for
  multi-dimensional conditions.
- **COMPUTE** — arithmetic expression evaluation, e.g.
  `COMPUTE TAX = SUBTOTAL * TAX-RATE ROUNDED`.
- **INSPECT** — character-level inspection and substitution. Common
  patterns: `INSPECT field TALLYING count FOR ALL 'A'` (count
  occurrences) and `INSPECT field REPLACING ALL ' ' BY '_'`.
- **STRING** and **UNSTRING** — concatenation and parsing. `STRING`
  joins multiple sources with delimiters; `UNSTRING` splits one source
  into multiple receivers based on delimiters.
- **MOVE** — copy data between fields, with implicit conversion rules
  that depend on the PICTURE clauses.

## The Report Writer

COBOL includes a built-in REPORT WRITER feature (`REPORT SECTION` in
the `DATA DIVISION`, `GENERATE` and `TERMINATE` verbs in the
`PROCEDURE DIVISION`). It defines page-oriented reports with
declarative grouping, totaling, and pagination. Two automatically
maintained data items are available:

- **PAGE-COUNTER** — incremented by one each time the report writer
  control system executes a page advance.
- **LINE-COUNTER** — tracks the current vertical position on the page.

The `PAGE` clause within a `REPORT` definition controls layout. If the
`HEADING` phrase is omitted from the `PAGE` clause, an implicit value
of `1` is assumed (i.e., `integer-2` defaults to 1), so report-heading
and page-heading groups begin at the top of the page.

## Subprograms and Linkage

COBOL programs can call other programs via the `CALL` verb. The called
program receives arguments via its `LINKAGE SECTION`, with parameters
declared in the `PROCEDURE DIVISION USING` clause. Calls can be
**static** (resolved at link time) or **dynamic** (resolved at runtime
via a name in a data item). The `EXIT PROGRAM` (or implicit fall-through
to end of procedure division) returns control to the caller.

## Communicating Unusual I/O Conditions

When a file operation encounters an unusual condition (end of file,
invalid key, I/O error), COBOL provides three mechanisms for
communicating the condition back to the object program:

1. **Status keys** — values placed in a `FILE STATUS` field after every
   I/O operation. The program inspects the field and branches on the
   value.
2. **Exception declaratives** — `DECLARATIVES` sections in the
   `PROCEDURE DIVISION` registered with `USE AFTER STANDARD EXCEPTION`,
   invoked automatically on the matching condition.
3. **Optional phrases on the I/O statement** — phrases like
   `AT END`, `INVALID KEY`, and `ON OVERFLOW` that supply inline
   handlers triggered when the condition occurs.

Most production code uses status keys for predictable conditions and
declaratives for catch-all error handling.

## Modern Usage

COBOL persists in production not because of technical superiority but
because the business logic it expresses has been validated against
decades of real-world edge cases. Replacing it with a rewrite carries
substantial risk of regressions in tax handling, regulatory
compliance, settlement calculations, and similar high-stakes domains.

Modern COBOL implementations (IBM Enterprise COBOL, Micro Focus
Visual COBOL, GnuCOBOL) support interoperation with Java and .NET,
JSON parsing, web services, and modern source control. The 2014 and
2023 standards added features such as nested programs, recursion,
user-defined functions, and Boolean and floating-point data types.

For new development, COBOL is rarely the right tool — but for
maintaining and extending the existing trillion-line installed base,
it remains essential. Training programs at IBM, large insurance
firms, and several U.S. state governments continue to produce
COBOL-fluent engineers each year.
