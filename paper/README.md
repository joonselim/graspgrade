# Paper

`main.tex` is the GraspGrade write-up in paper form. It is self-contained:
two columns, standard LaTeX packages only (`geometry`, `mathptmx`, `graphicx`,
`booktabs`, `amsmath`, `caption`, `hyperref`), an inline bibliography, and the
figures in `figures/`.

The tone follows the progress page at <https://graspgrade.vercel.app>: plain
statements, a number and an experiment id behind each claim, and the rows we
lose reported as they are. Section 4 mirrors the formulas and fixed thresholds
of that page's calculation block.

All numbers come from the experiment pages in `../docs/experiments/`:
task 1 from EXP-37, task 2 from EXP-40 and EXP-41, the copyable label from
EXP-16 and EXP-20, the object checks from EXP-18 and EXP-19, the soft object
check from EXP-42, the benchmark from EXP-28, and the cost from the experiment
index.

## Build

LaTeX is installed in WSL (`texlive-latex-recommended`,
`texlive-fonts-recommended`, `texlive-latex-extra`). Build with:

```bash
cd paper && pdflatex main.tex && pdflatex main.tex
```

Run it twice so the table and figure references resolve. The last build
(2026-11-03) gives 11 pages with no overfull box and no unresolved reference.

Overleaf works as well: upload `main.tex` and the `figures/` folder, keeping
the folder name.

## Before submitting

- Replace the author line in `main.tex` (marked with a `TODO` comment).
- The figures are copies of `../docs/experiments/img/`. If a figure there
  changes, copy it again.
