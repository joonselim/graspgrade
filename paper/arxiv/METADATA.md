# arXiv submission sheet

Upload file: `graspgrade_arxiv.tar.gz` (`main.tex` and `figures/`, 7 images).
It builds with pdflatex to 11 pages, no overfull box, no warning. The
bibliography is inside `main.tex`, so no `.bbl` file is needed.

## Start page

| Field | Value |
| --- | --- |
| Authorship | I am submitting as an author of this article |
| License | CC BY-NC-SA 4.0 (see below) |
| Archive | Computer Science (cs) |
| Primary subject class | cs.RO, Robotics |

## Metadata page

| Field | Value |
| --- | --- |
| Title | GraspGrade: Scoring Human Egocentric Video Clips for Robot Hand Imitation, Checked Against Touch Gloves and Motion Capture |
| Authors | Inseok Lee, Joonse Lim |
| Abstract | `abstract.txt` (1,796 characters, the limit is 1,920) |
| Comments | 11 pages, 7 figures, 8 tables. Project page: https://graspgrade.vercel.app . Code and benchmark: https://github.com/joonselim/graspgrade |
| Cross-lists | cs.CV, cs.LG |
| Report number, journal ref, DOI, MSC, ACM | leave empty |

The abstract in `abstract.txt` is the paper abstract without two summary
sentences, because the paper abstract has 2,104 characters. All numbers are
the same.

## License

Figure 3 (`exp16_examples.jpg`) draws HOT3D hand annotations, which are
CC BY-NC-SA 4.0 (`MEDIA_LICENSE.md`). The paper text already says these
figures are shared for non-commercial use. CC BY-NC-SA 4.0 matches that.
CC BY, CC BY-SA and CC Zero would not. The arXiv perpetual non-exclusive
license also does not clash, because it gives no reuse rights to readers.
The choice cannot be changed later.
