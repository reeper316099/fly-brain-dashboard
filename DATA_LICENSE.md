# Data attribution

This project's code is yours to license however you like in your repo
(e.g. MIT). The **connectome data it downloads is not** — it's a
separate dataset with its own license:

- **Dataset:** MaleCNS v1.0
- **Publishers:** HHMI Janelia FlyEM, University of Cambridge Dept. of
  Zoology, MRC Laboratory of Molecular Biology, Google Research
- **License:** [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- **Source:** https://male-cns.janelia.org/download/

CC-BY means you can use, share, and adapt the data, including
commercially, as long as you give appropriate credit. If you publish
anything built on this (write-ups, videos, forks), credit the dataset
and link back to the source above.

The data files themselves are excluded from git via `.gitignore` — this
repo only ships the code that downloads and processes them, not the
data itself (the `connectome-weights` file alone is 1.1 GB, which is
also just too big for a normal GitHub repo).
