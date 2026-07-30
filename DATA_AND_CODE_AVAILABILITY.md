# Data and Code Availability

This file holds the text to be placed in the article once the repository is
published and a DOI has been issued. Two placeholders must be replaced:

  <GITHUB_URL>   the repository address, for example
                 https://github.com/<account>/zagros-makran-benchmark
  <ZENODO_DOI>   the DOI minted by Zenodo for the archived release,
                 for example 10.5281/zenodo.0000000

Both paragraphs below state that the catalog export is available in the
repository. That statement is accurate only if data/Final_.csv is committed. If
the catalog is not committed, the corresponding clause must be replaced by:

  The catalog is available from the ISC Bulletin using the search parameters
  given in Section 2.

---

## Paragraph for the article

Data and Code Availability

The earthquake catalog analysed in this study was obtained from the
International Seismological Centre (ISC) Bulletin
(https://doi.org/10.31905/D808B830) for the region 25-29 N, 54-58 E over the
period 1998-2023. The catalog export used as input, the complete analysis
pipeline, and the configuration files required to reproduce every reported
quantity, table and figure are openly available at <GITHUB_URL> and archived at
https://doi.org/<ZENODO_DOI>. The repository includes the six sequential
scripts that construct the catalog, estimate completeness and the b-value,
train the four architectures under the common parameter budget, compute the
baselines and the resampling inference, produce the attribution profiles, and
collect all reported numbers into a single file. Random seeds are fixed and
recorded, and the software versions used to produce the reported results are
pinned in requirements.txt. The supplementary tables referenced in the text are
provided as comma-separated files in the same repository.

---

## Shorter variant, if the journal restricts the length of this section

Data and Code Availability

The catalog was obtained from the ISC Bulletin
(https://doi.org/10.31905/D808B830). The catalog export and the full analysis
pipeline, with fixed seeds and pinned software versions, are available at
<GITHUB_URL> and archived at https://doi.org/<ZENODO_DOI>.

---

## Note for the reference list

International Seismological Centre (2023). ISC Bulletin: event catalogue.
International Seismological Centre. https://doi.org/10.31905/D808B830
