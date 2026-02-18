# Consensus Sampling Method for Exploring High-Dimensional Energy Surfaces

## Guidelines for SISC Reproducibility Badges

Authors may request the **“SISC Reproducibility Badge: code and data available”** at manuscript submission.

### 1) Badge Criteria: “Code and Data Available”

To obtain this badge, authors should ensure all of the following:

- **Public availability of materials**: All computer code and data implementing the computational methods proposed in the paper are publicly available.
- **Reproducible numerical results**: Parameter settings are included in code and/or separate data files so readers can reproduce all numerical results (tables, figures, etc.).
- **Detailed README**: The repository includes a README that:
  - Describes code and data files in detail.
  - Explains clearly how to run the code to reproduce paper results.

### 2) Acceptable Mechanisms for Sharing Code/Data

Authors can provide reproducibility materials through one of the following mechanisms:

1. A **public permanent Git repository** (e.g., GitHub, Bitbucket, or similar) **plus** a zipped snapshot of that repository in SISC supplementary materials.
2. A **general-purpose open repository with DOI** (e.g., Zenodo, or similar).
3. **Supplementary materials** published with the SISC paper.

> **Not acceptable**: personal academic websites or similar non-persistent locations.

### 3) Additional Guidelines for Git Repositories (e.g., GitHub/Bitbucket)

When requesting the badge during submission:

- Provide the repository URL.
- Also deposit a **zipped snapshot** in supplementary materials to guarantee long-term access to a persistent, immutable, and findable copy.
- Prefer a **dedicated reproducibility repository** for this specific paper to improve discoverability.
- If using a larger existing repository:
  - Put this paper’s reproducibility materials in a clearly identified subfolder.
  - Provide a URL that points directly to that subfolder.
- In the main README (or subfolder README), clearly state:
  - Paper title.
  - Author names.
  - That the repository/subfolder is intended to provide reproducibility materials for the paper.
- Ensure associated files remain easy to access after future commits (e.g., bug fixes, ongoing development).

### 4) Evaluation Process for Badge Requests

At paper acceptance, the handling editor performs a high-level validation:

- Is code/data publicly available?
- Do materials sufficiently cover computational methods and computational tests described in the paper?
- Is there a detailed README describing provided material and reproduction steps?

If some requirements are not fully met but code is still valuable to readers, authors may include a direct code link in the manuscript even if the badge is not awarded.

### 5) Role of Referees and Editors During Review

- Referees may consult code/data (not mandatory) and may suggest requested changes toward badge eligibility.
- In revision decisions, handling editors may suggest or require updates to available code/data to support badge requirements.
