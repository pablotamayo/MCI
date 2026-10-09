
<h2 > 
The Multi-scale Copula Index (MCI): a new information-theoretic measure of association and its applications in functional cancer genomics

</h2>
<br>

**Authors:** Timothy Pham and Pablo Tamayo

**Date:** October 8, 2026

Cancer genomic analyses increasingly involve large, complex datasets representing diverse 'omics' variables, such as gene expression, mutations, copy number alterations, genetic perturbations (e.g. CRISPR or iRNA), drug sensitivity, and clinical phenotypes, making it crucial to accurately measure associations among these variables. Traditional correlation-based methods, like the Pearson correlation coefficient, often assume linearity, Gaussian distributions, and independence among variables, leading to oversimplified assumptions, less sensitive readouts and less accurate significance estimation. Information-theoretic measures overcome these limitations by being one step removed from the data and assessing dependencies at the level of probability distributions, thus providing a more data-agnostic, robust measure for identifying associations in common analysis tasks such as feature selection, pattern recognition, and predictive modeling. Building on these foundations, we introduce the Multi-scale Copula Index (MCI), a novel information-theoretic measure of association. The MCI offers significant advantages over existing methods by eliminating the need for density estimation and parameter tuning, thereby ensuring consistent estimation with reduced computational complexity. Furthermore, it enhances statistical power at lower sample sizes, exhibits strong robustness to outliers, and successfully captures both directionality and important symmetries under input-data transformations, such as reflection or complementation. Due to these strengths, the MCI is particularly useful for functional cancer genomic applications involving, e.g., transcriptional and perturbational data, with different statistical properties across a wide range of dataset sizes and complexity. Contextual associations often arise in functional genomics datasets: two genomic variables show little or no correlation in most samples, yet display a strong linkage within the subset that reflects the “active” biological context. These situations are ideal for quantifying associations with the MCI. We illustrate the performance of the MCI in a variety of benchmarks and cancer genomic examples and applications. We also introduce the Cumulative Entropy Diagram (CED), a graphical display that allows the visualization of the MCI between a reference vector, such as a phenotype, pathway activation or drug response, and a complete set of genomic features.It computes the Multi-scale Copula Index (MCI) between two vectors.

## Installation
Requires Python ≥3.11 

## Usage
```
MCI(x, y)
```

See `MCI Examples.ipynb` for examples.
