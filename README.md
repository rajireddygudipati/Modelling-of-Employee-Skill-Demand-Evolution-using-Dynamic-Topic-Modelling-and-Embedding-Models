# Modelling of Employee Skill Demand Evolution in the US 
### Using Dynamic Topic Modelling and Embedding Models

## About
This is a practicum project addressing the problem of the understanding of skill demands in the dynamic business environment of big economies. Understanding of the skill demand requires evaluation of the evolution of skill demand overtime and contextualizing this demand in terms of the present workforce and the national averages. Through the evaluation, a complete picture of the skill demand for big economies can be generated for use by policy makers and companies. 

## Data

For the project, the data was sourced from: 
- US Federal Government Job Postings API (https://developer.usajobs.gov/api-reference/)
- US Federal Payroll Data API (https://www.census.gov/data/datasets/2024/econ/apes/annual-apes.html) 
- National Labor Statistics API (https://data.bls.gov/oes/#/industry/000000)
- New York City API Data (https://data.cityofnewyork.us/resource/k397-673e.json)

## Package Requirements

### Pacakge Installations

```

!pip uninstall -y sentence-transformers transformers accelerate peft
!pip install "transformers==4.38.2" "sentence-transformers==2.6.1" "accelerate==0.27.2" "peft==0.8.2"

!pip install --no-cache-dir --upgrade "scikit-learn==1.4.2"
!pip install umap-learn

```

### Package Importations 

```

import requests
import pandas as pd
from datetime import datetime
import json
import re
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import umap
from sklearn.cluster import KMeans

```

## Models and Analyses

- Natural Language Processing (NLP) extracts key terms from textual data (O’Shaughnessy, 2026). In this case extraction of skill signals from job description.
- Topic Modelling employs the Latent Dirichlet Allocation(LDA) to group terms into topics based on thematic analysis (Hairani et al., 2024). In this case grouping of skill demand terms that provide signals for the skill demand.
- Dynamic Topic Modelling conducts LDA overtime to capture evolution of skill demand. 
- Embedded Models evaluates the similarity and semantics in textual data (Ajallouda et al., 2025). In this case similarity and semantics of skills. 
- Trend Analysis provides overview of time series data (Siswanto et al., 2025). In this case, overview of the time series for current and historic employment indicators; unemployment rate and average hourly pay. 
- Geospatial analysis plots attributes based on spatial characteristics of record into maps (Muhammad et al., 2025). In this case mapping of topics from the LDA across the US.

## Analysis Approach 

<img width="929" height="978" alt="image" src="https://github.com/user-attachments/assets/3262e377-cb47-43df-9b09-6c82a100773c" />

## Results Overview

### Topic Modelling 

<table>
  <tr>
    <td align="center">
      <img width="300" alt="Primary Case" src="https://github.com/user-attachments/assets/ab8d4733-7faa-4712-aaa4-3de7e7ea853f" />
      <br />
      <b>Primary Case (National/Federal Level)</b>
    </td>
    <td align="center">
      <img width="300" alt="Comparison Case 1" src="https://github.com/user-attachments/assets/6b8df90b-53ea-40e9-80d7-15fd73086561" />
      <br />
      <b>Comparison Case 1 (Local and Federal Government Level)</b>
    </td>
    <td align="center">
      <img width="300" alt="Comparison Case 2" src="https://github.com/user-attachments/assets/61355fd3-71dd-47a1-8268-a7d72d966397" />
      <br />
      <b>Comparison Case 2 (City Level for New York City)</b>
    </td>
  </tr>
</table>

### Geospatial Analysis for Primary Case (National/Federal Level)

<table>
  <tr>
    <td align="center">
      <img width="350" alt="Skill Demand Set 0" src="https://github.com/user-attachments/assets/4d1018b4-0a07-46e2-b29c-a457690216fa" />
      <br />
      <b>Skill Demand Set 0</b>
    </td>
    <td align="center">
      <img width="350" alt="Skill Demand Set 1" src="https://github.com/user-attachments/assets/5a83af8c-bd39-4df6-8ee5-5c7f67661b58" />
      <br />
      <b>Skill Demand Set 1</b>
    </td>
    <td align="center">
      <img width="350" alt="Skill Demand Set 2" src="https://github.com/user-attachments/assets/90d6be45-d862-47bd-baa1-8859bbd741b5" />
      <br />
      <b>Skill Demand Set 2</b>
    </td>
  </tr>
</table>

### Embedded Modelling for Primary Case (National/Federal Level)

<img width="917" height="852" alt="image" src="https://github.com/user-attachments/assets/0dcc7c88-e038-4a4a-a7b4-5d9b37ff7f73" />

## References

Ajallouda, L., Hassani Saissi, M., & Zellou, A. (2025). Embedding models: A comprehensive review with task‑oriented assessment. International Journal of Advanced Computer Science and Applications, 16(10), 539–550. http://dx.doi.org/10.14569/IJACSA.2025.0161056

Hairani, H., Janhasmadja, M., Tholib, A., Guterres, J. X., & Ariyanto, Y. (2024). Thesis topic modeling study: Latent Dirichlet Allocation (LDA) and machine learning approach. International Journal of Engineering and Computer Science Applications, 3(2), 51–60. https://doi.org/10.30812/ijecsa.v3i2.4375

Muhammad Hashim, Atta-ur Rahman, Muhammad Qasim, Muhammad Umar Farooq, Muhammad Dawood, Basit Nadeem, & Shazia Muneer. (2025). Application of Geospatial Approaches for Evaluation of Urban Growth Pattern and Trend Prediction of Multan City, Pakistan. International Journal of Innovations in Science & Technology, 7(9), 108–120. Retrieved from https://journal.50sea.com/index.php/IJIST/article/view/1512

O’Shaughnessy, D. (2026). An Overview of Recent Advances in Natural Language Processing for Information Systems. Applied Sciences, 16(2), 1122. https://doi.org/10.3390/app16021122

Siswanto, J., Goeltom, V. A. H., Pandawan, I. N. H., Lisangan, E. A., & Fitriani, A. (2025). Market trend analysis and data-based decision making in increasing business competitiveness. Sundara Advanced Research on Artificial Intelligence, 1(1), 1–8. https://journal.sundarapublishing.com/index.php/sundara/article/view/1





