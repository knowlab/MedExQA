<h1 align="center">  :health_worker: MedExQA  </h1>
<h3 align="center"> Medical Question Answering Benchmark with Multiple Explanations </h3>

<p align="center">
 :page_facing_up: <a href="https://arxiv.org/abs/2406.06331" target="_blank">Paper</a> • ![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white) <a href="https://github.com/knowlab/MedExQA" target="_blank">Code</a> • :arrow_double_down: <a href="https://huggingface.co/datasets/bluesky333/MedExQA" target="_blank">Dataset</a>  • :medical_symbol: <a href="https://huggingface.co/bluesky333/medphi2" target="_blank">MedPhi2</a><br>
</p>

## :new: News
- \[**July 2024**\] Our work is accepted at the ACL2024 BioNLP workshop. 
- \[**July 2024**\] We release <a href="https://huggingface.co/datasets/bluesky333/MedExQA" target="_blank">MedExQA</a> dataset. 
- \[**June 2024**\] We release <a href="https://huggingface.co/bluesky333/medphi2" target="_blank">MedPhi2</a> and <a href="https://arxiv.org/abs/2406.06331" target="_blank">Paper</a>. 


## Table of Contents
- [Benchmark Summary](#benchmark-summary)
- [Dataset Structure](#dataset-structure)
- [Citation Information](#citation-information)

##  Benchmark Summary
<a name="benchmark-summary"></a>
<a href="https://huggingface.co/datasets/bluesky333/MedExQA" target="_blank">MedExQA</a> is a novel benchmark in medical question-answering, to evaluate large language models’ (LLMs) understanding of medical knowledge through explanations.
The work is published in ACL2024 BioNLP workshop <a href="https://arxiv.org/abs/2406.06331" target="_blank">paper</a>. We address a major gap in current medical QA benchmarks which is the absence of comprehensive assessments of LLMs’ ability to generate nuanced medical explanations.
This work also proposes a new medical model, <a href="https://huggingface.co/bluesky333/medphi2" target="_blank">MedPhi2</a>, based on Phi-2 (2.7B). The model outperformed medical LLMs based on Llama2-70B in generating explanations.
From our understanding, it is the first medical language model benchmark with explanation pairs.

##  Dataset Structure
<a name="dataset-structure"></a>
Benchmark has 5 five distinct medical specialties that are underrepresented in current datasets.
| Specialty                   | # Examples | Explanation Similarity |
| -----                       | ---------- | ---------------------- |
| Biomedical Engineering      |    148     |  75.8 |
| Clinical Laboratory Science |      377   |  73.7 |
| Clinical Psychology         |      111   |  79.7 |
| Occupational Therapy        |   194      |  79.5 |
| Speech Language Pathology   |   135      |  80.5 |
| Total   |   965      |  78.7 |
Explanation Similarity represents the average cosine similarity of the explanation pairs.

## Citation
<a name="citation-information"></a>
<!-- If there is a paper or blog post introducing the model, the APA and Bibtex information for that should go in this section. -->

**BibTeX:**
```
@article{kim2024medexqa,
  title={MedExQA: Medical Question Answering Benchmark with Multiple Explanations},
  author={Kim, Yunsoo and Wu, Jinge and Abdulle, Yusuf and Wu, Honghan},
  journal={arXiv e-prints},
  pages={arXiv--2406},
  year={2024}
}
```
