(sec_citation)=

# Citing tskit

If you use `tskit` in your work, we recommend citing the "tskit paper":
> Ben Jeffery, Yan Wong, Kevin Thornton, Georgia Tsambos, Gertjan Bisschop, Yun
> Deng, E. Castedo Ellerman, Thomas B. Forest, Halley Fritze, Daniel Goldstein,
> Gregor Gorjanc, Graham Gower, Simon Gravel, Jeremy Guez, Benjamin C. Haller,
> Andrew D. Kern, Lloyd Kirk, Ivan Krukov, Hanbin Lee, Brieuc Lehmann,
> Hossameldin Loay, Matthew M. Osmond, Duncan S. Palmer, Nathaniel S. Pope, Aaron
> P. Ragsdale, Duncan Robertson, Murillo F. Rodrigues, Hugo van Kemenade, Clemens
> L. Weiß, Anthony Wilder Wohns, Shing H. Zhan, Brian C. Zhang, Marianne Aspbury,
> Nikolas A. Baya, Saurabh Belsare, Arjun Biddanda, Francisco Campuzano Jiménez,
> Ariella Gladstein, Bing Guo, Savita Karthikeyan, Warren W. Kretzschmar, Inés
> Rebollo, Kumar Saunack, Ruhollah Shemirani, Alexis Simon, Chris Smith, Jeet
> Sukumaran, Jonathan Terhorst, Per Unneberg, Ao Zhang, Peter Ralph, Jerome
> Kelleher, *Population-scale Ancestral Recombination Graphs with tskit 1.0*,
> arXiv:2602.09649v2,
> doi: [10.48550/arXiv.2602.09649](https://doi.org/10.48550/arXiv.2602.09649)

For citations that discuss ARGs and how these are represented in a tree sequence,
we recommend the [2024 ARG Genetics paper](<https://doi.org/10.1093/genetics/iyae100>)
and the [2016 msprime PLOS Computational Biology paper](<http://dx.doi.org/10.1371/journal.pcbi.1004842>):
> Yan Wong, Anastasia Ignatieva, Jere Koskela, Gregor Gorjanc, Anthony W 
> Wohns, Jerome Kelleher, *A general and efficient representation of ancestral 
> recombination graphs*, Genetics, Volume 228, Issue 1, September 2024, iyae100, 
> https://doi.org/10.1093/genetics/iyae100

> Jerome Kelleher, Alison M Etheridge and Gilean McVean (2016),
> *Efficient Coalescent Simulation and Genealogical Analysis for Large Sample Sizes*,
> PLOS Comput Biol 12(5): e1004842. doi: 10.1371/journal.pcbi.1004842

If you use summary statistics, please cite the
[2020 Genetics paper](https://doi.org/10.1534/genetics.120.303253):

> Peter Ralph, Kevin Thornton, Jerome Kelleher, *Efficiently Summarizing 
> Relationships in Large Samples: A General Duality Between Statistics of 
> Genealogies and Genomes*, Genetics, Volume 215, Issue 3, 1 July 2020, 
> Pages 779–797, https://doi.org/10.1534/genetics.120.303253


Bibtex records:

```bibtex
@misc{jeffery2026tskit,
      title={Population-scale Ancestral Recombination Graphs with tskit 1.0}, 
      author={Ben Jeffery and Yan Wong and Kevin Thornton and Georgia Tsambos
              and Gertjan Bisschop and Yun Deng and E. Castedo Ellerman and Thomas B. Forest
              and Halley Fritze and Daniel Goldstein and Gregor Gorjanc and Graham Gower and
              Simon Gravel and Jeremy Guez and Benjamin C. Haller and Andrew D. Kern and
              Lloyd Kirk and Ivan Krukov and Hanbin Lee and Brieuc Lehmann and Hossameldin
              Loay and Matthew M. Osmond and Duncan S. Palmer and Nathaniel S. Pope and Aaron
              P. Ragsdale and Duncan Robertson and Murillo F. Rodrigues and Hugo van Kemenade
              and Clemens L. Weiß and Anthony Wilder Wohns and Shing H. Zhan and Brian C.
              Zhang and Marianne Aspbury and Nikolas A. Baya and Saurabh Belsare and Arjun
              Biddanda and Francisco Campuzano Jiménez and Ariella Gladstein and Bing Guo and
              Savita Karthikeyan and Warren W. Kretzschmar and Inés Rebollo and Kumar Saunack
              and Ruhollah Shemirani and Alexis Simon and Chris Smith and Jeet Sukumaran and
              Jonathan Terhorst and Per Unneberg and Ao Zhang and Peter Ralph and Jerome
              Kelleher},
      year={2026},
      eprint={2602.09649},
      archivePrefix={arXiv},
      primaryClass={q-bio.PE},
      url={https://arxiv.org/abs/2602.09649}, 
}

@article{Wong2024ARGs,
  author    = {Wong, Yan and Ignatieva, Anastasia and Koskela, Jere and Gorjanc, Gregor and 
               Wohns, Anthony W and Kelleher, Jerome},
  title     = {A general and efficient representation of ancestral recombination graphs},
  journal   = {Genetics},
  volume    = {228},
  number    = {1},
  pages     = {iyae100},
  year      = {2024},
  doi       = {10.1093/genetics/iyae100}
}

@article{Kelleher2016msprime,
  author    = {Kelleher, Jerome and Etheridge, Alison M and McVean, Gilean},
  title     = {Efficient coalescent simulation and genealogical analysis for large sample sizes},
  journal   = {PLoS Computational Biology},
  volume    = {12},
  number    = {5},
  pages     = {e1004842},
  year      = {2016},
  publisher = {Public Library of Science}
}

@article{Ralph2020Stats,
  author    = {Ralph, Peter and Thornton, Kevin and Kelleher, Jerome},
  title     = {Efficiently Summarizing Relationships in Large Samples: A General Duality Between Statistics of Genealogies and Genomes},
  journal   = {Genetics},
  volume    = {215},
  number    = {3},
  pages     = {779--797},
  year      = {2020},
  doi       = {10.1534/genetics.120.303253}
}
```

# Funding and acknowledgements


The tskit software has benefited from input and contributions from
too many people to list here (but see the author list above).
We also gratefully acknowledge funding from the Robertson Foundation, the NIH
(research grants HG011395 and HG012473), and the NSF 
(research grant [OAC-2104115](https://www.nsf.gov/awardsearch/show-award/?AWD_ID=2104115)),
that has supported core tskit development.

