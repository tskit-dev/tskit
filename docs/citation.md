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
> Nature Methods (2026),
> doi: [10.1038/s41592-026-03325-1](https://doi.org/10.1038/s41592-026-03325-1)

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
@article{jeffery2026tskit,
  title={Population-scale Ancestral Recombination Graphs with tskit 1.0},
  author = {Jeffery, Ben and Wong, Yan and Thornton, Kevin and Tsambos, Georgia and
    Bisschop, Gertjan and Deng, Yun and Ellerman, E. Castedo and Forest, Thomas B. and Fritze, Halley and
    Goldstein, Daniel and Gorjanc, Gregor and Gower, Graham and Gravel, Simon and Guez, Jeremy and
    Haller, Benjamin C. and Kern, Andrew D. and Kirk, Lloyd and Krukov, Ivan and Lee, Hanbin and
    Lehmann, Brieuc and Loay, Hossameldin and Osmond, Matthew M. and Palmer, Duncan S. and
    Pope, Nathaniel S. and Ragsdale, Aaron P. and Robertson, Duncan and Rodrigues, Murillo F. and
    van Kemenade, Hugo and Wei{\ss}, Clemens L. and Wohns, Anthony Wilder and Zhan, Shing H. and
    Zhang, Brian C. and Aspbury, Marianne and Baya, Nikolas A. and Belsare, Saurabh and
    Biddanda, Arjun and Campuzano Jim{\'e}nez, Francisco and Gladstein, Ariella and Guo, Bing
    and Karthikeyan, Savita and Kretzschmar, Warren W. and Rebollo, In{\'e}s and Saunack, Kumar
    and Shemirani, Ruhollah and Simon, Alexis and Smith, Chris and Sukumaran, Jeet and
    Terhorst, Jonathan and Unneberg, Per and Zhang, Ao and Ralph, Peter and Kelleher, Jerome},
	id = {Jeffery2026},
	isbn = {1548-7105},
	journal = {Nature Methods},
	title = {Population-scale ancestral recombination graphs with tskit 1.0},
	url = {https://doi.org/10.1038/s41592-026-03325-1},
	year = {2026},
  doi = {10.1038/s41592-026-03325-1}
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

