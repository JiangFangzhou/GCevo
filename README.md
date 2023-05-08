# GCevo
Star-cluster evolution model for Liang, Jiang et al. (2023, arXiv:2304.14431 )

The properties of globular clusters (GCs) contain valuable information of their host galaxies and dark-matter halos. In some dwarf galaxies, the GC population exhibits strong radial mass segregation, indicative of dynamical-friction-driven orbital decay, which opens the possibility of using imaging data alone to constrain the dark-matter content of the galaxy. To explore this possibility, we develop a semi-analytical model of GC evolution, which starts from the initial mass function, the initial structure-mass relation, and the initial spatial distribution of the GC progenitors, and follows the effects of dynamical friction, tidal evolution, and two-body relaxation. Our model is generally applicable to GC-rich dwarf galaxies, and is publicly available here.  In the proof-of-concept study Liang, Jiang et al. (2023), we forward-model the GCs in a remarkable ultra-diffuse galaxy, NGC5846-UDG1, to match the observed GC mass, size, and spatial distributions, and to constrain the profile of the host halo and the origin of the GCs.We find that, with the assumptions of zero mass segregation when the star clusters were born, NGC5846-UDG1 is dark-matter poor compared to what is expected from stellar-to-halo-mass relations, and its halo concentration is low, irrespective of having a cuspy or a cored halo profile. Its GC population has an initial spatial distribution more extended than the smooth stellar distribution. We discuss the results in the context of scaling laws of galaxy-halo connections, and warn against naively using the GC-abundance-halo-mass relation to infer the halo mass of UDGs.  

The basic layout of this program is based upon that of the semi-analytical framework of satellite-galaxy (subhalo) evolution, SatGen, presented in Jiang et al. (2021). We refer interested users to the well-documented SatGen program for more details -- https://github.com/JiangFangzhou/SatGen

We have included new profile classes for GC and the diffuse host galaxy in the density-profile module (profiles.py).
We have implemented the Eddington's inversion method for velocity initialization, as in the initialization module (init.py).
We have developed a self-consistent star-cluster evolution model, and implemented in the satellite-evolution module (evolve.py). 
We inlcude the Petts et al. (2015,2016) treatment of dynamical friction for orbital evolution (with changes made in profiles.fDF)
We provide example programs and figures in the scripts folder.

For more programs, such as forward-modeling using the GC evolution model with MCMC inference of the dark-matter halo, please feel free to contact us at 
fangzhou.jiang@pku.edu.cn

We encourage users of this model and the broader SatGen framework for galaxy evolution to cite this project (Liang et al. 2023, arXiv:2304.14431) and Jiang et al. (2021, https://ui.adsabs.harvard.edu/abs/2021MNRAS.502..621J/abstract). Thanks. 
