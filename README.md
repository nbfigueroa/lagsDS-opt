## Locally Active Globally Stable (LAGS) - DS
Toolbox including optimization techniques for estimation of LAGS-DS (Locally Active Globally Stable) [1] for 2D and 3D tasks. It also includes a simulation of a 2D robot in which you can test your learned LAGS models.


### Installation Instructions
This package depends on two external packages:
- [phys-gmm](https://github.com/nbfigueroa/phys-gmm): Toolbox with GMM fitting approaches [2].
- [ds-opt](https://github.com/nbfigueroa/ds-opt): Toolbox for lpv-DS optimization [2]

These are included as submodules and can be downloaded as follows:

After cloning the repo one must initialize/download the submodules with the following commands:
```
cd ~./lagsDS-opt
git submodule init
git submodule update
```
In case you want to update the submodules to their latest version, you can do so with the following command:
```
git submodule update --remote
```

Note that [phys-gmm](https://github.com/nbfigueroa/phys-gmm) depends on [LightSpeed Matlab Toolbox](https://github.com/tminka/lightspeed) which should be installed seperately.

### Running the demo scripts

---

**References**     
> [1] Figueroa, N and Billard, A. "Locally Active Globally Stable Dynamical Systems: Theory, Learning and Experiments" [In Preparation]   
> [2] Figueroa, N. and Billard, A. (2018) "A Physically-Consistent Bayesian Non-Parametric Mixture Model for Dynamical System Learning". In Proceedings of the 2nd Conference on Robot Learning (CoRL).

**Contact**: [Nadia Figueroa](http://lasa.epfl.ch/people/member.php?SCIPER=238387) (nadia.figueroafernandez AT epfl dot ch)

**Acknowledgments**
This work was supported by the EU project [Cogimon](https://cogimon.eu/cognitive-interaction-motion-cogimon) H2020-ICT-23-2014.


