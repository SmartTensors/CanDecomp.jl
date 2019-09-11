## CanDecomp.jl: Candecomp/Parafac Tensor Decomposition ##

### Installation

After starting Julia, execute:

```julia
import Pkg; Pkg.add("CanDecomp")
```

or

```julia
import Pkg; Pkg.develop("CanDecomp")
```

**CanDecomp** is a module required by [NTFk](https://github.com/TensorDecompositions/NTFk.jl). For more information, visit [tensors.lanl.gov](http://tensors.lanl.gov)

### Examples

A simple problem demonstrating **CanDecomp** can be executed as follows.
First, generate a random CP (Candecomp/Parafac) tensor:

```julia
import CanDecomp

A = rand(2, 3)
B = rand(5, 3)
C = rand(10, 3)
T_orig = CanDecomp.totensor(A, B, C)
```

Then generate random initial guesses for the tensor factors:

```julia
Af = rand(size(A)...);
Bf = rand(size(B)...);
Cf = rand(size(C)...);
```

Execute **CanDecomp** to estimate the tensor factors based on the tensor `T_orig` only:

```julia
import StaticArrays
CanDecomp.candecomp!(StaticArrays.SVector(Af, Bf, Cf), T_orig, Val{:nnoptim}; regularization=1e-3, print_level=0, max_cd_iters=1000)
```

Construct the estimated tensor:

```julia
T_est = CanDecomp.totensor(Af, Bf, Cf);
```

Compare the estimated and the original tensors:

```julia
import NTFk
import LinearAlgebra

@info("Norm $(LinearAlgebra.norm(T_est .- T_orig))")

NTFk.plot2matrices(A, Af; progressbar=nothing)
NTFk.plot2matrices(B, Bf; progressbar=nothing)
NTFk.plot2matrices(C, Cf; progressbar=nothing)
NTFk.plotlefttensor(T, T_est; progressbar=nothing)
```

### Tensor Decomposition

**NTFk** performs a novel unsupervised Machine Learning (ML) method based on Tensor Decomposition coupled with sparsity and nonnegativity constraints.

**NTFk** has been applied to extract the temporal and spatial footprints of the features in multi-dimensional datasets in the form of multi-way arrays or tensors.

**NTFk** executes the decomposition (factorization) of a given tensor <img src="https://latex.codecogs.com/svg.latex?\Large&space;X" /> by minimization of the Frobenius norm:

<img src="https://latex.codecogs.com/svg.latex?\Large&space;\frac{1}{2}%7C%7C%20X-G\otimes_1A_1\otimes_2A_2\ldots\otimes_nA_n%7C%7C_F^2" />

<!-- X-G\otimes_1 A_1\otimes_2A_2\dots\otimes_nA_n_F^2 -->

where:

* <img src="https://latex.codecogs.com/svg.latex?\Large&space;n" /> is the dimensionality of the tensor <img src="https://latex.codecogs.com/svg.latex?\Large&space;X" />
* <img src="https://latex.codecogs.com/svg.latex?\Large&space;G" /> is a "mixing" core tensor
* <img src="https://latex.codecogs.com/svg.latex?\Large&space;A_1,A_2,\ldots,A_n" /> are "feature” factors (in the form of vectors or matrices)
* <img src="https://latex.codecogs.com/svg.latex?\Large&space;\otimes" /> is a tensor product applied to fold-in factors <img src="https://latex.codecogs.com/svg.latex?\Large&space;A_1,A_2,\ldots,A_n" />  in each of the tensor dimensions

<div style="text-align: center">
    <img src="figures/tucker-paper.png" alt="tucker" width=auto/>
</div>

The product <img src="https://latex.codecogs.com/svg.latex?\Large&space;G\otimes_1A_1\otimes_2A_2\ldots\otimes_nA_n" /> is an estimate of <img src="https://latex.codecogs.com/svg.latex?\Large&space;X" /> (<img src="https://latex.codecogs.com/svg.latex?\Large&space;X_{est}" />).

The reconstruction error <img src="https://latex.codecogs.com/svg.latex?\Large&space;X-X_{est}" /> is expected to be random uncorrelated noise.

<img src="https://latex.codecogs.com/svg.latex?\Large&space;G" /> is a <img src="https://latex.codecogs.com/svg.latex?\Large&space;n" />-dimensional tensor with a size and a rank lower than the size and the rank of <img src="https://latex.codecogs.com/svg.latex?\Large&space;X" />.
The size of tensor <img src="https://latex.codecogs.com/svg.latex?\Large&space;G" /> defines the number of extracted features (signals) in each of the tensor dimensions.

The factor matrices <img src="https://latex.codecogs.com/svg.latex?\Large&space;A_1,A_2,\ldots,A_n" /> represent the extracted features (signals) in each of the tensor dimensions.
The number of matrix columns equals the number of features in the respective tensor dimensions (if there is only 1 column, the particular factor is a vector).
The number of matrix rows in each factor (matrix) <img src="https://latex.codecogs.com/svg.latex?\Large&space;A_i" /> equals the size of tensor X in the respective dimensions.

The elements of tensor <img src="https://latex.codecogs.com/svg.latex?\Large&space;G" /> define how the features along each dimension (<img src="https://latex.codecogs.com/svg.latex?\Large&space;A_1,A_2,\ldots,A_n" />) are mixed to represent the original tensor <img src="https://latex.codecogs.com/svg.latex?\Large&space;X" />.

**NTFk** can perform Tensor Decomposition using [Candecomp/Parafac (CP)](https://en.wikipedia.org/wiki/Tensor_rank_decomposition) or [Tucker](https://en.wikipedia.org/wiki/Tucker_decomposition) decomposition models.

Some of the decomposition models can theoretically lead to unique solutions under specific, albeit rarely satisfied, noiseless conditions.
When these conditions are not satisfied, additional minimization constraints can assist the factorization.
A popular approach is to add sparsity and nonnegative constraints.
Sparsity constraints on the elements of G reduce the number of features and their mixing (by having as many zero entries as possible).
Nonnegativity enforces parts-based representation of the original data which also allows the Tensor Decomposition results for <img src="https://latex.codecogs.com/svg.latex?\Large&space;G" /> and <img src="https://latex.codecogs.com/svg.latex?\Large&space;A_1,A_2,\ldots,A_n" /> to be easily interrelated [Cichocki et al, 2009](https://books.google.com/books?hl=en&lr=&id=KaxssMiWgswC&oi=fnd&pg=PR5&ots=Lta2adM6LV&sig=jNPDxjKlON1U3l46tZAYH92mvAE#v=onepage&q&f=false).

### Publications:

- Vesselinov, V.V., Mudunuru, M., Karra, S., O'Malley, D., Alexandrov, B.S., Unsupervised Machine Learning Based on Non-Negative Tensor Factorization for Analyzing Reactive-Mixing, 10.1016/j.jcp.2019.05.039, Journal of Computational Physics, 2019. [PDF](https://gitlab.com/monty/monty.gitlab.io/raw/master/papers/Vesselinov%20et%20al%202018%20Unsupervised%20Machine%20Learning%20Based%20on%20Non-Negative%20Tensor%20Factorization%20for%20Analyzing%20Reactive-Mixing.pdf)
- Vesselinov, V.V., Alexandrov, B.S., O'Malley, D., Nonnegative Tensor Factorization for Contaminant Source Identification, Journal of Contaminant Hydrology, 10.1016/j.jconhyd.2018.11.010, 2018. [PDF](https://gitlab.com/monty/monty.gitlab.io/raw/master/papers/Vesselinov%20et%20al%202018%20Nonnegative%20Tensor%20Factorization%20for%20Contaminant%20Source%20Identification.pdf)
- O'Malley, D., Vesselinov, V.V., Alexandrov, B.S., Alexandrov, L.B., Nonnegative/binary matrix factorization with a D-Wave quantum annealer, PlosOne, 10.1371/journal.pone.0206653, 2018. [PDF](https://gitlab.com/monty/monty.gitlab.io/raw/master/papers/OMalley%20et%20al%202017%20Nonnegative:binary%20matrix%20factorization%20with%20a%20D-Wave%20quantum%20annealer.pdf)
- Stanev, V., Vesselinov, V.V., Kusne, A.G., Antoszewski, G., Takeuchi,I., Alexandrov, B.A., Unsupervised Phase Mapping of X-ray Diffraction Data by Nonnegative Matrix Factorization Integrated with Custom Clustering, Nature Computational Materials, 10.1038/s41524-018-0099-2, 2018. [PDF](https://gitlab.com/monty/monty.gitlab.io/raw/master/papers/Stanev%20et%20al%202018%20Unsupervised%20phase%20mapping%20of%20X-ray%20diffraction%20data%20by%20nonnegative%20matrix%20factorization%20integrated%20with%20custom%20clustering.pdf)
- Iliev, F.L., Stanev, V.G., Vesselinov, V.V., Alexandrov, B.S., Nonnegative Matrix Factorization for identification of unknown number of sources emitting delayed signals PLoS ONE, 10.1371/journal.pone.0193974. 2018. [PDF](https://gitlab.com/monty/monty.gitlab.io/raw/master/papers/Iliev%20et%20al%202018%20Nonnegative%20Matrix%20Factorization%20for%20identification%20of%20unknown%20number%20of%20sources%20emitting%20delayed%20signals.pdf)
- Stanev, V.G., Iliev, F.L., Hansen, S.K., Vesselinov, V.V., Alexandrov, B.S., Identification of the release sources in advection-diffusion system by machine learning combined with Green function inverse method, Applied Mathematical Modelling, 10.1016/j.apm.2018.03.006, 2018. [PDF](https://gitlab.com/monty/monty.gitlab.io/raw/master/papers/Stanev%20et%20al%202018%20Identification%20of%20release%20sources%20in%20advection-diffusion%20system%20by%20machine%20learning%20combined%20with%20Green's%20function%20inverse%20method.pdf)
- Vesselinov, V.V., O'Malley, D., Alexandrov, B.S., Contaminant source identification using semi-supervised machine learning, Journal of Contaminant Hydrology, 10.1016/j.jconhyd.2017.11.002, 2017. [PDF](https://gitlab.com/monty/monty.gitlab.io/raw/master/papers/Vesselinov%202017%20Contaminant%20source%20identification%20using%20semi-supervised%20machine%20learning.pdf)
- Alexandrov, B., Vesselinov, V.V., Blind source separation for groundwater level analysis based on non-negative matrix factorization, Water Resources Research, 10.1002/2013WR015037, 2014. [PDF](https://gitlab.com/monty/monty.gitlab.io/raw/master/papers/Alexandrov%20&%20Vesselinov%202014%20Blind%20source%20separation%20for%20groundwater%20pressure%20analysis%20based%20on%20nonnegative%20matrix%20factorization.pdf)

Research papers are also available at [Google Scholar](http://scholar.google.com/citations?user=sIFHVvwAAAAJ&hl=en), [ResearchGate](https://www.researchgate.net/profile/Velimir_Vesselinov) and [Academia.edu](https://lanl.academia.edu/monty)

### Presentations:

- Vesselinov, V.V., Physics-Informed Machine Learning Methods for Data Analytics and Model Diagnostics, M3 NASA DRIVE Workshop, Los Alamos, 2019. [PDF](http://monty.gitlab.io/presentations/Vesselinov%202019%20Physics-Informed%20Machine%20Learning%20Methods%20for%20Data%20Analytics%20and%20Model%20Diagnostics.pdf)
- Vesselinov, V.V., Unsupervised Machine Learning Methods for Feature Extraction, New Mexico Big Data &amp; Analytics Summit, Albuquerque, 2019. [PDF](http://monty.gitlab.io/presentations/vesselinov%202019%20Unsupervised%20Machine%20Learning%20Methods%20for%20Feature%20Extraction%20LA-UR-19-21450.pdf)
- Vesselinov, V.V., Novel Unsupervised Machine Learning Methods for Data Analytics and Model Diagnostics, Machine Learning in Solid Earth Geoscience, Santa Fe, 2019. [PDF](http://monty.gitlab.io/presentations/Vesselinov%202019%20GeoScienceMLworkshop.pdf)
- Vesselinov, V.V., Novel Machine Learning Methods for Extraction of Features Characterizing Datasets and Models, AGU Fall meeting, Washington D.C., 2018. [PDF](http://monty.gitlab.io/presentations/Vesselinov%202018%20Novel%20Machine%20Learning%20Methods%20for%20Extraction%20of%20Features%20Characterizing%20Datasets%20and%20Models%20LA-UR-18-31366.pdf)
- Vesselinov, V.V., Novel Machine Learning Methods for Extraction of Features Characterizing Complex Datasets and Models, Recent Advances in Machine Learning and Computational Methods for Geoscience, Institute for Mathematics and its Applications, University of Minnesota, 10.13140/RG.2.2.16024.03848, 2018. [PDF](http://monty.gitlab.io/presentations/Vesselinov%202018%20Novel%20Machine%20Learning%20Methods%20for%20Extraction%20of%20Features%20Characterizing%20Complex%20Datasets%20and%20Models%20LA-UR-18-30987.pdf)
- Vesselinov, V.V., Mudunuru. M., Karra, S., O'Malley, D., Alexandrov, Unsupervised Machine Learning Based on Non-negative Tensor Factorization for Analysis of Filed Data and Simulation Outputs, Computational Methods in Water Resources (CMWR), Saint-Malo, France, 10.13140/RG.2.2.27777.92005, 2018. [PDF](http://monty.gitlab.io/presentations/vesselinov%20et%20al%202018%20Unsupervised%20Machine%20Learning%20Based%20on%20Non-negative%20Tensor%20Factorization%20for%20Analysis%20of%20Filed%20Data%20and%20Simulation%20Outputs%20cmwr-ML-20180606.pdf)
- O'Malley, D., Vesselinov, V.V., Alexandrov, B.S., Alexandrov, L.B., Nonnegative/binary matrix factorization with a D-Wave quantum annealer [PDF](http://monty.gitlab.io/presentations/OMalley%20et%20al%202017%20Nonnegative%20binary%20matrix%20factorization%20with%20a%20D-Wave%20quantum%20annealer.pdf)
- Vesselinov, V.V., Alexandrov, B.A, Model-free Source Identification, AGU Fall Meeting, San Francisco, CA, 2014. [PDF](http://monty.gitlab.io/presentations/vesselinov%20bss-agu2014-LA-UR-14-29163.pdf)

Presentations are also available at [slideshare.net](https://www.slideshare.net/VelimirmontyVesselin), [ResearchGate](https://www.researchgate.net/profile/Velimir_Vesselinov) and [Academia.edu](https://lanl.academia.edu/monty)

### Videos:

- Progress of nonnegative matrix factorization process:

<div style="text-align: left">
    <img src="movies/m643.gif" alt="nmfk-example" width=60%  max-width=250px;/>
</div>

Videos are also available at [YouTube](https://www.youtube.com/playlist?list=PLpVcrIWNlP22LfyIu5MSZ7WHp7q0MNjsj)

### Patent:

Alexandrov, B.S., Vesselinov, V.V., Alexandrov, L.B., Stanev, V., Iliev, F.L., Source identification by non-negative matrix factorization combined with semi-supervised clustering, [US20180060758A1](https://patents.google.com/patent/US20180060758A1/en)

For more information, visit [monty.gitlab.io](http://monty.gitlab.io)

### Examples:

* [Machine Learning](https://madsjulia.github.io/Mads.jl/Examples/machine_learning/index.html)
* [Blind Source Separation (i.e. Feature Extraction)](https://madsjulia.github.io/Mads.jl/Examples/blind_source_separation/index.html)
* [Source Identification](https://madsjulia.github.io/Mads.jl/Examples/contaminant_source_identification/index.html)

Installation behind a firewall
------------------------------

Julia uses git for package management. Add in the `.gitconfig` file in your home directory:

```
[url "https://"]
        insteadOf = git://
```

or execute:

```
git config --global url."https://".insteadOf git://
```

Julia uses git and curl to install packages. Set proxies:

```
export ftp_proxy=http://proxyout.<your_site>:8080
export rsync_proxy=http://proxyout.<your_site>:8080
export http_proxy=http://proxyout.<your_site>:8080
export https_proxy=http://proxyout.<your_site>:8080
export no_proxy=.<your_site>
```

For example, if you are doing this at LANL, you will need to execute the
following lines in your bash command-line environment:

```
export ftp_proxy=http://proxyout.lanl.gov:8080
export rsync_proxy=http://proxyout.lanl.gov:8080
export http_proxy=http://proxyout.lanl.gov:8080
export https_proxy=http://proxyout.lanl.gov:8080
export no_proxy=.lanl.gov
```
