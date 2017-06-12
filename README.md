# spinchain-virasoro-demo
Demonstration code accompanying https://arxiv.org/abs/1706.01436

The notebooks generate the plots used in the paper from precomputed data. They also contain some additional plots which explore the matrix elements of the lattice generators Hn in more detail.

The notebooks were tested with Julia 0.5 and the plot code is made for PyPlot using matplotlib 2.

To regenerate the raw data, run the appropriate `ED_model.jl` script. Note that these scripts are *not* highly optimized and require a large amount of memory (up to approx. 20Gb) to run and a similar amount of disk storage if the raw eigenvectors are to be stored. With sufficient memory, however, they should complete in a matter of hours on a fast computer. 

Reducing the maximum system size will reduce resource requirements.