# Current density and gamma rate PCA for the CMS RPCs at the GIF++

**Author:** Alexis Fagot
**Mail:** alexis.fagot@SPAMNOTugent.be

## Requirements

This analysis algorithm requires Python3. Required libraries:

* **[Scikit-learn](https://scikit-learn.org/stable/install.html)**
* **[SciPy](https://www.scipy.org/install.html)** which contains *Numpy*, *Matplotlib* and *Pandas*, all used in the analysis

## Usage

As it is and for the needs of this specific analysis, the algorithm is only running through a single data file. Hence, the data file was hardcoded into the first lines of the code. To use, simply type:

    python3 PCA-analysis.py

## Output

The algorithm produces multiple plots:

* a *[Scree plot](https://en.wikipedia.org/wiki/Scree_plot)* showing the explained variance for all of the principal components
* a series of 4D plots, one for each component of the data phase space, in which the three main dimensions are the three first dimensions of the principal components phase space and the forth is the value held by the component in the data phase space.

The algorithm also produces a latex format table with the scalar products of the different principal components with respect to the others (this was to put into an article).
