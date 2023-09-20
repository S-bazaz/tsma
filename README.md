# tsma (time series model analyses)

This package offers tools for analyzing models that generate multivariate time series as output. 
The goal is to make the study of large models, especially those related to macroeconomic ABMs, more convenient.

## References:

This package contains an implementation of two papers: 

- Gross's model of 2022 used as example in the tutorials : https://doi.org/10.1016/j.ecolecon.2010.03.021
- Vandin & al statistical checking of 2021 : http://arxiv.org/abs/2102.05405

To learn more about signature transform see Adeline Fermanian's thesis of (2021) or https://arxiv.org/pdf/1911.13211.pdf

## Structure:

### model 

- **model** : model class that serves as baseline
- **gross2022** : implementation of Gross's model of 2022
![image](https://github.com/S-bazaz/tsma/assets/108877488/b64f0174-f149-4af8-9eae-25d9d9a03f68)

### collect
 
- **output_management** : This package includes functions for encoding and decoding parameters into databases, as well as storing and managing simulation data.
- **iterators** : functions used to gather simulations results
- **data_collect** :  allows for the exploration of parameters through statistical analyses, with the option for automatic saving

### visuals

- **fig_constructors** : assists in the creation of figures using Plotly, Seaborn, and Networkx
- **figures** : creates figures using Plotly, Seaborn, and Networkx
- **fig_management** : manages the creation and saving of multiple figures
- **dashboards** : for creating a dashboard on a local server, which allows for the visualization of a summary of a simulation with various sets of parameters.
![image](https://github.com/S-bazaz/tsma/assets/108877488/4bb7676d-b4d0-41ff-b952-81a0a5fe83ec)

### analyses

- **statistics** : Means, Stationarity tests, ...
- **vandin** : implementation of statistical analyses taken from : Vandin & al (2021)
- **metrics** : classes used to create clustering approaches for time series clustering
- **clustering** : 
    - multivariate time series unsupervised clustering
    - clustering scoring and selection
    - phase diagram visualization ( parameters space projection of clustering results )
    - comparison of clusters
![image](https://github.com/S-bazaz/tsma/assets/108877488/52f52609-ad79-4b35-8004-556f44e35d35)

### basics

- **transfers** : functions used to divide and join strings, lists and dictionaries
- **text_management** : name conversions, title creations ...

## Tutorials:

Find tutorials at : https://github.com/S-bazaz/tsma

- **tsma1** : how to format your model and use the saving system (save, overwrite, query ...)
- **tsma2** : how to use visualizations ( simple plots, dashboards ...)
- **tsma3** : vandin & al statistical checking and data collection (transient analysis, parameters exploration ...)
- **tsma4** : time series clustering ( clusters computation, clustering selection, phase diagram ...)
- **tsma5** : clustering comparison ( visualization per cluster, dendrograms, Jaccard coefficients network plots ...)
