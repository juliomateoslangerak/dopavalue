## Image analysis

We manually predefined and extracted the different regions of interest to restrict measurements to the desired 
anatomical domains and to reduce the volume of data to analyze.
We segmented fibers using Ilastik v1.4 (1) on mean intensity projections of the raw images while measurements were 
taken on average intensity projections. A set of Python scripts was developed to orchestrate the segmentation, 
extract measurements and data management using OMERO (2) as a data management backend. These scripts as well as the
detailed usage instructions can be found at https://github.com/juliomateoslangerak/dopavalue


## references
(1) Berg, S., Kutra, D., Kroeger, T. et al. ilastik: interactive machine learning for (bio)image analysis. 
Nat Methods 16, 1226–1232 (2019). https://doi.org/10.1038/s41592-019-0582-9

(2) Allan, C., Burel, JM., Moore, J. et al. OMERO: flexible, model-driven data management for experimental biology. 
Nat Methods 9, 245–253 (2012). https://doi.org/10.1038/nmeth.1896
