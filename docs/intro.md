# The Chalmers Cloud Ice Climatology (CCIC)

The Chalmers Cloud Ice Climatology is a novel, deep-learning-based climate record of ice-particle-concentrations in the atmosphere.
It provides spatially and temporally continuous coverage for most longitudes and latitudes between 60º S and 60º N.

## Example results

The animation below shows the evolution of column-integrated ice particle
concentrations, the total ice water path (TIWP), for the first week in January
2020.

<div style="width:100%;height:100%;overflow:hidden;"> 
  <video src="https://user-images.githubusercontent.com/28195522/266384146-9e1e89eb-1b68-46a7-8c63-3788b41b73d7.mp4" controls="controls" style="width:80%;" title="TIWP evolution for January 2020">
  </video>
</div>

## Relevance

The TIWP quantifies the amount of ice particles in clouds. Non-zero TIWP occurs
where water vapor is lifted up high-enough in the atmosphere to condense and
temperatures are low enough and/or concentrations of ice-nucleating particles
are high for cloud droplets to freeze. TIWP thus traces intensive convective
activity. Furthermore, ice particles play a crucial role in regulating the
radiative energy balance of the atmosphere.

The CCIC TIWP estimates have been thoroughly validated and shown to be
consistent with in-situ and air-borne and ground-based cloud-radar measurements
{cite:p}`amell_2023_ccic`. To the best of our knowledge, the CCIC TIWP 
is the only thoroughly-validated, high-resolution climate data record of TIWP
that is currently available.

In addition, CCIC can provide other cloud-related variables such as
the 3D distribution of ice-particle concentrations in the atmosphere (the total
ice water content, TIWC), two-dimensional and three-dimensional cloud
probabilities, as well as a three dimensional cloud classification. Due to
storage limitations, these products are not currently distributed with the CCIC
data record. However, users can access these estimates by running their own
retrievals locally.


## Applications

The CCIC dataset comprises multidecadal AI-powered cloud retrievals continuously covering all longitudes between 60º S and 60º N. A convolutional neural network is used to infer cloud properties from only one channel of geostationary satellite images from either the CPCIR {cite}`Janowiak_2001_RealTime, Janowiak_2017_CPCIR` or GridSat {cite}`Knapp_2014_CDR, knapp_2011_gridsat` data products. The retrieval implementation and its validation has been described in an pre-print recently published in EGUSphere {cite}`amell_2023_ccic`. Suggested analyses of the CCIC data include, among others, long-term climate analyses and studies of cloud processes.


<div style="width:100%;height:100%;overflow:hidden;"> 
  <video src="https://user-images.githubusercontent.com/28195522/266384146-9e1e89eb-1b68-46a7-8c63-3788b41b73d7.mp4" controls="controls" style="width:100%;height:100%;" title="TIWP evolution for January 2020">
  </video>
</div>




## References

```{bibliography}
```
