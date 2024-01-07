# The Chalmers Cloud Ice Climatology (CCIC)

The Chalmers Cloud Ice Climatology (CCIC) is a novel, deep-learning-based
climate record of ice-particle concentrations in the atmosphere. CCIC results
are available at high spatial and temporal resolution (0.08 ° / 3 h from 1983,
0.036 ° / 30 min from 2000) and thus ideally suited for evaluating
high-resolution weather and climate models or studying individual weather
systems.


## Example results

The animation below shows the evolution of column-integrated ice particle
concentrations, the total ice water path (TIWP), for the first week in January
2020.

<div style="width:100%;height:100%;overflow:hidden;"> 
  <video src="https://user-images.githubusercontent.com/28195522/266384146-9e1e89eb-1b68-46a7-8c63-3788b41b73d7.mp4" controls="controls" style="width:80%;" title="TIWP evolution for January 2020">
  </video>
</div>

## Relevance

The TIWP quantifies the amount of ice particles in clouds. High TIWP values occur in clouds that are high or thick enough to reach temperatures where cloud droplets freeze. TIWP thus traces clouds in the atmosphere and can be used to track and study storms. Moreover, by reflecting and absorbing radiation, ice clouds play a crucial role in regulating the Earth's radiative energy balance.

CCIC's TIWP estimates have been thoroughly validated and shown to be consistent with in-situ air-borne and ground-based cloud-radar measurements {cite:p}`amell_2023_ccic`. CCIC is the only thoroughly validated, high-resolution TIWP data record with temporally and spatially continuous coverage.

In addition, CCIC can provide other cloud-related variables, such as the 3D distribution of ice-particle concentrations in the atmosphere (the total ice water content, TIWC), two-dimensional and three-dimensional cloud probabilities, as well as a three-dimensional cloud classification. Due to storage limitations, these products are not currently distributed with the CCIC data record. However, users can access these estimates by running their retrievals locally using the ``ccic`` Python package.



## Applications

The two principal applications of CCIC are

 1. the study of cloud processes in individual weather systems,
 2. the validation of climate and weather models.
 
An example of how CCIC can be used to track and analyze storms is provided in link.


## References

```{bibliography}
```
