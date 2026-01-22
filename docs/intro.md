# The Chalmers Cloud Ice Climatology (CCIC)

The Chalmers Cloud Ice Climatology (CCIC) is a novel, deep-learning-based
climate record of ice-particle concentrations in the atmosphere. CCIC results
are available at high spatial and temporal resolution (0.07° / 3 h from 1983,
0.036° / 30 min from 2000) and thus ideally suited for evaluating
high-resolution weather and climate models or studying individual weather
systems. Furthemore, CCIC estimates are the most accurate compared to
CloudSat/CALIPSO-based reference estimates, in terms of global mean values,
long-term trends and spatial variability {cite:p}`pfreundschuh_2025_ccic`.


## Example results

### Recent results

The animation below shows the evolution of TIWP retrieved from the most recent available satellite observations.

<div style="width:100%;height:100%;overflow:hidden;"> 
  <video src="https://rain.atmos.colostate.edu/gprof_nn/ccic/tiwp.mp4" controls="controls" style="width:80%;" title="TIWP evolution for January 2020">
  </video>
</div>

### 3D structure of Typhoon Nanmadol

CCIC also produces vertically resolved estimates of the total ice water content,
i.e., the 3D distribution of ice particle concentrations in the atmosphere. The
image below shows the 3D structure of ice particles in typhoon Nanmadol on 19
September 2022 shortly before making landfall in Japan. An interactive version
of these results can be found [here](http://spfrnd.de/data/nanmadol.html).

```{figure} images/nanmadol.png
---
alt: Iso surfaces of total ice water content (TIWP) in Typhon Nanmadol on 19 September 2022 12:00:00 UTC shortly before making landfall in Japan.
width: 100%
align: center
---
Iso surfaces of total ice water content (TIWP) in Typhon Nanmadol on 19 September 2022 12:00:00 UTC shortly before making landfall in Japan.
```


## Relevance

The TIWP quantifies the amount of ice particles in clouds. Ice particles are formed in clouds that are high or thick enough to reach temperatures where cloud droplets freeze. High TIWP values occur in thick clouds typically produced by storms. Estimates of TIWP are therefore useful to study the cloud processes involved in the formation of these storms. Moreover, since ice particle reflect and absorb radiation, the distribution of TIWP playes an important role in regulating the Earth's radiative energy balance.

CCIC's TIWP estimates have been thoroughly validated and shown to be consistent with in-situ air-borne and ground-based cloud-radar measurements {cite:p}`amell_2023_ccic`. CCIC is the only thoroughly validated, high-resolution TIWP data record with temporally and spatially continuous coverage.

In addition, CCIC can provide other cloud-related variables, such as the 3D distribution of ice-particle concentrations in the atmosphere (the total ice water content, TIWC), two-dimensional and three-dimensional cloud probabilities, and three-dimensional cloud classification. Due to storage limitations, these products are not currently distributed with the CCIC data record. However, users can access these estimates by running their retrievals locally using the ``ccic`` Python package.


## Applications

The two principal applications of CCIC are

 1. the study of cloud processes in individual weather systems,
 2. the validation of climate and weather models.
 
Example applications of CCIC can be found in the {doc}`applications` chapter.


## Citing CCIC

If you are using CCIC for research or other purposes, please cite the following publication:

Amell, A., Pfreundschuh, S., and Eriksson, P.: The Chalmers Cloud Ice Climatology: Retrieval implementation and validation, EGUsphere [preprint], https://doi.org/10.5194/egusphere-2023-1953, 2023.
