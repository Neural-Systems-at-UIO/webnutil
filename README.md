# Introduction to Webnutil-service

Webnutil-service is a Python library for performing spatial analysis of labelling in histological brain sections with respect to a reference brain atlas. It aims to replicate the Quantifier feature of the Nutil software (RRID: SCR_017183) for use in an online version of the QUINT workflow. It is built around PyNutil: https://github.com/Neural-Systems-at-UIO/PyNutil. For more information about the QUINT workflow: https://quint-workflow.readthedocs.io/en/latest/.

# What does the webnutil-service do?

It takes two sets of input:
* Atlas-registration.json from QuickNII and VisuAlign or WebAlign and WebWarp
* Segmented brain section images revealing the labelling to be quantified in a unique RGB colour code.

Webnutil-service aims to identify segmented objects, register them to reference atlas regions, and quantify the regions, objects per region, and area fraction per region. It also assigns reference atlas coordinates to each object pixel for visualising the objects in 3D reference atlas space. 

Output:

* Reports with region area, object count per region, object area per region and area fraction.
* Point cloud in reference atlas space representing the segmented objects.

# Technical details

## How are regions defined? 

Webnutil-service creates atlas maps for each brain section internally using the linear and nonlinear markers in the atlas-registration.json. It uses these to define regions in the segmentations, which it uses to measure region areas in pixels. If the segmentations are larger than the atlas maps, it scales up the atlas maps to the size of the segmentations, then uses opencv nearest_neighbour to measure region areas: https://docs.opencv.org/3.4/da/d54/group__imgproc__transform.html. 

*Known issues*: When the segmentations have the same size as the images used for atlas registration, the results from webnutil-service and Nutil are identical. When the segmentations are larger than the images used for atlas registration, the region areas from webnutil-service and Nutil are different. Nutil does this calculation differently. It calculates a scaling factor and multiplies the region areas by the scaling factor to calculate new region areas. 

**Consider changing how webnutil-service performs this calculation to match Nutil**. This will make it easier to perform validation as results can be compared directly to Nutil.

*Validation*: Atlas map creation by webnutil-service is correct and has been validated for several datasets (test 1, synthetic dataset and test 2, ttA_NOP dataset requiring scaling of the atlas maps). Note that the atlas maps match the atlas maps created by VisuAlign even when there are no nonlinear adjustments (QuickNII produces slightly different atlas maps to VisuAlign). This is documented here: https://github.com/Neural-Systems-at-UIO/PyNutil/issues/38 

## How are objects defined and assigned to regions?

Webnutil-service uses skimage.measure.label to define objects using label connected regions of an integer array (1-connectivity) (https://scikit-image.org/docs/0.25.x/api/skimage.measure.html#skimage.measure.label). 

![image](https://github.com/user-attachments/assets/93cededf-b2e4-4c0d-846a-ad0d372ab08f)

It then uses the geometric center of the objects to assign them to regions using the default centroid method in scikit image.

![image](https://github.com/user-attachments/assets/c63f1ad6-306a-4db1-8110-929216fe6c52)

![image](https://github.com/user-attachments/assets/5e255cea-9ed5-4fa2-b40a-a8791b1eeff5)

*Known issues*: The total number of objects counted by webnutil-service and Nutil for the ttA_NOP test dataset differs, suggesting they may use different methods for defining objects. 

**How does Nutil define objects? - Look this up in the Nutil code.**

*Known issues*: For the synthetic test dataset which has segmentations of the same size as the images used for atlas-registration, the "object counts per region" using webnutil-service and Nutil are identical and correct. For test datasets with segmentations larger than the images used for atlas-registration, the "object counts per region" from webnutil-service are incorrect (I manually counted for one region, results were way off). Something is going wrong with assigning objects to regions when atlas maps are scaled. I also confirmed this for the synthetic dataset doubled in size. 

**To be investigated.** 

We also known that Nutil uses a different method for assigning objects to regions. 

**How does Nutil assign objects to regions? - Look this up in the Nutil code**.

## How are object areas (pixel_count) per region calculated?

To correctly calculate object areas per region (and area fraction), Nutil has a feature called "area_splitting". This means that Nutil assigns each object pixel to its overlapping region, then calculates object pixels per region / region area (as opposed to assigning objects using the geometric center and then dividing object area /region area, which will give incorrect results when objects overlap several atlas regions). For webnutil-service, the intention was to implement pixel_count by the same method as Nutil with area splitting.

**To be validated**

## How are area fractions calculated? 

In webnutil_service, area fraction = pixel_count/ region_area. 

**This has been validated.**



