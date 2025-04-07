**GSoC EXXA Test**  
2025  
ML4Sci

The purpose of EXXA is to use machine learning to analyze exoplanets, their systems, and their formation. 

Astronomical data often takes the form of images, so expertise in computer vision techniques is often essential for understanding and working with observations. Telescopes, such as [ALMA](https://almascience.nrao.edu/), [Kepler](https://archive.stsci.edu/kepler/data_search/search.php) and [NASA](https://exoplanetarchive.ipac.caltech.edu/), produce [publicly available data](https://almascience.nrao.edu/aq/?result_view=project) that can be used (with proper citations) by amateur and professional researchers alike. However, research often supplements this data with synthetic data from simulations.

For this project, participants will be required to complete two tasks: one general test and one specific to their subproject (i.e., EXXA1, EXXA2, etc.). EXXA4 (“Foundation Models for Exoplanet Characterization”) must complete all three.

**General Test**

This test will focus on using computer vision techniques on synthetic observations produced from hydrodynamics simulations and radiative transfer calculations.

**Overview**

Protoplanetary disks are the sites of planet formation. The most recent generation of telescopes is able to resolve these objects in sufficient detail for the rigorous study of their properties, leading to a dramatic and rapid advancement in planet formation theory. Using synthetic observations that mimic data obtained from these observatories has allowed researchers to understand how specific conditions will manifest themselves in observations. See [Terry et al. (2022)](https://iopscience.iop.org/article/10.3847/1538-4357/aca477) for an example of synthetic data creation and use.

**Task**

Using provided synthetic continuum observations of ALMA (1250 microns), found [here](https://drive.google.com/drive/folders/1VkS3RHkAjiKjJ6DnZmEKZ_nUv4w6pz7P?usp=sharing), create a machine learning model capable of unsupervised clustering of the disks. No labels will be provided. The data is provided as .fits files, the standard format of observational data. [Astropy](https://www.astropy.org/) is a particularly useful package for working with .fits data. Each image is a data cube containing 4 600x600 layers. Only the first one is relevant, i.e., index 0\. 

The successful completion of this test will result in an automated pipeline of data loading, loading a trained model created by the participant, and passing the data through the model to create, label, and visualize clusters of the synthetic data. There are many ways in which this data can be clustered, but the number of planets/presence of any planets is of particular interest. Many of the images show clear signatures of one or more planets, so visual inspection is a useful tool. Beware of simply clustering the disks by viewing angle.

**Deliverables**

* Script (ideally Jupyter Notebook) that includes the data-loading process and model creation, training, and testing. The script should be able to be run from start to finish without user intervention. It should produce clear plots that allow quick performance analysis. Data may be augmented in whatever manner you please, but that process should be automated to allow us to test using withheld data. Google Colab the preferred method of sharing the results.  
* If the training process is long, please deliver a pre-trained model of yours, ideally through a Google Drive. This is preferred regardless of training time in order to facilitate the evaluation of the task.

**Metrics**

Models will be judged on the clarity of clusters produced and the properties that the clusters find. The judges have all data pertaining to the disks and will run analyses of which properties are the most important in determining the clusters. Ideally, the clusters will correspond to properties pertaining to planets.

The quality of the cluster presentation and labeling will be taken into account. A clear presentation and data labeling that facilitates easy study will be judged highly.

The choice of model is of secondary importance to performance, but participants should use state-of-the-art models that are appropriate for this task.

Code should be clear, well-documented, and run without bugs or significant end-user modification.

**Image-Based Test**

This test is for the image-based EXXA projects (EXXA1, EXXA2, and EXXA4), i.e., those that deal with protoplanetary disks. 

**Task**

Using the [same dataset as the general test](https://drive.google.com/drive/folders/1VkS3RHkAjiKjJ6DnZmEKZ_nUv4w6pz7P?usp=sharing), train an autoencoder to output the images resembling the inputs. The overall architecture of the model is up to the participant, but there *must be an accessible latent space*, i.e., a user should be able to feed in an image and access it when it has been encoded in the latent space.

**Metrics**

**Quantitative:**

* MSE between input and output  
* Multiscale SSIM between input and output (e.g., [pytorch-msssim](https://github.com/jorge-pessoa/pytorch-msssim) or [PyTorch Lightning MS-SSIM](https://lightning.ai/docs/torchmetrics/stable/image/multi_scale_structural_similarity.html))

Qualitative metrics are the same as the general test. The models will be tested on withheld data in the same format.

**Deliverable**

Same as the general test. A user should be able to run inference on the entire pipeline with new data and should also be able to access an encoded image in the latent space easily.

**Sequential Test**

This test is for the projects that rely on sequential data that may need to be generated by the participant. This applies to EXXA3, EXXA4, and EXXA5, i.e., those that deal with spectra

**Overview**

One of the most successful methods to detect exoplanets is using light curves. Several thousand planets have been discovered this way. The basic idea is that exoplanets crossing in front of their host stars will obscure part of the star, which decreases the amount of light that we see from that star. By carefully measuring the brightness over time, planets can be identified by the periodic dimming. The extent of the dimming depends on the specific parameters of the stellar system. For a basic introduction, see [this blog](https://avanderburg.github.io/tutorial/tutorial.html).

**Task**

Using these concepts, create a simulated dataset of transit curves. Include as many physical and system parameters that you think are necessary. [PyTransit](https://github.com/hpparvi/PyTransit) is a good example package. Participants can use it, any other package they find, or make their own. Feel free to supplement the synthetic data with observational data. Use this data to train a classifier that determines whether or not a given transit curve shows the presence of a planet. 

**Deliverables**

* Same as the general test.

**Metrics**

**Quantitative:**

* ROC curves and calculated AUC  
* Testing data will include *real observations* so take into account that noisy data will be used to judge


Qualitative metrics are the same as the general test. The models will be tested on withheld data in the same format.

**Deliverable**

Same as the general test. A user should be able to run inference on the entire pipeline using withheld data with minimal effort.

