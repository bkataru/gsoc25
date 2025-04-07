ML4SCI DeepFalcon GSoC 2025 Tasks 

Below are the tasks that we will use to evaluate prospective students for the DeepFalcon projects. After completing the first 2 common tasks, please thoroughly complete the specific third task for your project of interest and (optionally) any other third task if you would like to also be considered for all DeepFalcon projects (this may increase your chances of success, but make sure to not do it at the expense of the specific project you are more interested in)

**Note:** please work in your own github branch (i.e. NO PRs should be made). Send us a link to your code when you are finished and we will evaluate it. 

**Dataset**: Data of Quark/Gluon jet events available [here](https://drive.google.com/file/d/1WO2K-SfU2dntGU4Bb3IYBp9Rh7rtTYEr/view?usp=sharing). The dataset consists of 3 channels (ECAL, HCAL and Tracks) each containing 125x125 images.

**Common Task 1\. Auto-encoder of the quark/gluon events**

* Please train a variational auto-encoder to learn the representation based on three image channels (ECAL, HCAL and Tracks) for the dataset. 

* Please show a side-by-side comparison of original and reconstructed events. 

**Common Task 2\. Jets as graphs** 

* Please choose a graph-based GNN model of your choice to classify (quark/gluon) jets. Proceed as follows:  
1. Convert the images into a point cloud dataset by only considering the non-zero pixels for every event.  
2. Cast the point cloud data into a graph representation by coming up with suitable representations for nodes and edges.  
3. Train your model on the obtained graph representations of the jet events.  
*  Discuss the resulting performance of the chosen architecture. 

**Specific Task 1 (if you are interested in “Graph Representation Learning for Fast Detector Simulation” project):**

* **Please train a simple graph autoencoder on this dataset.** Please show a visual side-by side comparison of the original and reconstructed events and appropriate evaluation metric of your choice. Compare to the VAE model results.

**Specific Task 2 (if you are interested in “Diffusion Models for Fast Detector Simulation” project):**

* Use a Diffusion Network model to represent the events in task 1\. Please show a side-by side comparison of the original and reconstructed events and appropriate evaluation metric of your choice that estimates the difference between the two.

**Specific Task 3 (if you are interested in “Graph Transformers for Fast Detector Simulation” project):**

* Use a transformer model of your choice as a generative model. Check the existing literature for ideas on transformer models applied to generative tasks. Train it on the events of task 1 and evaluate its generative performance. 

**Specific Task 4 (if you are interested in “Optimal Transport for HEP” project):**

* Build an autoencoder with an architecture of your choice and apply it to the MNIST dataset while choosing 2 digits of your choice (0 and 4, 1 and 9 , etc). Proceed in doing 2 variations of this autoencoder:   
1. Map the latent space representation into a standard normal distribution using optimal transport and sample from it to the decoder.  
2. (Bonus) Map a random Gaussian noise vector to the latent space representation and sample from it to the decoder.  
* Apply a similar model to the dataset in Task 1

\--------------------------------------------------------------------------------------------------------------------

**Test Submission Instructions:** Please send us your CV and a link to all your completed work (github repo, Jupyter notebook \+ pdf of Jupyter notebook with output) to [ml4-sci@cern.ch](mailto:ml4-sci@cern.ch) with Evaluation Test: DeepFalcon in the title.

