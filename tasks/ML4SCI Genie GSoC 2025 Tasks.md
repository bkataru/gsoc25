ML4SCI Genie GSoC 2025 Tasks 

Below are the tasks that we will use to evaluate prospective students for the Genie projects. After completing the first 2 common tasks, please thoroughly complete the specific third task for your project of interest and (optionally) any other third task if you would like to also be considered for all Genie projects (this may increase your chances of success, but make sure to not do it at the expense of the specific project you are more interested in)

**Note:** please work in your own github branch (i.e. NO PRs should be made). Send us a link to your code when you are finished and we will evaluate it. 

**Dataset**: Data of Quark/Gluon jet events available [here](https://drive.google.com/file/d/1WO2K-SfU2dntGU4Bb3IYBp9Rh7rtTYEr/view?usp=sharing). The dataset consists of 3 channels (ECAL, HCAL and Tracks) each containing 125x125 images.

**Common Task 1\. Auto-encoder of the quark/gluon events**

* Please train an auto-encoder to learn the representation based on three image channels (ECAL, HCAL and Tracks) for the dataset. 

* Please show a side-by-side comparison of original and reconstructed events. 

**Common Task 2\. Jets as graphs** 

* Please choose a graph-based GNN model of your choice to classify (quark/gluon) jets. Proceed as follows:  
1. Convert the images into a point cloud dataset by only considering the non-zero pixels for every event.  
2. Cast the point cloud data into a graph representation by coming up with suitable representations for nodes and edges.  
3. Train your model on the obtained graph representations of the jet events.  
*  Discuss the resulting performance of the chosen architecture. 

**Specific Task 1 (if you are interested in “Deep Graph Anomaly Detection with Contrastive Learning” project):**

* Classify the quark/gluon data with a model that learns data representation with a contrastive loss.  
* Evaluate the classification performance on a test dataset.

**Specific Task 2 (if you are interested in “Learning Parametrization with Implicit Neural Representations” project):**

* Use an INR model of your choice to represent the events in task 1\. Please show a visual side-by side comparison of the original and reconstructed events and appropriate evaluation metric of your choice.

**Specific Task 3 (if you are interested in “Learning the Latent Structure with Diffusion Models” project):**

* Use a Diffusion Network model to represent the events in task 1\. Please show a side-by side comparison of the original and reconstructed events and appropriate evaluation metric of your choice that estimates the difference between the two.

**Specific Task 4 (if you are interested in “Non-local GNNs for Jet Classification” project):**

* Build a non-local GNN model to complete task 2\. Compare results for non-local GNN and baseline GNN using ROC-AUC as a metric.

\--------------------------------------------------------------------------------------------------------------------

**Test Submission Instructions:** Please send us your CV and a link to all your completed work (github repo, Jupyter notebook \+ pdf of Jupyter notebook with output) to [ml4-sci@cern.ch](mailto:ml4-sci@cern.ch) with Evaluation Test: Genie in the title.

