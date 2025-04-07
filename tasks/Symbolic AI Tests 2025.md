**Tasks for Prospective GSoC 2025 Applicants for the**   
***Symbolic Calculation Projects***

\--------------------------------------------------------------------------------------------------------------------

**Submission Guidelines**	

* **IN ORDER TO BE CONSIDERED AS AN APPLICANT FOR GOOGLE SUMMER OF CODE YOU MUST ALSO SUBMIT A PROPOSAL THROUGH THE GOOGLE SUMMER OF CODE PORTAL**  
* Additional tasks are listed below  
* You are required to submit Jupyter Notebooks for each task clearly showing your implementation.  
* Please also put your solution in a github repository and send us a link.  
* You must calculate and present the required evaluation metrics.  
* When completed, please send your **CV**, a link to your **solution repository**, your **notebooks** and **trained model weights** to [ml4-sci@cern.ch](mailto:ml4-sci@cern.ch) with Evaluation Test: SYMBA in the title

**References:**

*AI Feynman: a Physics-Inspired Method for Symbolic Regression*, Udrescu & Tegmark (2019), [arXiv:1905.11481](https://arxiv.org/abs/1905.11481).

Below are the tasks that we will use to evaluate prospective students for [***Symbolic Calculation Projects***](https://ml4sci.org/gsoc/projects/2024/project_SYMBA.html).  After completing the first common test, please thoroughly complete the specific second test for your project of interest and (optionally) other second tests if you would like to also be considered for additional Symbolic Calculation projects at the same time (this may increase your chances of success, but make sure you don't do it at the expense of the specific project you are interested in)

**Note:** please work in your own github branch (i.e. NO PRs should be made). Send us a link to your code when you are finished and we will evaluate it. 

**Common Task 1:**  
Do part 1.1 for projects related to symbolic regression and 1.2 for projects related to square amplitude calculation.

**Common Task 1.1.** **Dataset preprocessing**   
**Dataset**:

[https://space.mit.edu/home/tegmark/aifeynman.html](https://space.mit.edu/home/tegmark/aifeynman.html)   
**Note: The authors of this dataset are not affiliated with ML4SCI**

**Description:**  
Download the Feynman\_with\_units.tar.gz features and corresponding FeynmanEquations.csv targets. Preprocess and tokenize the target data and document your rationale for choice of tokenization.

**Common Task 1.2 Dataset preprocessing**  
**Dataset:**

[https://alabama.box.com/s/xhgr2onrn503jyse2fs5vxtapg0oifcs](https://alabama.box.com/s/xhgr2onrn503jyse2fs5vxtapg0oifcs) 

**Dataset:**  
Download the dataset (split across 10 files) and preprocess and tokenize the target data and document your rationale for choice of tokenization. Data file is formatted with rows like   
“event type : Feynman diagram : amplitude : squared amplitude”  
Here the amplitudes are the input sequences and squared amplitudes are the target sequences. Note that indices like \_123456 grow over the course of the dataset and should be normalized for each amplitude and squared amplitude. Use an 80-10-10 split of train-val-test across all files.

**Common Task 2: Train/Evaluate Transformer model**  
Train a generic next-token-prediction Transformer model to map the input data to the tokenized output sequences. Evaluate performance on the test set using sequence accuracy as a metric.

**Specific Test 3: Train/Evaluate advanced model**  
Repeat task two including checking sequence accuracy but with a model that leverages some slightly more advanced techniques. The model you use should relate to the project you’re applying for.

**3.1: Next-Generation Transformer Models for Symbolic Calculations of Squared Amplitudes in HEP**  
Model: Transformer model with a contemporary innovation added such as KAN layers, reinforcement learning, genetic algorithms, specialized long-sequence attention, etc. which improves the performance compared to a basic transformer.

**3.2: State-space Models for Squared Amplitude Calculation in High-Energy Physics**  
Model: State-space model such as mamba or other model.

**3.3: Transformer Models for Symbolic Regression**  
Model: Transformer model with a contemporary innovation added such as KAN layers, reinforcement learning, genetic algorithms, specialized long-sequence attention, etc. which improves the performance compared to a basic transformer.

**3.4: Titans for squared amplitude calculation**  
Model: One of the core architectures from Google’s paper introducing Titans concept

**3.5: Evolutionary and Transformer Models for Symbolic Regression**  
Model: Transformer model integrated with an evolutionary pipeline. It’s possible to start from previous year’s projects but should introduce a substantial innovation.

**3.6: Symbolic empirical representation of squared amplitudes in high-energy physics**  
Model: Transformer with novel approach for tokenization, data representation and/or preprocessing that leads to better performance than basic tokenization with normalized indices.

**3.7: Foundation models for symbolic regression tasks**  
Model: Novel foundation model for symbolic regression tasks. Should be sufficiently novel beyond the current literature.