# **RP_KPLS-IS**
This repo holds all the codes that was involved in the Research Project.

# DESCRIPTION OF EACH FILE IN THE REPO -->

* The code named << **40,100 linear - AKIS** >> is the AKIS model that works on high dimensional problem which is scalable. By default it is set at 40 dimensions. The variable that holds the dimension value is << stochastic_dim >>.
* The code named << **40,100 linear - KPLS** >> is the KPLS model that works on high dimensional problem which is scalable too. The stopping criteria can be seen in the function named << **exploration()** >>. The code is built specifically for the problem in hand. While adapting the code to a new problem, take caution in modifying the stopping criteria which takes the standard deviation/ variance of the composed distribution. In the problem in hand, it is simpler since all the 40 random variables are sampled from the same kind of distribution.
* The code named << **40,100 linear - main_AKMCS_U** >> is the AKMCS model to run the high dimensional problem.
* The code named << **LowDim - AKIS** >> is the AKIS model that works on the low dimensional Cantilever beam problem adopted from OpenTurns.
* The code named << **LowDim - KPLS** >> is the KPLS model that works on the low dimensional problem.
* The code named << **LowDim - main_AKMCS_U** >> is the AKMCS model that works on the same low dimensional problem.
* The code named << **gaussian_process** >> is the code that handles the training of the GP surrogate. It is not advised to modify this code - ! DO NOT EDIT !
* The code named << **regression_models** >> is one of the dependancies of the gaussian process code - ! DO NOT EDIT !
* The code named << **correlation_models** >> is one of the dependancies of the gaussian process code - ! DO NOT EDIT !
* The code named << **Crude_MC** >> is the Crude Monte Carlo simulation to estimate any probability. It is based on the Monte Carlo estimator inbuilt in OpenTurns.
* The code named << **create_symbolic_function** >> is used to create the symbolic model for the FORM analysis. This is specific to the problem and is imported all the models at high dimension. While using the models for a new problem, this must be changed fully, in case you are working with analytical function.

# RECOMMENDATIONS BEFORE YOU START WITH THE CODES -->

- Recommended flow to get a hang of the codes:
  - start with "LowDim" codes, from AKMCS --> AKIS --> KPLS. (Use crude Monte Carlo code is required before starting this step if you don't have the Probability value already from a trusted source.
  - To check the performance in high dimension, test out codes starting with  "40,100 linear", in order preferred as : AKMCS --> AKIS --> KPLS
  - Interesting things to test once you are used to the codes: play around with the stopping criterion for AKIS and KPLS, play around with the formulation of new Importance sampling density, run the codes on a much higher dimension value and see the scope of improvement.
 
# ATTENTION USERS !! -->

- _Note that with more dimensions/ more complex limit states/ more conservative stopping criteria, the training or Pf computation can prolong and be computationally intensive, especially for AKMCS model. To paint a picture, for the high dimension problem, a limit state of Y<0 which is expected to have a probability of occurence around 1.6e-3 to 2e-3 was too intense to execute using the AKMCS model on an **Intel i5 1035G1 CPU @ 1 GHz with 8 Gb RAM** which however ran seamlessly using the AKMCS and KPLS models. Hence trying to do so on a machine with limited computation capacity will crash the system leading to a forced shut down and loss of unsaved data._
