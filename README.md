### Methods

Test text

| Name | Discretization required? | External Discretization? | Type of discretization | Target Values | Paper Name | Paper year | No. Citations | Quality Measures | Comments |
| ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
| PRIM 		| No 	| No | - | Nominal/Numeric | Bump-Hunting in high-dimensional data | 1999 | - | Density | - |
| CN2  		| Yes 	| Yes | - | Nominal | The CN2 induction algorithm | 1989 | 3160 | - | Available in WEKA softwarte, for sake of completeness |
| CN2-SD 		| Yes 	| Yes | - | Nominal | Subgroup discovery with CN2-SD | 2004 | 426 | WRAcc [2] | - |
| Apriori-SD 	| Yes 	| Yes | - | Nominal | APRIORI-SD: Adapting association rule learning to | 2008 | 160 | WRAcc [2] | - |
| DSSD 		| Yes 	| No | equi-width (?) | Nominal/Numeric  | Diverse subgroup set discovery | 2012 | 93 | [2] | Discretization only within subgroups for refinement |
| MergeSD 	| No 	| No | - | Nominal | On subgroup discovery in numerical domains | 2009 | 94 | [1] | - |
| LENS 		| No 	| No | - | Nominal | Informative summarization of numeric data | 2019 | 1 | Information gain (difference of kullback distance) | - |
| SD-Map 		| - 	| - | - | - | - | - | - | - | - |Yes
| DP-Subgroup	| - 	| - | - | - | - | - | - | - | Predecessor of DSSD? |
| BSD 			| - 	| - | - | - | - | - | - | - | - | 

[1] g * (box_mean - mean) With g generality measure; Grosskreutz, Henrik, and Stefan Rüping. “On Subgroup Discovery in Numerical Domains.”

[2] ![Leeuwen, Matthijs van, and Arno Knobbe. “Diverse Subgroup Set Discovery.”](./readme_resources/qualt_table.png)



### Ideas

All ideas as a heap. You're welcome to give it better structure. Please always add whatever you find useful!
Remember, that these ideas will appear constantly, but to sucessfully finish the thesis your schedule should be of first priority. That is, always start with some MVP (https://en.wikipedia.org/wiki/Minimum_viable_product) and then refine it, if you have some time. The MVP, could be, e.g. *five* generation techniques Yes *five* complex models + *15* datasets + framework for experiments, which will allow to add new generation techniques, datasets and complex models easily.

| Direction| Idea | Description | Status |
| ------ | ------ | ------ | ------ |
| Generation | Shift points | Baseline. Either shift everything in one direction or add a noice, where standard deviation is a parameter. The label is *retained*.| generators.PointShift: PointShift() |
| Generation | Add noise | Baseline. The same as above, but the new label is created with *complex* metamodel. | generators.PointShift: PointShift() |
| Generation | MUNGE | Bucilua, C. et al. 2006. Model compression. Proceedings of the 12th ACM SIGKDD international conference on Knowledge discovery and data mining - KDD ’06 (2006), 535 Implementations: https://github.com/lapis-zero09/MUNGE, https://github.com/Prometheus77/munge | generators.Munge: munge_annoy() |
| Generation | Gaussian(?) mixture | NBE(?), ME algorithm | NaiveBayesEstimation.py implements this; addtionally orignal C implementation is wrapped in NBEWrapper package -> way faster but sometimes errors appear (in C code)  |
| Generation | Uniform | Another baseline | random_samples_uniform() |
| Generation | Normal | Another baseline | random_samples_normal()|
| Generation | KDE | Kernel density estimation (https://towardsdatascience.com/modality-tests-and-kernel-density-estimations-3f349bb9e595) | Available in sklearn.neighbours VA: *check the link I added*|
| Generation | GAN | https://towardsdatascience.com/understanding-generative-adversarial-networks-gans-cd6e4651a29 How to train: https://github.com/soumith/ganhacks| |
| Generation | One-class SVM | Not sure if it can be used for generation... |  |
| Generation | Like in Trepan | Craven, M.W. and Shavlik, J.W. 1996. Extracting tree-structured representations of trained neural networks. Advances in Neural Information Processing Systems (1996), 24–30. | sample_trepan() |
| Metamodels | Kriging | A.k.a. gaussian processes |   |
| Metamodels | Several from the study | Malley, J.D. et al. 2012. Probability Machines: Consistent probability estimation using nonparametric learning machines. Methods of Information in Medicine. 51, 1 (2012), 74–81. |   |
| Metamodels | SVM | with care: see here https://stats.stackexchange.com/questions/302567/why-is-it-wrong-to-interpret-svm-as-classification-probabilities or here https://www.researchgate.net/post/Can_we_assign_probability_to_SVM_results_instead_of_a_binary_output2 | |
| PRIM alternative | Tree-GA | Hirose, H. and Koga, G. 2013. A comparative study in the bump hunting between the Tree-GA and the PRIM. Studies in Computational Intelligence (2013), 13–25 | Bad paper, doesn't worth to try |
| PRIM alternative | 13 different, with implementations | Loh, Wei‐Yin, Luxi Cao, and Peigen Zhou. "Subgroup identification for precision medicine: A comparative review of 13 methods." Wiley Interdisciplinary Reviews: Data Mining and Knowledge Discovery: e1326.| Interesting, but all implementations are in R. Evaluate overhead of doing R function calls. otherwise I should maybe choose 1-2. |
| PRIM alternative | Rule learning? | http://www.keel.es/ -> included algorithms -> classification -> Rule learning. Classification rules are more suitable for classification, rather than for bump-hunting. So maybe these are irrelevant | |
| PRIM alternative |  | http://www.keel.es/ -> included algorithms -> classification -> Rule learning. Classification rules are more suitable for classification, rather than for bump-hunting. So maybe these are irrelevant | |
| PRIM Alternative | Subgroup discovery with pysubgroup https://bitbucket.org/florian_lemmerich/pysubgroup/src/master/ ||
| Datasets | Dropbox | Deleted, but can revive... |   |
| Datasets | openml benchmark | https://docs.openml.org/benchmark/ |  |
| Datasets | UCI | https://archive.ics.uci.edu/ml/datasets.php |  |
| Datasets | From AML paper | https://www.automl.org/wp-content/uploads/2019/05/AutoML_Book_Chapter8.pdf |  |
| Datasets | Some references | http://pages.stat.wisc.edu/%7Eloh/apps.html | |
| Related work? | See paper | https://arxiv.org/abs/1906.06852 This might be *very* similar to what we are doing. | |
| Datasets | Penn Machine Learning Benchmarks | https://github.com/EpistasisLab/penn-ml-benchmarks | |


#### Plan

| Big part| Week| ToDo | Done | Comment |
| ------- | ------ | ------ | ------ | ------ |
| PRIM + generators| Before 15.07 | - | ... |  |
| Generators| 15.07-21.07 |  | ... |  |
| Generators Finish | 22.07-28.07 | Refactor & Finalize methods + Docs in LaTex (&code doc?)| NBE fixed, and original implementation added, Other methods are fine; Some docs added |  |
| Complex Models | 29.07-04.08 | SVM+Logreg, Krigin, find a 5th? | b-NN++, regRF+, SVR+, Gauss-Processes+, SGD(?)+ , ANNs+ | Stochastic gradient descent may be interesting, SVR implementation needs deeper review |
| - | 05.08-11.08 | - | ... |  |
| Evaluation of Experiment Setup + Begin PRIM Alternatives | 12.08-18.08| Defnie coarse structure of experiments, Choose a framework  |  |  |
| PRIM Alternatives | 19.08-25.08 |  | ... |  |
| Datasets? -> Experiments | 26.08-01.09 |  | Depends on time datasets take |  |
| - | 02.09-08.09 |  | ... |  |
| Bugfixing | 09.09-15.09 |  | ... |  |
| Experiments | 16.09 - 23.09 |  | ... |  |
| Elaboration | 24.09 - 01.10 |  | ... |  |

 \+ = available, ++=implemented

----
Notes from 4.10.19
1) Get the datasets from existing papers using PRIM
  a) write an email -> show it to me -> send it
  b) explain me the access problem with "hospital data"
  
2) PRIM and CART
  a) decide whether you want to compare with CART
  b) send me the paper comparing these two, which you have found
3) PRIM stopping - make sure, your stopping criterion is good
4) PRIM interpretability - use the number of restricted dimensions. Hint: for interpretability, to make it fair, use the stopping criterion, which depends on the true (not artificial) data only
5) Presentation - check the requirements (regarding dates) and take the slot
6) "I" or "We" - I will check this
7) Experiments - include 'dummy' generator and 'dummy' metamodel, which do nothing. This is in order to split the effect of labeling with probabilities and more points.
8) Number of points: try 3-4 different numbers (e.g., 300, 600, 1200, 2400). Try to obtain the 'learning curve' for the dataset, which:
  a) is big
  b) shows good performance of our method
9) Citations: earlier fork or reviews/surveys - up to you, but be sure that the stuff you intend to refer to is indeed in respective work.
Did I forget something?
Additionally
10) do a couple experiments with the number of generated points, to make sure, that 10 000 (as you currently use) is enough

----
Notes from 26.11.19
Some results of the today's meeting:
send requests for the data
try to figure out if everything works well
a. why on preliminary experiments the improvement was that more notable than now?
b. do your results look similar to my preliminary experiments with no points creation, just relabeling?
c. why the AUC variance of the dummy method is so small?
d. why the relative AUC and density variances are not similar for the same methods and datasets?
  3. to obtain datasets with intermediate number of dimensions - apply feature selection to the large ones. But remember, that the original large dataset and the one resulted from feature selection cannot be treated as to completely different (independent) - so try to increase variety at the same time
  4. In your work create one plot comparing 1-2 selected combinations generator + metamodelwith dummy for all datasets you used - so that one can see the effect of the approach
  5. your baselines are a. dummy - doing nothing b. dummy generator + metamodel and c. uniform generator + metamodel
(could you push this notes to GitLab as you did with ones from some previous discussion?)