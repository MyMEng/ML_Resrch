# **Introduction to Machine Learning- Research Project**
# **Implementation of simple SEM-SUPERVISED learning**

---

## Introductions
This repository contains `python` script which performs simple *semi-supervised* learning using `WEKA` ML software. For comparison purposes the script also performs supervised learning to provide statistics at the end.

## To run (UNIX based systems):
If you have GIT installed just open terminal and clone repository:

    cd ~
    git clone https://github.com/So-Cool/ML_Resrch

Then change current directory to ML_Resrch:

    cd ML_Resrch

Run program with command:

    ./semi-supervised.py [pair1] [pair2]

for instance:

 - **[par1]** = `emotions.arff`

 - **[par2]** = `emotions.xml` *(optional)*

i.e.:

    	./semi-supervised.py emotions.arff emotions.xml

Where:

1. **\[par1\]** is a file containing all labeled training data in 'arff' format.

2. **\[par2\]** is an *optional* argument that should be given while data file is a multi label set. It specifies the 'xml' file with labels list. If it is not specified the program assumes that there is only one label but asks for confirmation on the run. If this argument is not specified the user can still manually enter the correct number of target labels. In case of multi label prediction the data is split into *n='number of labels'* separate data sets- one target label per set- and then the experiment is performed on all of them separately accumulating the scores and predictions.

The format of 'xml' file is described as follows:

		<?xml version="1.0" encoding="utf-8"?>
		<labels xmlns="Link to the website containing data set">
	    <label name="Target Label 1"></label>
	    <label name="Target Label 2"></label>
	    <label name="Target Label 3"></label>
	    </labels>

## Program input
1. The input is [`y`] to confirm or `integer` specifying the correct number of target labels.

		x labels have(has) been found. Is it Correct?
		To confirm type 'y' or if the number is incorrect please give true number of labels:

2. The input is `integer` specyfying number of instances that you want to use for initial learning of classifier. The instances are selected at random and contain ground truth.

	`How many out of x instances do you want to use for supervised learning?:`

3. The input is either [`s`] if you don't want to boost the classifier or `integer` representing the number of instances to use for boosting procedure.

		All elements appended.
		All elements appended.
		There are: 44 instances that agree in all 4 classifiers.
		There are: 44 instances that agree in 3 out of 4 classifiers.
		There are: 32 instances that agree in 2 out of 4 classifiers.
		There are: 0 instances that agree in non of classifiers.
		Priority in choosing instances for boost operation is given to ones that agrees in most of classifiers.
		How many out of 120 instances do you want to use to boost classifier?
		If you want to stop boosting operation and check accuracy of classifier put letter [s]: 

4. `integer` representing number of trials to perform on each of classifier with whole training set or cross-validation rounds if no test set provided. 

	`How many times do you want to perform tests with c-v of supervised learning?: `

5. `name` of test data set in 'arff' format. If [`none`] typed the script will use whole training set to test the performance in semi-supervised case and cross-validation will be performed in supervised case.

	`Please give name of external test file. If you don't have one c-v will be performed on whole data set; in this case  type [none]: `

6. Finally statistics are displayed.

		Confusion matrix over 2 repetitions for SUPERVISED learning:

			509	107

			150	354

		Confusion matrix for SEMI-SUPERVISED learning (each value multiplied  by number of repetitions):

			288	328

			88	416

		863 instances out of 1120 instances were predicted correctly in SUPERVISED learning.
		704 instances out of 1120 instances were predicted correctly in SEMI-SUPERVISED learning.


## The Experiment
To this end, the script uses 4 implemented in `WEKA` classifiers: `IBk`, `J48`, `SMO` with `Polynomial Kernel` and `SMO` with `RBF Kernel`. It the trains all of them on randomly chosen sample of data instances and predicts class and all the other instances of given *train* file. Then it shows statistics of classification and allows user to define number of instances to enrich train set with ground truth for them assigned as predicted class(majority class of 4 predictions for each classifier). This classifier reinforcing process is repeated until there is no instances form train set left or user aborts training procedure. Finally the performance of classifier is evaluated on external test set or if there is no such on whole data set supplied for training and statistics are displayed.

Detailed description of the results is contained in *The Experiment* section of the report which is also included in this repository as: `report/report.pdf`.