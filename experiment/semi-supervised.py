#!/usr/bin/env python


# import section
# import os
import sys
import subprocess
# import numpy
from random import randint
# import pickle


# global variables
weka = "java -cp weka.jar "
removeFilter1 = "weka.filters.unsupervised.attribute.Remove -R "
removeFilter2 = " -i "
removeFilter3 = " -o "
addIDfilter = "weka.filters.unsupervised.attribute.AddID -i "
#	J48
J481a = "weka.classifiers.trees.J48 -C 0.25 -M 2 -p 1 -s "
J481b = "weka.classifiers.trees.J48 -C 0.25 -M 2 -v -o -s "
J482 = " -t "
J483 = " -T "
#	SMO-Poly
SMOP1a = ( "weka.classifiers.functions.SMO -C 1.0 -L 0.0010 -P 1.0E-12 " +
	"-p 1 -s " )
SMOP1b = ( "weka.classifiers.functions.SMO -C 1.0 -L 0.0010 -P 1.0E-12 " +
	"-v -o -s " )
SMOP2 = ( " -N 0 -V -1 -W 1 -K \"weka.classifiers.functions." +
	"supportVector.PolyKernel -C 250007 -E 1.0\" -t " )
SMOP3 = " -T "
#	SMO-RBF
SMOR1a = ( "weka.classifiers.functions.SMO -C 1.0 -L 0.0010 -P 1.0E-12 " +
	"-p 1 -s " )
SMOR1b = ( "weka.classifiers.functions.SMO -C 1.0 -L 0.0010 -P 1.0E-12 " +
	"-v -o -s " )
SMOR2 = ( " -N 0 -V -1 -W 1 -K \"weka.classifiers.functions.supportVector." +
	"RBFKernel -C 250007 -G 0.01\" -t " )
SMOR3 = " -T "
#	IBk
IBk1a = "weka.classifiers.lazy.IBk -p 1 -s "
IBk1b = "weka.classifiers.lazy.IBk -v -o -s "
IBk2 = ( " -K 1 -W 0 -A \"weka.core.neighboursearch.LinearNNSearch -A " +
	"\\\"weka.core.EuclideanDistance -R first-last\\\"\" -t " )
IBk2 = " -T "


# functions
################################################################################
# count number of labels to predict
#
#
def countLabels(labels) :
	counter = 0

	for i in labels :
		if "<label name=\"" in i and "\"></label>" in i :
			counter += 1

	return counter
#
#
#
################################################################################
################################################################################
# give # to all data instances to be able to distinguish them later on
#
#
def IDdata(fileName):
	clas = subprocess.Popen( weka + addIDfilter + fileName + " -o " +
		fileName[0:-5] + "_ID.arff", stdout=subprocess.PIPE, shell=True )
	(out, err) = clas.communicate()
	if err!=None:
		print "Error while appending ID:\n" + err
		exit()
#
#
#
################################################################################
################################################################################
# remove ground truth for labels
#
#
def rmAttributes(fileName, IDrange):
	clas = subprocess.Popen( weka + removeFilter1 + IDrange + removeFilter2 +
		fileName[0:-5] + "_ID.arff" + removeFilter3 + fileName[0:-5] +
		"_unlabeled.arff", stdout=subprocess.PIPE, shell=True )
	(out, err) = clas.communicate()
	if err!=None:
		print "Error while removing labels:\n" + err
		exit()
#
#
#
################################################################################
################################################################################
# count number of data instances; return header of arff file and return list of
#	instances in arff file
#
def handleInstances(dataList):
	dataStarted = False
	counter = 0
	header = []
	instances = []
	for line in dataList:
		# exclude empty lines and commented lines
		if line[0] == '%' or line[0] == "\n" or line[0] == "\r" :
			continue

		# until "@data" put everything to header
		if not dataStarted and line.find("@data") == -1 :
			header.append(line)
		elif not dataStarted and "@data" in line :
			header.append(line)
			dataStarted = True
		# put all the rest into instances
		elif dataStarted :
			counter += 1
			instances.append(line)

	return (counter, header, instances)
#
#
#
################################################################################
################################################################################
# count number of attributes present in the data set
#
#
def countAtributes( data ) :
	count = 0

	# if it's empty return error
	if not data :
		print "Error encountered. Data file empty!"
		exit()
	# count commas what corresponds to attributes
	for i in data[0] :
		if i == ',' :
			count += 1

	# one less comma than instances
	return count + 1
#
#
#
################################################################################
################################################################################
# generate a list of 'x' indexes to pick for supervised learning
#
#
def supIndex(noToExtract, noInstances) :
	if noToExtract > noInstances :
		print( "You are extracting more elements that there is instances." +
			str(noToExtract) + " out of " + str(noInstances) + "!" )
		exit()

	remove = []

	for i in range(noToExtract):
		lock = 0
		# generate random between 0 and noInstances included
		r = randint(0, noInstances - 1)
		# if already appended to remove find another one
		# ???????what if all are included???????
		# try 1.5*noInstances times if not throw error
		while r in remove :
			if lock > 1.5*noInstances :
				# distinct element not found; throw error
				print "I think that you want to extract to many elements."
				exit()
			lock += 1
			r = randint(0, noInstances - 1)
		remove.append(r)
	return remove
#
#
#
################################################################################
################################################################################
# generate 2 lists: one containing training(labeled) instances and the other
# list of instances to instances to improve a classifier
#
def createTT(removeInd, instances) :
	rm = removeInd[:]
	test = []
	training = []

	for ind, val in enumerate(instances):
		# instance with ground-truth
		if ind in rm :
			# append element to training
			training.append(val)
			# remove ind element
			rm.remove(ind)
			continue
		test.append(val)

	if not rm :
		print "All elements appended."

	return (training, test)
#
#
#
################################################################################
################################################################################
# create arff files with selected instances
#
#
def saveToarff(fileName, arffHeader, unlabeledArffHeader, Training, Test,
	unlabeledTest, removeLabels) :
	# open two files to write
	testStream = open( fileName[0:-5]+"_unlabeledTest.arff", 'w')
	trainingStream = open( fileName[0:-5]+"_labeledTraining.arff", 'w')

	# save header info to streams
	for i in arffHeader :
		trainingStream.write(i)
	if removeLabels :
		for i in unlabeledArffHeader :
			testStream.write(i)
	else :
		for i in arffHeader :
			testStream.write(i)

	# save training info to streams
	for i in Training :
		trainingStream.write(i)

	# save test info to stream
	if removeLabels :
		for i in unlabeledTest :
			testStream.write(i)
	else :
		for i in Test :
			testStream.write(i)
#
#
#
################################################################################
################################################################################
# train selected classifiers with newly created arff files and test on boosting
#	data
#
def trainClassifier(fileName) :
	# classify with J48
	clas = subprocess.Popen( weka + J481a + fileName[0:-5] + "" +
		fileName[0:-5] + "_ID.arff", stdout=subprocess.PIPE, shell=True )
	(J48, err) = clas.communicate()
	if err!=None:
		print "Error while classifying J48:\n" + err
		exit()

	# classify with IBk
	clas = subprocess.Popen( weka + addIDfilter + fileName + " -o " +
		fileName[0:-5] + "_ID.arff", stdout=subprocess.PIPE, shell=True )
	(IBk, err) = clas.communicate()
	if err!=None:
		print "Error while classifying IBk:\n" + err
		exit()

	# classify with SMO-Poly
	clas = subprocess.Popen( weka + addIDfilter + fileName + " -o " +
		fileName[0:-5] + "_ID.arff", stdout=subprocess.PIPE, shell=True )
	(SMOP, err) = clas.communicate()
	if err!=None:
		print "Error while classifying SMO-Poly:\n" + err
		exit()

	# classify with SMO-RBF
	clas = subprocess.Popen( weka + addIDfilter + fileName + " -o " +
		fileName[0:-5] + "_ID.arff", stdout=subprocess.PIPE, shell=True )
	(SMOR, err) = clas.communicate()
	if err!=None:
		print "Error while classifying SMO-RBF:\n" + err
		exit()

	return( IBk, J48, SMOP, SMOR )
#
#
#
################################################################################
################################################################################
# make sens out from weka output
#
#
def extractOutput( rawIBk, rawJ48, rawSMOP, rawSMOR ) :
	IBk, J48, SMOP, SMOR = [], [], [], []

	return ( IBk, J48, SMOP, SMOR )
#
#
#
################################################################################


# main program
#	read arguments to list
argumentList = list(sys.argv)

#	check correct number of arguments
if len(argumentList)!=3:
	print ( "There should be 2 arguments given:" + "\n" +
		"	-*- data set in 'arff' format" + "\n"
		"	-*- file containing labels in 'xml' format" )
		# + "\n" + "	-*- number of iterations" )
	exit()

#	open labels data
rawLabels = open(argumentList[2], 'r')
labels = list(rawLabels)
#	...and count them
noLabels = countLabels(labels)
#	check whether counting is correct
print( str(noLabels) + " labels have(has) been found. Is it Correct?" )
noLabelsUsr = None
while not noLabelsUsr :
	try:
		noLabelsUsr = raw_input( "To confirm type 'y' or if the number is " +
			"incorrect please give true number of labels: " )
		# if user put 'y' continue
		if noLabelsUsr == 'y' :
			continue

		noLabels = int(noLabelsUsr)

	except ValueError:
		print 'Invalid Number. For YES write \'y\' and confirm with \'return\'.'
		noLabelsUsr = None

#	put ID as a first element of data
IDdata(argumentList[1])

#	open file with IDs
rawFile = open(argumentList[1][0:-5] + "_ID.arff", 'r')
fileList = list(rawFile)

#	count the number of instances and get data list and header list
(noInstances, arffHeader, data) = handleInstances(fileList)

#	count number of attributes
noAtributes = countAtributes(data)

#	remove the ground truth for labels
IDrangeRm = ( str(noAtributes-noLabels+1) + "-" + str(noAtributes) )
rmAttributes(argumentList[1], IDrangeRm)

#	get list of data instances with removed ground truth
rawFileUnlabeled = open(argumentList[1][0:-5] + "_unlabeled.arff", 'r')
fileUnlabeled = list(rawFileUnlabeled)
(empty, unlabeledArffHeader, unlabeledData) = handleInstances( fileUnlabeled )

#	ask how many to use for supervised learning
sup = None
while not sup :
	try:
		sup = int( raw_input( "How many out of " + str(noInstances) +
			" instances do you want to use for " + "supervised learning?: " ) )
	except ValueError:
		print 'Invalid Number'

#	Give a list of indexes to use for supervised learning
supIndexes = supIndex( sup, noInstances )

#	extract supIndexes and write to arff file
(Training, Test) = createTT(supIndexes, data)
(empty, unlabeledTest) = createTT(supIndexes, unlabeledData)

#	convert lists to arff files and write set_training.arff and set_test.arff
#	rmLabels decides whether to use labeled data or unlabeled as test set
rmLabels = True
saveToarff(argumentList[1], arffHeader, unlabeledArffHeader, Training, Test,
	unlabeledTest, rmLabels)

# 1
#	train classifiers on initial train set with mentioned schemes and test
#	(predict) on rest and return raw outputs
( rawIBk, rawJ48, rawSMOP, rawSMOR ) = trainClassifier(argumentList[1])

#	make sens of outputs
( IBk, J48, SMOP, SMOR ) = extractOutput( rawIBk, rawJ48, rawSMOP, rawSMOR )

#	ask for numbers of samples to to add and boost classifier
boostNums = None
boostNum = 0
while not boostNums :
	try:
		boostNums = raw_input( "How many out of " + str(noInstances-sup) +
			" instances do you want to use to boost classifier?" + "\n" +
			"If you want to stop boosting operation and check accuracy of " +
			"classifier put letter [s]." )

		# if user put 's' stop boosting
		if boostNums == 's' :
			# go to 'cross-validating' classifier
			continue

		boostNum = int( boostNums )

		# number must be greater than 0
		if boostNum <= 0 :
			print "Number must be greater than 0!"
			boostNums = None

	except ValueError:
		print 'Invalid Number. I you want to stop put [s].'
		boostNums = None

#	rebuilt classifier with # of samples defined by user choosing all instances
#	where majority of classifiers agrees
#boils down to rebuilding datasets and going back to stage #1


#
##	Sunday
#
#	check accuracy of created classifier

#	then perform n-times with each of classifiers with cross validation
#	to compare accuracy of results
