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
classAsignA = "weka.classifiers.trees.J48 -C 0.25 -M 2 -p 1 -s "
classAsignB = " -t "
classifierA = "weka.classifiers.trees.J48 -C 0.25 -M 2 -v -o -s "
classifierB = " -t "
#	SMO-Poly
classAsignA = ( "weka.classifiers.functions.SMO -C 1.0 -L 0.0010 -P 1.0E-12 " +
	"-p 1 -s " )
classAsignB = ( " -N 0 -V -1 -W 1 -K \"weka.classifiers.functions." +
	"supportVector.PolyKernel -C 250007 -E 1.0\" -t " )
classifierA = ( "weka.classifiers.functions.SMO -C 1.0 -L 0.0010 -P 1.0E-12 " +
	"-v -o -s " )
classifierB = ( " -N 0 -V -1 -W 1 -K \"weka.classifiers.functions." +
	"supportVector.PolyKernel -C 250007 -E 1.0\" -t " )
#	SMO-RBF
classAsignA = ( "weka.classifiers.functions.SMO -C 1.0 -L 0.0010 -P 1.0E-12 " +
	"-p 1 -s " )
classAsignB = ( " -N 0 -V -1 -W 1 -K \"weka.classifiers.functions." +
	"supportVector.PolyKernel -C 250007 -E 1.0\" -t " )
classifierA = ( "weka.classifiers.functions.SMO -C 1.0 -L 0.0010 -P 1.0E-12 " +
	"-v -o -s " )
classifierB = ( " -N 0 -V -1 -W 1 -K \"weka.classifiers.functions." +
	"supportVector.PolyKernel -C 250007 -E 1.0\" -t " )
#	IBk
classAsignA = "weka.classifiers.lazy.IBk -p 1 -s "
classAsignB = ( " -K 1 -W 0 -A \"weka.core.neighboursearch.LinearNNSearch -A " +
	"\\\"weka.core.EuclideanDistance -R first-last\\\"\" -t " )
classifierA = "weka.classifiers.lazy.IBk -v -o -s "
classifierB = ( " -K 1 -W 0 -A \"weka.core.neighboursearch.LinearNNSearch -A " +
	"\\\"weka.core.EuclideanDistance -R first-last\\\"\" -t " )


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
def labelData(fileName):
	clas = subprocess.Popen( weka + addIDfilter + fileName + " -o " +
		fileName[0:-5] + "_ID.arff",	stdout=subprocess.PIPE, shell=True )
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
def unlabelData(fileName, IDrange):
	clas = subprocess.Popen( weka + removeFilter1 + IDrange + removeFilter2 +
		fileName + removeFilter3 + fileName[0:-5] + "_unlabeled.arff",
		stdout=subprocess.PIPE, shell=True )
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

	return count
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
# unlabeled instances to improve a classifier
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
def saveToarff(arffHeader, Training, Test, removeLabels) :
	# open two files to write
	testStream = open( argumentList[1][0:-5]+"_unlabeledTest.arff", 'w')
	trainingStream = open( argumentList[1][0:-5]+"_labeledTraining.arff", 'w')

	# save header info to streams
	for i in arffHeader :
		testStream.write(i)
		trainingStream.write(i)

	# save training info to streams
	for i in Training :
		trainingStream.write(i)

	# save test info to stream
	if removeLabels :
		for i in Test :
			j = i[0:-5]
			testStream.write(j)
	else :
		for i in Test :
			testStream.write(i)
#
#
#
################################################################################
################################################################################
# train selected classifiers with newly created arff files
#
#
def trainClassifier() :
	arffHeader = []
	Training = []
	Test = []

	return (arffHeader, Training, Test)
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
labelData(argumentList[1])

#	open file with IDs
rawFile = open(argumentList[1][0:-5] + "_ID.arff", 'r')
fileList = list(rawFile)

#	count the number of instances and get data list and header list
(noInstances, arffHeader, data) = handleInstances(fileList)

#	count number of attributes
noAtributes = countAtributes(data)

#	remove the ground truth for labels
IDrangeRm = ( str(noAtributes-noLabels+1) + "-" + str(noAtributes) )
unlabelData(argumentList[1], IDrangeRm)

#	ask how many to use for supervised learning
sup = None
while not sup :
	try:
		sup = int( raw_input( "How many out of " + str(noInstances) + " instances do you want to use for " +
			"supervised learning?: " ) )
	except ValueError:
		print 'Invalid Number'

#	Give a list of indexes to use for supervised learning
supIndexes = supIndex( sup, noInstances )

#	extract supIndexes and write to arff file
#	write set_training.arff and set_test.arff
(Training, Test) = createTT(supIndexes, data)

#	convert lists to arff files
#	rmLabels decides whether to remove labels from test set
rmLabels = True
saveToarff(arffHeader, Training, Test, rmLabels)

#	train classifiers on initial train set with mentioned schemes and write them
#	to files
trainClassifier()

#	ask for numbers of samples to to add and boost classifier
#	classify randomly chosen samles choosing all that agrees in majority
#	rebuilt classifier

#	then perform n-times with each with each of classifiers to compare results
