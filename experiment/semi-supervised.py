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
# give # to all data instances to be able to distinguish them later on
#
#
def labelData(fileName):
	clas = subprocess.Popen(
		"java -cp weka.jar weka.filters.unsupervised.attribute.AddID -i " +
		fileName + " -o " + fileName[0:-5] + "_ID.arff",
		stdout=subprocess.PIPE, shell=True )
	(out, err) = clas.communicate()
	if err!=None:
		print "Error while appending ID:\n" + err
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
# generate a list of 'x' indexes to pick for supervised learning
#
#
def supIndex(noToExtract, noInstances) :
	if noToExtract > noInstances :
		print "You are extracting more elements that there is instances."
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


# main program
#	read arguments to list
argumentList = list(sys.argv)

#	check correct number of arguments
if len(argumentList)!=2:
	print ( "There should be 1 argument given:" + "\n" +
		"	-*- filename in 'arff' format" )
		# + "\n" + "	-*- number of iterations" )
	exit()

#	put ID as a first element of data
labelData(argumentList[1])

#	open file with IDs
rawFile = open(argumentList[1][0:-5] + "_ID.arff", 'r')
fileList = list(rawFile)

#	count the number
(no, arffHeader, data) = handleInstances(fileList)

#	ask how many to use for supervised learning
sup = None
while not sup :
	try:
		sup = int( raw_input( "How many out of " + str(no) + " instances do you want to use for " +
			"supervised learning?: " ) )
	except ValueError:
		print 'Invalid Number'

#	Give a list of indexes to use for supervised learning
supIndexes = supIndex( sup, no )

#	extract supIndexes and write to arff file
#	train classifiers with mentioned schemes and write them to files

#	ask for numbers of samples to to add and boost classifier
#	classify randomly chosen samles choosing all that agrees in majority
#	rebuilt classifier

#	then perform n-times with each with each of classifiers to compare results
