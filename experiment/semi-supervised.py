#!/usr/bin/env python

import sys
import subprocess
from random import randint
from random import choice
from shutil import copy
# import os
# import numpy as np
# import section
# import pickle


# global variables
weka = "java -cp weka.jar "
removeFilter1 = "weka.filters.unsupervised.attribute.Remove -R "
removeFilter2 = " -i "
removeFilter3 = " -o "
addIDfilter = "weka.filters.unsupervised.attribute.AddID -i "

meta1 = "weka.classifiers.meta.FilteredClassifier -F \'" # remove filter + in
meta2 =	" \' -W " # + classifier w/o options + -t train -T test -p 1
meta3 = " -- " # other classifer options -C 0.25 -M 2

trainingSwitch = " -t "
testSwitch = " -T "

#	J48
J481a = "weka.classifiers.trees.J48 -v -o -s "
J481b = " -C 0.25 -M 2 "
J482a = "weka.classifiers.trees.J48 -p 1 "
J482b = J481b
#	SMO-Poly
SMOP1a = "weka.classifiers.functions.SMO -v -o -s "
SMOP1b = ( " -C 1.0 -L 0.0010 -P 1.0E-12 -N 0 -V -1 -W 1 -K \"weka." +
	"classifiers.functions.supportVector.PolyKernel -C 250007 -E 1.0\"" )
SMOP2a = "weka.classifiers.functions.SMO -p 1 "
SMOP2b = SMOP1b
#	SMO-RBF
SMOR1a = "weka.classifiers.functions.SMO -v -o -s "
SMOR1b = ( " -C 1.0 -L 0.0010 -P 1.0E-12 -N 0 -V -1 -W 1 -K \"weka." +
	"classifiers.functions.supportVector.RBFKernel -C 250007 -G 0.01\"" )
SMOR2a = "weka.classifiers.functions.SMO -p 1 "
SMOR2b = SMOR1b
#	IBk
IBk1a = "weka.classifiers.lazy.IBk -v -o -s "
IBk1b = ( " -K 1 -W 0 -A \"weka.core.neighboursearch.LinearNNSearch -A " +
	"\\\"weka.core.EuclideanDistance -R first-last\\\"\"" )
IBk2a = "weka.classifiers.lazy.IBk -p 1 "
IBk2b = IBk1b


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
		# ???????what if all are included??????? - ERROR
		# try 1.5*noInstances times if not throw error
		while r in remove :
			if lock > 1.5*noInstances :
				# distinct element not found; throw error
				print "I think that you want to extract to many elements."
				exit()
			lock += 1
			r = randint(0, noInstances - 1)
		remove.append(r)

		# zip remove with bunch of 0 - zero means the class was given in super
		zeros = len(remove) * [0]
		removeZip = zip(remove, zeros)

	return removeZip
#
#
#
################################################################################
################################################################################
# extract values of attribute to predict
#
#
def extractTargets( arffHeader ) :
	targetClasses = [ "ORIGINAL" ]
	# find last attribute
	for line in reversed(arffHeader) :
		if "@attribute" in line :
			# get last element of line which is classes, without brackets,
			#	and split on commas
			for name in (line.split()[-1][1:-1]).split(',') :
				targetClasses.append(name)
			break

	return targetClasses
#
#
#
################################################################################
################################################################################
# generate 2 lists: one containing training(labeled) instances and the other
# list of instances to instances to improve a classifier
#
def createTT(removeInd, instances, targetClasses) :
	rm = removeInd[:]
	test = []
	training = []

	for ind, val in enumerate(instances):
		# instance with ground-truth
		i = [j for j, x in enumerate(rm) if x[0] == ind]
		if i :
		# if ind in rm :
			# edit value bu putting predicted class at the end
			# if 0 "ORIGINAL" encountered don't alter class with ground truth
			if rm[i[0]][1] != 0 :
				lineList = ("".join(val.split())).split(',')
				lineList[-1] = targetClasses[rm[i[0]][1]]
				value = ','.join(lineList)
				value += "\n"
			else :
				value = val
			# append element to training
			training.append(value)
			# remove ind element
			rm.remove(rm[i[0]])
			continue
		# else append to test
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
	unlabeledTest, removeLabels, noSuper) :
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

	testStream.close()
	trainingStream.close()

	# if it's set containing only of randomly chosen instances(first iteration)
	#	make a copy for supervised learning on small set statistics
	if noSuper == len(Training) :
		# shutil.copy(src, dst)
		print( "cp " + fileName[0:-5] + "_labeledTraining.arff " +
			fileName[0:-5] + "_initial.arff" )
		err = copy( fileName[0:-5] + "_labeledTraining.arff", 
			fileName[0:-5] + "_initial.arff" )
		if err :
			print "Could not make a copy of original file:\n" + err
			exit()
#
#
#
################################################################################
################################################################################
# train selected classifiers with newly created arff files and test on boosting
#	data
#
def trainClassifier(fileName) :
	# take a random seed
	# r = 0

	# classify with J48
	# r = randint(1, 1000000)
	clas = subprocess.Popen( weka + meta1 + removeFilter1 + str(1) + meta2 +
		J482a + trainingSwitch + fileName[0:-5] + "_labeledTraining.arff" + 
		testSwitch + fileName[0:-5] + "_unlabeledTest.arff" + meta3 + J482b,
		stdout=subprocess.PIPE, shell=True )
	(J48, err) = clas.communicate()
	if err!=None:
		print "Error while classifying J48:\n" + err
		exit()

	# classify with IBk
	# r = randint(1, 1000000)
	clas = subprocess.Popen( weka + meta1 + removeFilter1 + str(1) + meta2 +
		IBk2a + trainingSwitch + fileName[0:-5] + "_labeledTraining.arff" + 
		testSwitch + fileName[0:-5] + "_unlabeledTest.arff" + meta3 + IBk2b,
		stdout=subprocess.PIPE, shell=True )
	(IBk, err) = clas.communicate()
	if err!=None:
		print "Error while classifying IBk:\n" + err
		exit()

	# classify with SMO-Poly
	# r = randint(1, 1000000)
	clas = subprocess.Popen( weka + meta1 + removeFilter1 + str(1) + meta2 +
		SMOP2a + trainingSwitch + fileName[0:-5] + "_labeledTraining.arff" + 
		testSwitch + fileName[0:-5] + "_unlabeledTest.arff" + meta3 + SMOP2b,
		stdout=subprocess.PIPE, shell=True )
	(SMOP, err) = clas.communicate()
	if err!=None:
		print "Error while classifying SMO-Poly:\n" + err
		exit()

	# classify with SMO-RBF
	# r = randint(1, 1000000)
	clas = subprocess.Popen( weka + meta1 + removeFilter1 + str(1) + meta2 +
		SMOR2a + trainingSwitch + fileName[0:-5] + "_labeledTraining.arff" + 
		testSwitch + fileName[0:-5] + "_unlabeledTest.arff" + meta3 + SMOR2b,
		stdout=subprocess.PIPE, shell=True )
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
	rawData = [rawIBk, rawJ48, rawSMOP, rawSMOR]
	data = [ [], [], [], [] ] # IBk, J48, SMOP, SMOR

	# extract _ID_ and _prediction-ID_ from raw data
	#	find (*) brackets and extract id from inside
	#	 frist found brackets contain 'ID' inside so ignore it
	for ind, val in enumerate(rawData) :
		ID = []
		prediction = []
		# find all brackets
		tempIDa = [j for j, y in enumerate(val) if y == '(']
		tempIDb = [j for j, y in enumerate(val) if y == ')']
		# if number of brackets do not agree throw exception
		if len(tempIDa) != len(tempIDb) :
			print "Number of '(' and ')' does not match. Error encountered!"
			exit()
		# extract content of brackets
		for i in range(1, len(tempIDa)) :
			 ID.append( int( val[tempIDa[i]+1:tempIDb[i]] ) )

		# find all the ':'
		tempPre = [j for j, y in enumerate(val) if y == ':']
		# if number of ':'/2 is not equal to number of ')' throw error
		if len(tempPre)/2 != len(tempIDa)-1 :
			print( "Number of '(' and ')' does not match with number of ':'. " +
				"Error encountered!" )
			exit()
		# extract predictions
		# WARNING - restriction up to 9 values of predicted label - ERROR
		for i in range(1, len(tempPre), 2) :
			prediction.append( int( val[tempPre[i]-1:tempPre[i]] ) )

		# make a tuple (ID, prediction) and put into appropriate list
		# check whether theres exact amount of elements in both lists to zip
		if len(ID) != len(prediction) :
			print( "Number of IDs does not match with number of predictions. " +
				"Error encountered!" )
			exit()
		# zip all results into pairs (ID, prediction) and sort according to ID
		data[ind] = zip( ID, prediction )[:]
		# sort data according to ID
		data[ind].sort(key=lambda tup: tup[0])

	return ( data[0], data[1], data[2], data[3] )
#
#
#
################################################################################
################################################################################
# pair matching predictions
#	each input element is a list of tuples (ID, prediction)
#
def matchPredictions( IBk, J48, SMOP, SMOR ) :
	# 0: all the same predictions | 1: 3/4 | 2: 2/4 | 3: none
	predictionQnt, predictionInd = [0, 0, 0, 0], [[], [], [], []]
	predictionClass = [[], [], [], []]

	# length of all inputs is the same
	# all are sorted so should have the same IDs
	for i in range(len(IBk)) :
		if IBk[i][0] == J48[i][0] == SMOP[i][0] == SMOR[i][0] :
			# if all 4 predictions are the same
			if IBk[i][1] == J48[i][1] == SMOP[i][1] == SMOR[i][1] :
				predictionQnt[0] += 1
				predictionInd[0].append( IBk[i][0] )
				predictionClass[0].append( IBk[i][1] )

			# if 3 out of 4 predictions are the same
			elif IBk[i][1] == J48[i][1] == SMOP[i][1] :
				predictionQnt[1] += 1
				predictionInd[1].append( IBk[i][0] )
				predictionClass[1].append( IBk[i][1] )
			elif IBk[i][1] == J48[i][1] == SMOR[i][1] :
				predictionQnt[1] += 1
				predictionInd[1].append( IBk[i][0] )
				predictionClass[1].append( IBk[i][1] )
			elif IBk[i][1] == SMOP[i][1] == SMOR[i][1] :
				predictionQnt[1] += 1
				predictionInd[1].append( IBk[i][0] )
				predictionClass[1].append( IBk[i][1] )
			elif J48[i][1] == SMOP[i][1] == SMOR[i][1] :
				predictionQnt[1] += 1
				predictionInd[1].append( J48[i][0] )
				predictionClass[1].append( J48[i][1] )

			# if 2 out of 4 predictions are the same
			# also check whether other 2 are the same and if so toss a coin
			elif IBk[i][1] == J48[i][1] :
				if SMOP[i][1] == SMOR[i][1] :
					predictionQnt[2] += 1
					predictionInd[2].append( IBk[i][0] )
					# toss coin
					if randint(0,1) :
						predictionClass[2].append( IBk[i][1] )
					else :
						predictionClass[2].append( SMOP[i][1] )

				else :
					predictionQnt[2] += 1
					predictionInd[2].append( IBk[i][0] )
					predictionClass[2].append( IBk[i][1] )
			elif IBk[i][1] == SMOP[i][1] :
				if J48[i][1] == SMOR[i][1] :
					predictionQnt[2] += 1
					predictionInd[2].append( IBk[i][0] )
					# toss coin
					if randint(0,1) :
						predictionClass[2].append( IBk[i][1] )
					else :
						predictionClass[2].append( J48[i][1] )
				else :
					predictionQnt[2] += 1
					predictionInd[2].append( IBk[i][0] )
					predictionClass[2].append( IBk[i][1] )
			elif IBk[i][1] == SMOR[i][1] :
				if J48[i][1] == SMOP[i][1] :
					predictionQnt[2] += 1
					predictionInd[2].append( IBk[i][0] )
					# toss coin
					if randint(0,1) :
						predictionClass[2].append( IBk[i][1] )
					else :
						predictionClass[2].append( J48[i][1] )
				else :
					predictionQnt[2] += 1
					predictionInd[2].append( IBk[i][0] )
					predictionClass[2].append( IBk[i][1] )
			elif J48[i][1] == SMOP[i][1] :
				if IBk[i][1] == SMOR[i][1] :
					predictionQnt[2] += 1
					predictionInd[2].append( IBk[i][0] )
					# toss coin
					if randint(0,1) :
						predictionClass[2].append( J48[i][1] )
					else :
						predictionClass[2].append( IBk[i][1] )
				else :
					predictionQnt[2] += 1
					predictionInd[2].append( IBk[i][0] )
					predictionClass[2].append( J48[i][1] )
			elif J48[i][1] == SMOR[i][1] :
				if IBk[i][1] == SMOP[i][1] :
					predictionQnt[2] += 1
					predictionInd[2].append( IBk[i][0] )
					# toss coin
					if randint(0,1) :
						predictionClass[2].append( J48[i][1] )
					else :
						predictionClass[2].append( IBk[i][1] )
				else :
					predictionQnt[2] += 1
					predictionInd[2].append( IBk[i][0] )
					predictionClass[2].append( J48[i][1] )
			elif SMOP[i][1] == SMOR[i][1] :
				if IBk[i][1] == J48[i][1] :
					predictionQnt[2] += 1
					predictionInd[2].append( IBk[i][0] )
					# toss coin
					if randint(0,1) :
						predictionClass[2].append( SMOP[i][1] )
					else :
						predictionClass[2].append( IBk[i][1] )
				else :
					predictionQnt[2] += 1
					predictionInd[2].append( IBk[i][0] )
					predictionClass[2].append( SMOP[i][1] )
			else :
				predictionQnt[3] += 1
				predictionInd[3].append( IBk[i][0] )
				# as all predictions are different randomly choose one class
				tmpls = [ IBk[i][1], J48[i][1], SMOP[i][1], SMOR[i][1] ]
				predictionClass[3].append( choice(tmpls) )

		else :
			print "Sorted indexes do not match. Unknown error!"
			exit()

	return ( predictionQnt, predictionInd, predictionClass )
#
#
#
################################################################################
################################################################################
# rebuild data sets with given details
#
#
def rebuildSets( boostNum, predictionInd, predictionClass, supIndexes ) :
	bnt = boostNum
	runOutOfIndexes = False
	tempInd = []
	tempClass = []

	# start emptying the first list
	for i in range( len( predictionInd ) ) :
		for j in range( len( predictionInd[i] ) ) :
			if bnt <= 0 :
				runOutOfIndexes = True
				break
			# convert ID to index (subtract 1). Later starts from 0 ID from 1
			tempInd.append( predictionInd[i][j] - 1 )
			tempClass.append( predictionClass[i][j] )
			bnt -= 1
		if runOutOfIndexes :
			break

	# zip ind with class
	temp = zip( tempInd, tempClass )
	# merge lists
	supIndexes += temp
	# sort indexes
	supIndexes.sort(key=lambda tup: tup[0])

	# 0 in supClasses means original class
	return supIndexes
#
#
#
################################################################################
################################################################################
# combine results into one common confusion matrix
#
#
def extractConfMx( tempConf, outConf ) :
	# check whether it's first iteration
	if len(outConf) == 0 :
		empty = True
	else :
		empty = False

	# for all 4 classifiers
	for i in tempConf :
		tempIndex = [x for x, y in enumerate(i) if y == '|']
		# for all found '|' go revers until '\n' found
		for row, j in enumerate(tempIndex) :
			# check whether it's empty if so add each time new row
			if empty :
				outConf.append([])

			for k in reversed(range(j)) :
				if i[k]=="\n" :
					temp = i[k+1 : j].split()
					for col, l in enumerate(temp) :
						# check whether it's first iteration if so insert number
						if empty :
							outConf[row].append(int(l))
						# else cumulative sum
						else :
							outConf[row][col] += int(l)
					break
		# after first classifier is done the output is no longer empty
		empty = False

	return outConf
#
#
#
################################################################################
################################################################################
# save the model for semi-supervised set and test it on whole set
#
#
def performSemiSupervised( fileName, extTest, n ) :
	# external test file?
	if extTest == "none" :
		# if none test on whole data set
		testOnMe = ( testSwitch + fileName[0:-5] + "_ID.arff" )
	else :
		# else in ID-ed copy of test file
		# create ID-ed copy of test file
		IDdata(extTest)
		testOnMe = ( testSwitch + extTest[0:-5] + "_ID.arff" )
	
	outConf = []
	tempConf = []

	# classify with IBk
	r = randint(1, 1000000)
	clas = subprocess.Popen( weka + meta1 + removeFilter1 + str(1) + meta2 +
		IBk1a + str(r) + trainingSwitch + fileName[0:-5] + "_labeledTraining." +
		"arff" + testOnMe + meta3 + IBk1b, stdout=subprocess.PIPE, shell=True )
	(out, err) = clas.communicate()
	tempConf.append(out)
	if err!=None:
		print "Error while classifying IBk:\n" + err
		exit()

	# classify with J48
	r = randint(1, 1000000)
	clas = subprocess.Popen( weka + meta1 + removeFilter1 + str(1) + meta2 +
		J481a + str(r) + trainingSwitch + fileName[0:-5] + "_labeledTraining." +
		"arff" + testOnMe + meta3 + J481b, stdout=subprocess.PIPE, shell=True )
	(out, err) = clas.communicate()
	tempConf.append(out)
	if err!=None:
		print "Error while classifying J48:\n" + err
		exit()

	# classify with SMO-Poly
	r = randint(1, 1000000)
	clas = subprocess.Popen( weka + meta1 + removeFilter1 + str(1) + meta2 +
		SMOP1a + str(r) + trainingSwitch + fileName[0:-5] + "_labeledTraining."+
		"arff" + testOnMe + meta3 + SMOP1b, stdout=subprocess.PIPE, shell=True )
	(out, err) = clas.communicate()
	tempConf.append(out)
	if err!=None:
		print "Error while classifying SMO-Poly:\n" + err
		exit()

	# classify with SMO-RBF
	r = randint(1, 1000000)
	clas = subprocess.Popen( weka + meta1 + removeFilter1 + str(1) + meta2 +
		SMOR1a + str(r) + trainingSwitch + fileName[0:-5] + "_labeledTraining."+
		"arff" + testOnMe + meta3 + SMOR1b, stdout=subprocess.PIPE, shell=True )
	(out, err) = clas.communicate()
	tempConf.append(out)
	if err!=None:
		print "Error while classifying SMO-RBF:\n" + err
		exit()

	# combine results into one common confusion matrix
	outConf = extractConfMx( tempConf, outConf )

	# multiply by n to have comparative results with supervised learning
	for i in range(len(outConf)) :
		for j in range(len(outConf[i])) :
			outConf[i][j] *= n

	return outConf
#
#
#
################################################################################
################################################################################
# save the model for semi-supervised set and test it on whole set
#
#
def performSupervised( fileName, extTest, n ) :
	# external test file?
	if extTest == "none" :
		testOnMe = ""
	else :
		# create ID-ed copy of test file
		IDdata(extTest)
		testOnMe = ( testSwitch + extTest[0:-5] + "_ID.arff" )

	# do full set training or only firstly selected training
	if "init" in fileName :
		trainOnMe = ( fileName[0:-5] + "_initial.arff" )
	else :
		trainOnMe = ( fileName[0:-5] + "_ID.arff" )

	# take a random seed
	r = 0

	outConf = []
	tempConf = []

	for i in range(n) :
		tempConf = []
		# classify with IBk
		r = randint(1, 1000000)
		clas = subprocess.Popen( weka + meta1 + removeFilter1 + str(1) + meta2 +
			IBk1a + str(r) + trainingSwitch + trainOnMe + 
			testOnMe + meta3 + IBk1b, stdout=subprocess.PIPE, shell=True )
		(out, err) = clas.communicate()
		tempConf.append(out)
		if err!=None:
			print "Error while classifying IBk:\n" + err
			exit()

		# classify with J48
		r = randint(1, 1000000)
		clas = subprocess.Popen( weka + meta1 + removeFilter1 + str(1) + meta2 +
			J481a + str(r) + trainingSwitch + trainOnMe + 
			testOnMe + meta3 + J481b, stdout=subprocess.PIPE, shell=True )
		(out, err) = clas.communicate()
		tempConf.append(out)
		if err!=None:
			print "Error while classifying J48:\n" + err
			exit()

		# classify with SMO-Poly
		r = randint(1, 1000000)
		clas = subprocess.Popen( weka + meta1 + removeFilter1 + str(1) + meta2 +
			SMOP1a + str(r) + trainingSwitch + trainOnMe + 
			testOnMe + meta3 + SMOP1b, stdout=subprocess.PIPE, shell=True )
		(out, err) = clas.communicate()
		tempConf.append(out)
		if err!=None:
			print "Error while classifying SMO-Poly:\n" + err
			exit()

		# classify with SMO-RBF
		r = randint(1, 1000000)
		clas = subprocess.Popen( weka + meta1 + removeFilter1 + str(1) + meta2 +
			SMOR1a + str(r) + trainingSwitch + trainOnMe + 
			testOnMe + meta3 + SMOR1b, stdout=subprocess.PIPE, shell=True )
		(out, err) = clas.communicate()
		tempConf.append(out)
		if err!=None:
			print "Error while classifying SMO-RBF:\n" + err
			exit()

		# combine results into one common confusion matrix
		outConf = extractConfMx( tempConf, outConf )

	return outConf
#
#
#
################################################################################
################################################################################
# print list with nice formating
#
#
def printList( ls ) :
	print "\n"
	for i in ls :
		tmp = "	"
		for j in i :
			tmp += ( str(j) + "	" )
		tm = tmp[0:-1]
		tm += "\n"
		print tm
#
#
#
################################################################################
################################################################################
# count trace and sum of matrix
#
#
def getStatistics( mx ) :
	diag = 0
	summed = 0

	# check whether it's squared matrix
	if len(mx) != len(mx[0]) :
		print "Matrix is not square!"
		exit()

	for i in range(len(mx)) :
		diag += mx[i][i]
		for j in range(len(mx[i])) :
			summed += mx[i][j]
	
	return (diag, summed)
#
#
#
################################################################################
################################################################################
# from multi-label file create single label file of selected index
#
#
def partialFile( orgFile, singleLabel, noLabels, allAtributesNo ) :
	newFile = ( orgFile[0:-5] + "_SL.arff" )
	IDrange = ( "\"first-" + str(allAtributesNo - noLabels) + ", " +
		str(allAtributesNo - singleLabel ) + "\"" )
	clas = subprocess.Popen( weka + removeFilter1 + IDrange + " -V " +
		removeFilter2 +	orgFile + removeFilter3 + newFile,
		stdout=subprocess.PIPE, shell=True )
	(out, err) = clas.communicate()
	if err!=None:
		print "Error while removing labels:\n" + err
		exit()
	return newFile
#
#
#
################################################################################
################################################################################
# accumulate two confusion matrices by pairwise summation
#
#
def accumulateLists( cumulative, results ) :
	# if cumulative empty it's firs iteration so copy
	if not cumulative :
		cumulative = results[:]
	# else add up
	else :
		for i in range(len(cumulative)) :
			for j in range(len(cumulative[i])) :
				cumulative[i][j] += results[i][j]

	return cumulative
#
#
#
################################################################################


# main program
#	read arguments to list
argumentList = list(sys.argv)

#	check correct number of arguments
if len(argumentList) < 2:
	print ( "There should be at least 1st argument given:" + "\n" +
		"	-*-1st: data set in 'arff' format" + "\n"
		"	-*-2nd: file containing labels in 'xml' format" )
	print( "If only 1st argument is given data set is treated as single class" +
		" classification.\n" + "If both arguments are given data is treated " +
		"as multi class classification an classified on each label " +
		"separately.\n" + "If you haven't supplied 2nd argument and you still" +
		" want to classify multi-label dataset please provide number of " +
		"labels when asked to confirm." )
		# + "\n" + "	-*- number of iterations" )
	exit()

#	if second argument not suppled consider data set as single labeled
if len(argumentList) == 2 :
	noLabels = 1
else :
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

#	count number of all attributes
allAtributes = None
if noLabels != 1 :
	(empty, empty, data) = handleInstances(list(open(argumentList[1])))
	allAtributes = countAtributes(data)

#
## TAG-A
#
cumulativeSemisup = []
cumulativeSup = []
cumulativeInitialSup = []
cumulativeRepetitions = 0
#	multi-label edition with 1 label at a time using
#	if more than one label to predict for each label create a separate single
#	class file classify it and repeat for each file adding up scores
for singleLabel in range(noLabels):
	if noLabels == 1 :
		argumentFilename = argumentList[1]
	else :
		# create first file
		print( "\n\nDoing " + str(singleLabel+1) + " out of " + str(noLabels) +
			" labels." )
		argumentFilename = partialFile( argumentList[1], singleLabel, noLabels,
			allAtributes )

	#	put ID as a first element of data
	IDdata(argumentFilename)

	#	open file with IDs
	rawFile = open(argumentFilename[0:-5] + "_ID.arff", 'r')
	fileList = list(rawFile)

	#	count the number of instances and get data list and header list
	(noInstances, arffHeader, data) = handleInstances(fileList)

	#	count number of attributes
	noAtributes = countAtributes(data)

	#	extract values of attribute to predict
	targetClasses = extractTargets( arffHeader )

	#	remove the ground truth for labels
	IDrangeRm = ( str(noAtributes-noLabels+1) + "-" + str(noAtributes) )
	rmAttributes(argumentFilename, IDrangeRm)

	#	get list of data instances with removed ground truth
	rawFileUnlabeled = open(argumentFilename[0:-5] + "_unlabeled.arff", 'r')
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

	#	whether to make next iteration
	cont = True
	while cont :
		# 1
		#	extract supIndexes and write to arff file
		(Training, Test) = createTT(supIndexes, data, targetClasses)
		(empty, unlabeledTest) = createTT(supIndexes, unlabeledData, targetClasses)

		#	convert lists to arff files and write set_training.arff and
		#	set_test.arff rmLabels decides whether to use labeled data or unlabeled
		#	as test set
		rmLabels = False # True
		saveToarff(argumentFilename, arffHeader, unlabeledArffHeader, Training, Test,
			unlabeledTest, rmLabels, sup)

		cont = True
		#	train classifiers on initial train set with mentioned schemes and test
		#	(predict) on rest and return raw outputs
		( rawIBk, rawJ48, rawSMOP, rawSMOR ) = trainClassifier(argumentFilename)

		#	make sens of outputs
		( IBk, J48, SMOP, SMOR ) = extractOutput( rawIBk, rawJ48, rawSMOP, rawSMOR )

		#	check matching predictions
		( predictionQnt, predictionInd, predictionClass ) = matchPredictions( IBk,
			J48, SMOP, SMOR )

		#	ask for numbers of samples to to add and boost classifier
		boostNums = None
		boostNum = 0
		priorityNum = None
		while ( (not boostNums) and (not priorityNum) ) :
			try:
				# give current statistics
				print( "========================================" +
					"========================================" )
				print( "There are: " + str(predictionQnt[0]) + " instances that " +
					"agree in all 4 classifiers." )
				print( "There are: " + str(predictionQnt[1]) + " instances that " +
					"agree in 3 out of 4 classifiers." )
				print( "There are: " + str(predictionQnt[2]) + " instances that " +
					"agree in 2 out of 4 classifiers." )
				print( "There are: " + str(predictionQnt[3]) + " instances that " +
					"agree in non of classifiers." )
				print( "Priority in choosing instances for boost operation is " +
					"given to ones that agrees in most of classifiers." )
				print( "========================================" +
					"========================================\n" )

				boostNums = raw_input( "How many out of " +
					str(noInstances-len(supIndexes)) + " instances do you want to" +
					" use to boost classifier?\nIf you want to stop boosting " +
					"operation and check accuracy of classifier put letter [s]: " )

				# if user put 's' stop boosting
				if boostNums == 's' :
					# go to 'cross-validating' classifier
					cont = False
					continue


				priorityNum = str( raw_input( "Give priority to [A]:4/4 | [B]: 3/4 | " +
					"[C]: 2/4 ?: " ) )
				print priorityNum
				# check if priorityNum is one of 'A', 'B' or 'C'
				if not( ( priorityNum ==  'A' ) or ( priorityNum ==  'B' ) or
					( priorityNum ==  'C' ) ) :
					print "Priority must be either of: 'A', 'B' or 'C'."
					priorityNum = None

				boostNum = int( boostNums )

				# number must be greater than 0
				if boostNum <= 0 :
					print "Number must be greater than 0!"
					boostNums = None

			except ValueError:
				print 'Invalid Number. I you want to stop put [s].'
				boostNums = None

		#	rebuilt classifier with # of samples defined by user choosing all
		#	instances where majority of classifiers agrees boils down to rebuilding
		# datasets and going back to stage #1
		if cont :
			# change order of classification
			if priorityNum == 'A' :
				pass
			elif priorityNum == 'B' :
				# i[b], i[a] = i[a], i[b]
				predictionInd[0], predictionInd[1] = predictionInd[1], predictionInd[0]
				predictionClass[0], predictionClass[1] = predictionClass[1], predictionClass[0]
			elif priorityNum == 'C' :
				predictionInd[0], predictionInd[1], predictionInd[2] = predictionInd[2], predictionInd[0], predictionInd[1]
				predictionClass[0], predictionClass[1], predictionClass[2] = predictionClass[2], predictionClass[0], predictionClass[1]
			else :
				print "Unknown order assuming [A]."

			supIndexes = rebuildSets( boostNum, predictionInd, predictionClass,
				supIndexes )


	#	then perform n-times with each of classifiers with cross validation
	#	to compare accuracy of results
	repetitions = None
	while not repetitions :
		try:
			repetitions = int( raw_input( "How many times do you want to perform " +
				"tests with c-v of supervised learning?: " ) )
			if repetitions <= 0 :
				repetitions = None
				print "Number must be greater than 0."
		except ValueError:
			print 'Invalid Number'

	#	ask for external test set if not supplied use whole set and cross-validation
	extTest = str( raw_input( "Please give name of external test file. If you " +
		"don't have one c-v will be performed on whole data set; in this case " +
		" type [none]: " ) )

	#	check accuracy of created data set
	semisupResults = performSemiSupervised( argumentFilename, extTest, repetitions )
	cumulativeSemisup = accumulateLists(cumulativeSemisup, semisupResults)
	supResults = performSupervised( argumentFilename, extTest, repetitions )
	cumulativeSup = accumulateLists(cumulativeSup, supResults)

	#	perform supervised on first selected elements in first iteration
	supInitialResults = performSupervised( (argumentFilename[0:-5] + ".init"),
		extTest, repetitions )
	cumulativeInitialSup = accumulateLists(cumulativeInitialSup,
		supInitialResults)

	#	accumulate repetitions
	cumulativeRepetitions += repetitions

#
## TAG-B
#

#	print confusion matrices for both
print( "Confusion matrix over " + str(cumulativeRepetitions) +
	" repetitions for SUPERVISED learning for full set:" )
printList( cumulativeSup )
print "\n"
print( "Confusion matrix over " + str(cumulativeRepetitions) +
	" repetitions for SUPERVISED learning for initial semi-sup set:" )
printList( cumulativeInitialSup )
print "\n"
print( "Confusion matrix for SEMI-SUPERVISED learning (each value multiplied " +
	" by number of repetitions):" )
printList( cumulativeSemisup )
print "\n"
(diag, summed) = getStatistics( cumulativeSup )
print( str(diag) + " instances out of " + str(summed) + " instances were " +
	"predicted correctly in SUPERVISED learning with full set." )
(diag, summed) = getStatistics( cumulativeInitialSup )
print( str(diag) + " instances out of " + str(summed) + " instances were " +
	"predicted correctly in SUPERVISED learning with initial semi-sup set." )
(diago, summedo) = getStatistics( cumulativeSemisup )
print( str(diago) + " instances out of " + str(summedo) + " instances were " +
	"predicted correctly in SEMI-SUPERVISED learning." )
