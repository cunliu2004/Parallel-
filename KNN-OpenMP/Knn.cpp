#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <typeinfo>
#include <algorithm>
#include <cmath>
#include <valarray>
#include "Knn.h"

using namespace std;

Knn::Knn() {

}

Knn::~Knn() {

}

void Knn::readFile(string file) {

	string line;

	ifstream reader;
	reader.open(file);

	vector<string> splitString;

	//first pass changes classification to a numerical classification
	/*
		1 = iris-setosa
		2 = iris-versicolor
		3 = iris-virginica
	*/
	if (reader.is_open()) {

		//clear test file (used for running multiple times)
		system("./ClearFile.sh");

		//fix dataset to have numerical response variables
		while (getline(reader, line)) {
			splitString = split(line, ',');

			ofstream writer("irisNumClass.txt", ios::app);
			if (writer.is_open()) {
				for (string s : splitString) {
					if (s == "Iris-setosa") {
						writer << "1" << endl;
					}
					else if (s == "Iris-versicolor") {
						writer << "2" << endl;
					}
					else if (s == "Iris-virginica") {
						writer << "3" << endl;
					}
					else {
						writer << s << ",";
					}
				}
				writer.close();
			}
		}

		reader.close();
	}
	else {
		cout << "file not found" << endl;
	}

	//clear variables to use them again
	line = "";

	//second pass reads each line into a vector and pushes that vector into a vector of vectors
	reader.open("irisNumClass.txt");

	double r;
	int train = 0;
	int test = 0;

	//split data into training and test data
	if (reader.is_open()) {
		while (getline(reader, line)) {
			splitString = split(line, ',');

			r = ((double)rand() / (RAND_MAX));

			if (r < .8) {
				trainingDataset.push_back(splitString);
			}
			else {
				testingDataset.push_back(splitString);
			}

		}
		reader.close();
	}

	int count = 0;

	//classify the given test instances
	for (int i = 0; i < testingDataset.size(); ++i)
	{
		int predictedClass = classify(neighbors(trainingDataset, testingDataset[i], 5));
		cout << "The predicted class of the given data instance is: ";

		for (string s : testingDataset[i]) {
			cout << s << ",";
		}
		cout << "-> ";
		switch (predictedClass) {
		case 1:
			cout << "Iris-setosa" << endl;
			break;
		case 2:
			cout << "Iris-versicolor" << endl;
			break;
		case 3:
			cout << "Iris-virginica" << endl;
			break;
		default:
			cout << "something went wrong" << endl;
			break;
		}
		count++;
	}

	cout << "Classified " << count << " instances." << endl;
	/*//print to make sure it read in correctly
	for (vector<string> v : trainingDataset) {
		for(string s : v) {

			cout << stof(s) << ",";

			if(s == "1" || s == "2" || s == "3") {
				cout << endl;
			}
		}
	}

	cout << "size: " << trainingDataset.size() << endl;
	cout << "\n-----------\n" << endl;

	for (vector<string> v : testingDataset) {
		for(string s : v) {

			cout << stof(s) << ",";

			if(s == "1" || s == "2" || s == "3") {
				cout << endl;
			}
		}
	}

	cout << "size: " << testingDataset.size() << endl;*/

}
//衡量相似度(余弦相似度)
double Knn::dotProductSimilarity(vector<string> v1, vector<string> v2) {

	double topSum = 0;
	double v1SqSum = 0;
	double v2SqSum = 0;
	//转为浮点数计算每个距离的平方
	for (int i = 0; i < v1.size() - 1; ++i)
	{
		topSum += stof(v1[i]) * stof(v2[i]);
		v1SqSum += pow(stof(v1[i]), 2);
		v2SqSum += pow(stof(v2[i]), 2);
	}
	//计算L2范数
	double bottom1 = pow(v1SqSum, .5);
	double bottom2 = pow(v2SqSum, .5);
	double dotProduct = topSum / (bottom1 * bottom2);

	return dotProduct;
}

vector < vector<string> > Knn::neighbors(vector< vector<string> > trainSet, vector<string> testInstance, int numOfNeighbors) {

	vector < vector<string> > distances;
	double vDistances = 0;
	vector<string> instanceAndDist;

#if defined(_OPENMP)
	omp_lock_t writelock;

	omp_init_lock(&writelock);
#endif

	//caclulate cosine similarity and append them to the training set
#pragma omp parallel for schedule(auto)
	for (int i = 0; i < trainSet.size(); ++i)
	{
#if defined(_OPENMP)
		omp_set_lock(&writelock);
#endif
		vDistances = dotProductSimilarity(testInstance, trainSet[i]);

		//append distance to specific instance
		trainSet[i].push_back(to_string(vDistances));

#if defined(_OPENMP)
		omp_unset_lock(&writelock);
#endif
	}

	//sort the training set by the Cosine similarity
	sort(trainSet.begin(), trainSet.end(), [](const vector<string>& a, const vector<string>& b) {
		return stof(a[5]) < stof(b[5]);
		});

	//take the k-nearest neighbors and add them to the neigbors vector
	vector< vector<string> > neighbors;

	for (int i = 0; i < numOfNeighbors; ++i)
	{
		neighbors.push_back(trainSet[trainSet.size() - (i + 1)]);
	}

	cout << "num of neighbors: " << neighbors.size() << endl;

	for(vector<string> v : neighbors) {
		for(string s : v) {
			cout << stof(s) << ",";
		}
		cout << endl;
	}

	return neighbors;
}

//calculate which response variable has the highest count among all of the neighbors and classify based on that
int Knn::classify(vector< vector<string> > n) {

	int counts[] = { 0, 0, 0 };

	int value;

	for (vector<string> v : n) {
		value = stoi(v[4]) - 1;
		counts[value]++;
	}

	int max = counts[0];
	int maxindex = 0;

	for (int i = 0; i < 3; ++i)
	{
		if (max < counts[i]) {
			max = counts[i];
			maxindex = i;
		}
	}
	//cout << maxindex + 1 << endl;
	return maxindex + 1;

}

//custom split method
vector<string> Knn::split(string s, char c) {
	string temp = "";
	vector<string> v;

	for (int i = 0; i < s.length(); ++i)
	{
		if (s[i] != c) {
			temp += s[i];
		}
		else if (s[i] == c && temp != "") {
			v.push_back(temp);
			temp = "";
		}
	}
	if (temp != "") {
		v.push_back(temp);
	}
	return v;
}