#include "Knn.h"
#include <iostream>
#include <string>
#include <fstream>
#include <ctime>
#include <time.h>
using namespace std;

int main(int argc, char** argv) {

	Knn k;

	clock_t begin = clock();
	//fstream f("iris.txt");
	//int i = 0;
	//f << i;
	//std::cout << i;
	k.readFile("iris.txt");

	clock_t end = clock();

	cout << "Time elapsed: " << ((float)(end - begin) / CLOCKS_PER_SEC) << " seconds" << endl;

	return 0;

}