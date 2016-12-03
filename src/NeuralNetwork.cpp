//============================================================================
// Name        : NeuralNetwork.cpp
// Author      : 
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
#include <math.h>
#include <cstdlib>
#include <stdlib.h>
#include <time.h>
#include <fstream>
#include <sstream>
#include <cmath>
#include <vector>

#include "NeuralNet.h"

void printVector(vector<float> in, string name);

using namespace std;

int main(){

	//srand(time(NULL));

	float initLearningRate = 0.25;
	int input = 1;
	vector<int> layerSize;
	layerSize.push_back(10);
	layerSize.push_back(10);
	layerSize.push_back(1);
	NeuralNet net(input, layerSize, initLearningRate);

	float goal = 1;
	float dlr = 1;
	for( int i=0; i<1000000; i++){

		net.updateLearningRate(dlr);

		vector<float> input;
		input.push_back( float(rand() % 2000) / 10000 );
		//input.push_back( float(rand() % 2000) / 10000 );
		//input.push_back( float(rand() % 2000) / 10000 );

		goal = 3*input[0];//*sin(input[0]);// + pow(input[1],2) + pow(input[2],4);

		vector<float> out = net.activate(input);


		vector<float> gt;
		gt.push_back( goal );
		net.train( gt );

		cout << i << " : goal = " << goal << " and output = " << out[0] << endl;
	}

	return 0;
}


