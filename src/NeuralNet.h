/*
 * NeuralNet.h
 *
 *  Created on: Nov 11, 2016
 *      Author: andy
 */

#ifndef NEURALNET_H_
#define NEURALNET_H_

#include "Layer.h"

#include <vector>
#include <stdlib.h>
#include "math.h"
#include <iostream>

using namespace std;

class NeuralNet {
public:
	NeuralNet(int inputs, vector<int> l, float learningRate);
	virtual ~NeuralNet();

	int nLayers;
	vector<Layer> layers;
	float learningRate;

	void train( vector<float> &gt );
	vector<float> activate( vector<float> &input );
	void updateLearningRate(float &dlr);
};

#endif /* NEURALNET_H_ */
