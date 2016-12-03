/*
 * Node.h
 *
 *  Created on: Oct 26, 2016
 *      Author: andy
 */

#ifndef NODE_H_
#define NODE_H_

#include <vector>
#include <stdlib.h>
#include "math.h"
#include <iostream>

using namespace std;

class Node {
public:
	Node(int nll, float learningRate);
	virtual ~Node();

	float activateFastSigmoid(vector<float> &inputs);
	float activateSigmoid(vector<float> &inputs);

	void backProp(vector<float> &deltaPlus, vector<float> &weightsPlus);
	void backProp(float &gt);

	void evolve();

	float output;
	float delta;
	float learningRate;
	float weightedSum;
	vector<float> weights;
	vector<float> inputs;

};

#endif /* NODE_H_ */
