	/*
	 * Layer.h
	 *
	 *  Created on: Oct 26, 2016
	 *      Author: andy
	 */

	#ifndef LAYER_H_
	#define LAYER_H_

	#include "Node.h"
	#include <vector>
	#include <stdlib.h>
	#include "math.h"
	#include <iostream>

	using namespace std;

	class Layer {
	public:
		Layer(int nnodes, int nll, int depth, float learningRate);
		virtual ~Layer();

		vector<float> outputs;
		vector<float> deltas;

		vector<Node> nodes;
		int nNodes;
		int depth;

		float learningRate;

		void getDeltas();
		vector<float> getOutputs();
		vector<float> getWeightsPlus(int &n);

		vector<float> activateFastSigmoid( vector<float> &inputs );
		vector<float> activateSigmoid( vector<float> &inputs );

		void trainHiddenLayer( Layer &layerPlus );
		void trainOuterLayer( vector<float> &groundTruth );

		void updateLearningRate(float &dlr);

	};

	#endif /* LAYER_H_ */
