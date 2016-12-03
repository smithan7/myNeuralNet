/*
 * NeuralNet.cpp
 *
 *  Created on: Nov 11, 2016
 *      Author: andy
 */

#include "NeuralNet.h"

NeuralNet::NeuralNet(int inputs, vector<int> layerSize, float learningRate) {

	Layer layer0(layerSize[0], inputs, 0, learningRate);
	layers.push_back( layer0 );

	nLayers = layerSize.size();
	for(int la = 1; la<nLayers; la++){
		Layer layer(layerSize[la], layerSize[la-1], la, learningRate);
		layers.push_back(layer);
	}

	this->learningRate = learningRate;
}

NeuralNet::~NeuralNet(){}

void NeuralNet::train( vector<float> &gt ){

	layers[layers.size()-1].trainOuterLayer( gt );
	for(size_t l=layers.size()-2; l>-1; l--){
		layers[l].trainHiddenLayer( layers[l+1] );
	}
}

vector<float> NeuralNet::activate( vector<float> &inputs ){

	for(int l=0; l<nLayers; l++){
		inputs = layers[l].activateFastSigmoid( inputs );
	}
	return inputs;
}

void NeuralNet::updateLearningRate(float &dlr){
	for(int l=0; l<nLayers; l++){
		layers[l].updateLearningRate(dlr);
	}
}



