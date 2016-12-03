/*
 * Layer.cpp
 *
 *  Created on: Oct 26, 2016
 *      Author: andy
 */



#include "Layer.h"

void printVector(vector<float> &in, string name);

Layer::Layer(int nNodes, int nll, int depth, float learningRate){

	this->depth = depth;
	this->nNodes = nNodes;
	for(int n=0; n<nNodes; n++){
		Node a(nll, learningRate);
		nodes.push_back(a);
		outputs.push_back(0);
	}
	this->learningRate = learningRate;
}

Layer::~Layer() {}

void Layer::trainOuterLayer( vector<float> &gt ){

	for(int n=0; n<nNodes; n++){
		nodes[n].backProp( gt[n] );
	}
}

void Layer::trainHiddenLayer( Layer &layerPlus ){

	for(int n=0; n<nNodes; n++){
		vector<float> weightsPlus = layerPlus.getWeightsPlus( n );
		nodes[n].backProp( layerPlus.deltas,  weightsPlus );
	}
}

vector<float> Layer::getWeightsPlus( int &n ){
	vector<float> wp;

	for(int i=0; i<nNodes; i++){
		wp.push_back( nodes[i].weights[n] );
	}
	return wp;
}

vector<float> Layer::getOutputs(){
	for(int n=0; n<nNodes; n++){
		outputs[n] = nodes[n].output;
	}
	return outputs;
}

void Layer::getDeltas(){
	for(int n=0; n<nNodes; n++){
		deltas[n] = nodes[n].delta;
	}
}

vector<float> Layer::activateFastSigmoid(vector<float> &inputs){

	for(int i=0; i<nNodes; i++){
		outputs[i] = nodes[i].activateFastSigmoid(inputs);
	}
	return outputs;
}

vector<float> Layer::activateSigmoid(vector<float> &inputs){
	for(int i=0; i<nNodes; i++){
		outputs[i] = nodes[i].activateSigmoid(inputs);
	}
	return outputs;
}

void Layer::updateLearningRate(float &dlr){
	for(int i=0; i<nNodes; i++){
		nodes[i].learningRate *= dlr;
	}
}

void printVector(vector<float> &in, string name){

	cout << name << ": ";
	for(size_t i=0; i<in.size(); i++){
		cout << in[i];
		if( i+1 <= in.size() ){
			cout << ", ";
		}
	}
	cout << endl;
}

