/*
 * Node.cpp
 *
 *  Created on: Oct 26, 2016
 *      Author: andy
 */

#include "Node.h"

void printVector(vector<float> in, string name);
float dotProduct(vector<float> &a, vector<float> &b);
float abs( float &x );

Node::Node(int nll, float learningRate){
	for(int i=0; i<nll; i++){
		weights.push_back( float(rand() % 100000) / 100000);
	}
	weights.push_back( float(rand() % 100000) / 100000);

	weightedSum = 0;
	delta = 0;
	output = 0;
	this->learningRate = learningRate;
}

Node::~Node(){}


float Node::activateFastSigmoid(vector<float> &inputs){
	this->inputs = inputs;
	this->inputs.push_back(1);
	//printVector( inputs, "activate::inputs");
	//printVector( weights, "activate::weights");
	weightedSum = dotProduct(weights, this->inputs);
	//cerr << "wSum" << endl;
	output = 0.5+0.5*weightedSum / (1 + abs(weightedSum) );

	//cout << "activate::output: " << output << endl;

	//printVector( weights, "activate::weights");
	return this->output;
}

float Node::activateSigmoid(vector<float> &inputs){
	this->inputs = inputs;
	this->inputs.push_back(1);

	//printVector( inputs, "activate::inputs");
	//printVector( weights, "activate::weights");

	weightedSum = dotProduct(weights, inputs);
	//cout << "activate::weightedSum: " << weightedSum << endl;
	output = 1/(1 + exp(-weightedSum));

	//cout << "activate::output: " << output << endl << endl;

	return output;
}

void Node::backProp(vector<float> &deltaPlus, vector<float> &wieghtsPlus ){

	this->delta = dotProduct( deltaPlus, wieghtsPlus)*this->output*(1-this->output);

	for(size_t i=0; i<this->weights.size(); i++){
		weights[i] += this->learningRate * this->delta * inputs[i];
	}
}

void Node::backProp( float &gt ){

	//printVector( weights, "backprop::weights");

	float error = gt - this->output;
	this->delta = error*gt*(1-gt);
	//cout << "backprop::output: " << output << endl;
	//cout << "backprop::delta: " << delta << endl;
	//cout << "backprop::error: " << error << endl;
	//printVector(inputs, "backprop::inputs");

	for(size_t i=0; i<this->weights.size(); i++){
		//cout << "adjust: " << this->learningRate * this->delta * inputs[i] << endl;
		weights[i] += this->learningRate * this->delta * inputs[i];
	}
	//printVector(weights, "backprop::weights");

	//cin.ignore();
}

void Node::evolve(){
	//printVector(weights, "evolve::weights prior");
	for(size_t i=0; i<this->weights.size(); i++){
		weights[i] += this->learningRate * (float(rand() % 1000) / 500 - 1); // lr * rand(-1 -> 1)
	}
	//printVector(weights, "evolve::weights");
}

float abs( float &x ){
	if( x<0 ){
		x *= -1;
		return x;
	}
	else{
		return x;
	}
}

float dotProduct(vector<float> &a, vector<float> &b){
	if(a.size() - b.size() == 0){
		float sum = 0;
		for(size_t i=0; i<a.size(); i++){
			sum += a[i] * b[i];
		}
		return sum;
	}
	else{
		cerr << "BAD DOT PRODUCT" << endl;
		return -1;
	}
}
