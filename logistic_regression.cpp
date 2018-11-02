//Basic and minimalistic logistic regression for dense data, SGD
//assuming no missing values
//Data only numerical
//Data in format -> First line gives number of featuers
//Then, Label feat1_value feat2_value feat3_value ...
//Initially assuming data can be fitted in memory

//Later on, introduce feature to read data in chunks, do Gradient Descent (Using complete data for making one weight update)
//Also introduce mini batch gradient descent

#include <bits/stdc++.h>
#include <iostream>
#include <fstream>
#include <math.h>

using namespace std;

vector < vector <double> > data;
vector <double> labels; // filled during reading data 
vector <double> weights; // initialized to all 0's while reading data
vector <double> predictions; //untransformed predictions .. resized in the starting while reading data
vector <double> gradients; // initialized to all 0's while reading data
enum Setup{
	Gradient_Descent,
	SGD,
	MBGD
};

double sigmoid(double x){
	return (1/(1+exp(-x))); 
}

int readData(){
	string file_path;
	cout << "Enter Training Data File Path:" << endl; 
        cin >> file_path;
        fstream myFile(file_path);
	int features;
        if(myFile.is_open()){
                double temp;
                myFile >> features;
                while(!myFile.eof()){
                        data.resize(data.size()+1);
                        int data_size = data.size();
                        myFile >> temp;
                        labels.push_back(temp);
                        data[data_size-1].push_back(1); //pushing feature corresponding to bias
                        for(int d = 0; d < features; d++){
                                myFile >> temp;
                                data[data_size - 1].push_back(temp);
                        }
                }
		data.pop_back();
                myFile.close();
		for(int w = 0 ; w < features + 1; w++){
			weights.push_back(0);
			gradients.push_back(0);
		}
		predictions.resize(data.size()); //untransformed predictions
		return features;
        }
        else{
                cout << "bro, data file ke bina na ho paaega!" << endl;
                return -1;
        }
}

void PredictRow(int d){
	predictions[d]=0.0;
	for(int w=0; w< weights.size(); w++)
		predictions[d] += data[d][w]*weights[w];
	
}

void UpdateGradsOneInstance(int d){
	for(int g = 0; g < gradients.size(); g++)
		gradients[g] = (sigmoid(predictions[d]) - labels[d])*data[d][g];	
}

void UpdateGradsFullData(){
	for(int g = 0; g < gradients.size(); g++)
		gradients[g] = 0;
	for(int d =0 ; d< data.size(); d++){
		for( int g = 0; g < gradients.size(); g++)
			gradients[g] += (sigmoid(predictions[d]) - labels[d])*data[d][g];
	}
	for( int g = 0; g < gradients.size(); g++)
		gradients[g] = gradients[g]/data.size();
}

void UpdateWeights(double learning_rate){
	for(int w = 0; w < weights.size(); w++){
		weights[w] = weights[w] - learning_rate*(gradients[w]);
	}	
}

void LogisticRegression(int passes, int features, enum Setup setup, double learning_rate){
	for(int pass = 0; pass < passes; pass++){
		if(setup==0){
		//GD
			for(int d = 0; d < data.size() ; d++)
				PredictRow(d);
			UpdateGradsFullData();
			UpdateWeights(learning_rate);
		}
		else if(setup==1){
		//SGD
			for(int d = 0 ; d < data.size(); d++){
				PredictRow(d);	
				UpdateGradsOneInstance(d);
				UpdateWeights(learning_rate);
			}
		}
		else{
		//MBGD
		}
	}
}

void PrintWeights(){
	for(int w = 0 ; w < weights.size(); w++){
		cout << "Feature " << w << ": " << weights[w];
		cout << endl;
	}
}

int main(int argc, char *argv[]){
	ios_base::sync_with_stdio(false);
	int passes = 3, features;
	double learning_rate=0.1;
	if(argc==1)
		cout << "Using default number of passes : 3" << endl;
	else{
		passes = atoi(argv[1]);
		cout << "Using " << passes << " number of passes." << endl;
	}
	features = readData();
	if(features==-1)
		return 0;
	enum Setup setup;
	cout << "Enter type of logistic regression to fit: (GD=0, SGD=1, MBGD=2)" << endl;
 	int setting;
	cin >> setting;
	if(setting == 0 )
		setup = Gradient_Descent;
	else if (setting == 1)
		setup = SGD;
	else if(setting == 2)
		setup = MBGD;
	else{
		cout << "Ye wala nai aata" << endl;
		return 0;
	}
	cout << "Enter learning rate: " << endl;
	cin >> learning_rate;
	LogisticRegression(passes, features, setup, learning_rate);	
	PrintWeights();
	return 0;
}

