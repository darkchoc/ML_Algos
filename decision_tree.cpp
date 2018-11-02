#include <bits/stdc++.h>

using namespace std;

int num_features;
vector<vector<double> > data;
int min_child;

struct Node {
	struct Node* left_child;
	struct Node* right_child;
	struct Node* parent;
	vector<int> data_points;
	int num_pos;
	int num_neg;
	bool is_leaf;
	int split_feat;
	double split_val; //will store class probability in case of leaf node.
	Node(): left_child(NULL), right_child(NULL), parent(NULL), num_pos(0), num_neg(0), is_leaf(false), split_feat(0), split_val(0) {}
};

typedef struct Node Node;

double CalEntropy(int left_recs, int left_pos, int left_neg, int right_recs, int right_pos, int right_neg){
	if(left_recs==0 || right_recs==0)
			return 0;
	double entropy_left;
	double entropy_right;
	if(left_neg==0 || left_pos==0)
			entropy_left=0;
	else
		entropy_left =  -(((double)left_pos/(double)left_recs)*log2((double)left_pos/(double)left_recs)) - ((double)left_neg/(double)left_recs)*log2((double)left_neg/(double)left_recs);
	if(right_neg==0|| right_pos==0)
			entropy_right=0;
	else
		entropy_right = -(((double)right_pos/(double)right_recs)*log2((double)right_pos/(double)right_recs)) - ((double)right_neg/(double)right_recs)*log2((double)right_neg/(double)right_recs);
	entropy_right = ((double)right_recs/(double)(right_recs+left_recs))*entropy_right;
	entropy_left = ((double)left_recs/(double)(left_recs+right_recs))*entropy_left;
	return entropy_right+entropy_left;
}

void SplitNode(Node *node){
	//cout << "SplitNode called and pos=" << node->num_pos << " neg=" << node->num_neg << " dataset="; 
	//for(int i = 0 ; i < node->data_points.size(); i++){
	//	cout << node->data_points[i] << " ";
	//}
	//cout << endl;
	if(node->num_pos==0 || node->num_neg==0 || node->data_points.size()<=min_child){
		//write code to make it a leaf 
	//	cout << "making a leaf" << endl;
		node->is_leaf=true;
		node->split_val = node->num_pos/(node->num_pos + node->num_neg);
		return;
	}
	int selected_feat;
	double split_val;
	double parent_entropy;
	parent_entropy = -((double)node->num_pos/(double)node->data_points.size())*log2((double)node->num_pos/(double)node->data_points.size());
	parent_entropy += -((double)node->num_neg/(double)node->data_points.size())*log2((double)node->num_neg/(double)node->data_points.size());
	//cout << "Parent entropy is : " << parent_entropy << endl;
	double max_info_gain = -1;
	for(int feat = 1; feat < num_features; feat++){
		//First find different attrib values
		set<double> values;
		for(int i = 0 ; i < node->data_points.size(); i++){
			values.insert(data[node->data_points[i]][feat]);
		}
		if (values.size()==1)
			continue;
		for(auto it: values){
			if(it == *values.rbegin())
				continue;
			int left_node_recs=0;
			int right_node_recs=0;
			int left_node_pos=0;
			int left_node_neg=0;
			int right_node_pos=0;
			int right_node_neg=0;
			for(int i = 0; i < node->data_points.size(); i++){
				if(data[node->data_points[i]][feat] <= it){
					left_node_recs++;
					if(data[node->data_points[i]][0]==1)
						left_node_pos++;
					else
						left_node_neg++;
				}
				else{
					right_node_recs++;
					if(data[node->data_points[i]][0]==1)
						right_node_pos++;
					else
						right_node_neg++;
				}
			}
			//cout << "Considering: " << feat << " with val " <<  it << endl;
			//cout << left_node_recs << " " << left_node_pos << " " << left_node_neg << " " << right_node_recs << " " << right_node_pos << " " << right_node_neg << endl; 
			double child_entropy = CalEntropy(left_node_recs, left_node_pos, left_node_neg, right_node_recs, right_node_pos, right_node_neg);
			if(max_info_gain < parent_entropy - child_entropy){
				max_info_gain = parent_entropy - child_entropy;
				selected_feat = feat;
				split_val = it;
			}
		}
	}
	//cout << "Max Info gain: " << max_info_gain << " Feature selected: " << selected_feat << " split_val: " << split_val << endl;
	if(max_info_gain== -1){
		//means records same 
		node->is_leaf = true;
		node->split_val = node->num_pos/(node->num_pos + node->num_neg);
		return;
	}
	node->split_feat = selected_feat;
	node->split_val = split_val;
	//Now make left child and right child
	node->left_child = new Node();
	node->right_child = new Node();
	node->left_child->parent = node;
	node->right_child->parent = node;
	for(int i = 0 ; i < node->data_points.size(); i++){
		if(data[node->data_points[i]][selected_feat] <= split_val){
			node->left_child->data_points.push_back(node->data_points[i]);
			if(data[node->data_points[i]][0]==1)
				node->left_child->num_pos++;
			else 
				node->left_child->num_neg++;
		}
		else{
			node->right_child->data_points.push_back(node->data_points[i]);
			if(data[node->data_points[i]][0]==1)
				node->right_child->num_pos++;
			else
				node->right_child->num_neg++;
		}
	}
	SplitNode(node->left_child);
	SplitNode(node->right_child);
	return;
}

void Predict_rec(Node* root, int index){
	Node* current_node = root;
	while(current_node->is_leaf!=true){
		if(data[index][current_node->split_feat] <= current_node->split_val){
			current_node = current_node->left_child;
		}
		else{
			current_node = current_node->right_child;
		}
	}
	cout << current_node->split_val << endl;
	return;
}

void PrintTree(Node* root, std::ofstream& outfile){
	if(root->is_leaf){
		outfile << "Leaf: " << root->split_val << " P:" << root->num_pos << " N:" << root->num_neg << endl;
	}
	else{
		outfile << "Node: " << root->split_feat << " " << root->split_val << " P:" << root->num_pos << " N:" << root->num_neg << endl;
		PrintTree(root->left_child, outfile);
		PrintTree(root->right_child, outfile);
	}
	return;
}

int main(){
	ios_base::sync_with_stdio(false);
	//data format
	// only numerical values
	//each row represents one instance and number of columns represents the attribute value
	//First line of data consists of two number, num_records and num_features.
	// First column in label (0/1)
	int num_records;
	string file_path;
	cout << "Enter train data path: " << endl;
	cin >> file_path;
	cout << "Enter min child: " << endl;
	cin >> min_child;
	fstream my_file (file_path);
	if(my_file.is_open()){
		my_file >> num_records;
		my_file >> num_features;
		data.resize(num_records, vector<double> (num_features));
		for(int i = 0 ; i < num_records; i++){
			for(int j = 0; j < num_features; j++){
				my_file >> data[i][j];
			}
		}
		my_file.close();
	}
	Node* root = new Node();
	for(int i = 0; i < num_records; i++){
		root->data_points.push_back(i);
		if(data[i][0]==1)
			root->num_pos++;
		else
			root->num_neg++;
	}
	SplitNode(root);
	string print_file;
	cout << "Enter file to which to print: " << endl;
	cin >> print_file;
	std::ofstream outfile;
	outfile.open(print_file, std::ios_base::app);
	PrintTree(root, outfile);

	cout << "Enter test file path" << endl;
	cin >> file_path;
	fstream test_file(file_path);
	if(test_file.is_open()){
		test_file >> num_records;
		test_file >> num_features;
		data.resize(num_records, vector<double> (num_features));
		for(int i = 0; i < num_records; i++){
			for(int j =0 ; j < num_features; j++){
				test_file >> data[i][j];
			}
		}
		test_file.close();
	}
	for(int i = 0 ; i < num_records; i++){
		Predict_rec(root, i);
	}
}