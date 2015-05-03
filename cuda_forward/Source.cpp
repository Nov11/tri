#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <vector>
#include <unordered_map>
#include <cassert>
#include <algorithm>
#include "timeutility.h"
#include "Header.h"
using namespace std;



bool getInteger(char*& ptr, int& result)
{
	int sign = 1;
	result = 0;
	enum state { START, SIGN, PROC };
	state s = START;
	while (ptr != NULL){
		if (*ptr == NULL){
			if (s == PROC){
				result *= sign;
				return true;
			}
			else{
				return false;
			}

		}
		else if (isspace(*ptr)){
			while (*ptr != NULL && isspace(*ptr))ptr++;
			if (s == PROC){
				result *= sign;
				return true;
			}
			else if (s == START){
				continue;
			}
			else{
				return false;
			}
		}
		else if (*ptr <= '9' && *ptr >= '0'){
			result = result * 10 + *ptr - '0';
			if (s == START || s == SIGN){
				s = PROC;
			}
			ptr++;
		}
		else if (*ptr == '-'){
			sign = -1;
			if (s == START){
				s = SIGN;
			}
			else if (s == PROC){
				return false;
			}
			ptr++;
		}
	}
	//ptr==null
	return false;
}

bool getTwoInt(char*& p, int& node1, int& node2){
	bool ret1 = getInteger(p, node1);
	bool ret2 = getInteger(p, node2);
	if (ret1 == false || ret2 == false){
		return false;
	}
	return true;
}
void read_edges(ifstream& ifs, OriEdgeList& oel)
{
	string tmp;
	/*for (int i = 0; i < line_of_comments; i++){
		getline(ifs, tmp);
		if (i == line_to_read_node_and_edge){
			stringstream ss(tmp);
			while (ss >> tmp){
				if (tmp == "Nodes:"){
					ss >> oel.node_num;
				}
				else if (tmp == "Edges:"){
					ss >> oel.edge_num;
				}
			}
		}
	}*/
	while (ifs.peek() == '#'){
		getline(ifs, tmp);
		string::size_type pos;
		pos = tmp.find("Nodes:");
		if (pos != string::npos){
			stringstream ss(tmp);
			ss.seekg(pos, stringstream::beg);
			ss >> tmp;
			ss >> oel.node_num;
			ss >> tmp;
			ss >> oel.edge_num;
		}
	}
	cout << "node number: " << oel.node_num << " edge number: " << oel.edge_num << endl;
	if (oel.node_num == -1 || oel.edge_num == -1){
		cerr << " node_num or edge_num  not initialized. Aborting...." << endl;
		exit(1);
	}
	oel.edges.reserve(oel.edge_num);
	/*stringstream ss;
	ss << ifs.rdbuf();*/
	//cout << "after ss << ifs.rdbuf()  ss is " << ss.good() << " and ifs is " << ifs.good() << endl;
	ifstream::pos_type cur_pos = ifs.tellg();
	ifs.seekg(0, ifstream::end);
	ifstream::pos_type end_pos = ifs.tellg();
	ifs.seekg(cur_pos, ifstream::beg);
	size_t len = end_pos - cur_pos;
	char* p = new char[len + 1];
	ifs.read(p, len);
	p[len] = NULL;
	int node1, node2;
	char* tmp_ptr = p;
	//getTwoInt(tmp_ptr, node1, node2);
	//cout << node1 << "[[[[[[" << node2 << endl;
	//string test(p, 50);
	//cout << "!!" <<  test << "!! len " << len << endl;
	while (getTwoInt(tmp_ptr, node1, node2)){
		oel.edges.push_back(make_pair(node1, node2));
	}
	//cout << node1 << " !!! " << node2 << endl;
	cout << "read " << oel.edges.size() << " edges from file" << endl;
	delete[] p;
	//exit(1);
}

//void read_edges(ifstream& ifs, vector<pair<int, int>>& edges)
//{
//	string tmp;
//	int node_num = -1;
//	int edge_num = -1;
//	for (int i = 0; i < line_of_comments; i++){
//		getline(ifs, tmp);
//		if (i == 2){
//			stringstream ss(tmp);
//			while (ss >> tmp){
//				if (tmp == "Nodes:"){
//					ss >> node_num;
//				}
//				else if (tmp == "Edges:"){
//					ss >> edge_num;
//				}
//			}
//		}
//	}
//	cout << "node number: " << node_num << " edge number: " << edge_num << endl;
//	if (node_num == -1 || edge_num == -1){
//		cerr << " node_num or edge_num  not initialized. Aborting...." << endl;
//		exit(1);
//	}
//	edges.reserve(edge_num);
//	stringstream ss;
//	ss << ifs.rdbuf();
//	int node1, node2;
//	while (ss >> node1 >> node2){
//		edges.push_back(make_pair(node1, node2));
//	}
//	cout << "read " << edges.size() << " edges from file" << endl;
//}

int intersect(vector<int>& vi, vector<int>& vj)
{
	typedef vector<int>::iterator Iter;
	int ret = 0;
	for (Iter beg_vi = vi.begin(), beg_vj = vj.begin(); beg_vi != vi.end() && beg_vj != vj.end();){
		if (*beg_vi == *beg_vj){
			ret++;
			beg_vi++;
			beg_vj++;
		}
		else if (*beg_vi > *beg_vj){
			beg_vj++;
		}
		else{
			beg_vi++;
		}
	}
	return ret;
}

int forward(AdjList& adjl)
{
	AdjList A;
	int ret = 0;
	A.adj_list.resize(adjl.node_num);
	for (size_t i = 0; i < adjl.adj_list.size(); i++){
		for (auto& j : adjl.adj_list[i]){
			ret += intersect(A.adj_list[i], A.adj_list[j]);
			A.adj_list[j].push_back(i);
		}
	}
	return ret;
}

void sort_by_degree(unordered_map<int,int>& hashmap)
{
	vector<pair<int, int>> degree;
	degree.reserve(hashmap.size());
	for (auto& i : hashmap){
		degree.push_back(i);
	}
	sort(degree.begin(), degree.end(), [](pair<int, int>& p1, pair<int, int>& p2){return p1.second > p2.second; });
	int index = 0;
	for (auto& i : degree){
		hashmap[i.first] = index++;
	}
}



void make_adj_list(OriEdgeList& oel, AdjList& adjl, bool is_compact = false)
{
	assignOriToAdjl(oel, adjl);
	unordered_map<int, int> hashmap;
	for (auto& i : oel.edges){
		hashmap[i.first]++;
		hashmap[i.second]++;
	}
	sort_by_degree(hashmap);
	for (auto& i : oel.edges){
		int node1 = hashmap[i.first];
		int node2 = hashmap[i.second];
		if (node1 > node2){
			swap(node1, node2);
		}
		adjl.adj_list[node1].push_back(node2);
		if (is_compact){
			adjl.adj_list[node2].push_back(node1);
		}
	}
	if (is_compact){
		for (auto& i : adjl.adj_list){
			sort(i.begin(), i.end());
		}
	}
}

void make_adj_list_backward(OriEdgeList& oel, AdjList& adjl)
{
	make_adj_list(oel, adjl, true);
}

void make_adj_list_cuda(OriEdgeList& oel, AdjList& adjl)
{
	assignOriToAdjl(oel, adjl);
	std::unordered_map<int, int> hashmap;
	for (auto& i : oel.edges){
		hashmap[i.first]++;
		hashmap[i.second]++;
	}
	sort_by_degree(hashmap);
	for (auto& i : oel.edges){
		int node1 = hashmap[i.first];
		int node2 = hashmap[i.second];
		if (node1 > node2){
			std::swap(node1, node2);
		}
		adjl.adj_list[node2].push_back(node1);
	}
	for (auto& i : adjl.adj_list){
		sort(i.begin(), i.end());
	}
}

int reverse_edge_forward(AdjList& adjl)
{
	int ret = 0;
	for (size_t i = 0; i < adjl.adj_list.size(); i++){
		for (auto& j : adjl.adj_list[i]){
			vector<int>& node1 = adjl.adj_list[i];
			vector<int>& node2 = adjl.adj_list[j];
			ret += intersect(node1, node2);
		}
	}
	return ret;
}

int intersect_compact_forward(vector<vector<int>>& list, int i, int j)
{
	int ret = 0;
	typedef vector<int>::iterator Iter;
	for (Iter iter1 = list[i].begin(), iter2 = list[j].begin(); iter1 != list[i].end() && iter2 != list[j].end();){
		if (*iter1 > i || *iter2 > i){
			break;
		}
		if (*iter1 == *iter2){
			ret++;
			iter1++;
			iter2++;
		}
		else if (*iter1 < *iter2){
			iter1++;
		}
		else{
			iter2++;
		}
	}
	return ret;
}

int compact_forward(AdjList& adjl)
{
	int ret = 0;
	for (size_t i = 0; i < adjl.adj_list.size(); i++){
		for (auto& j : adjl.adj_list[i]){
			if (j < (long)i){
				continue;
			}
			ret += intersect_compact_forward(adjl.adj_list, i, j);
		}
	}
	return ret;
}

int serial_reverse_edge_foward(ifstream& ifs)
{
	AdjList adjl;
	{
		OriEdgeList oel;
		read_edges(ifs, oel);
		make_adj_list_cuda(oel, adjl);
	}
	CpuTime ct;
	ct.startTimer();
	int ret = reverse_edge_forward(adjl);
	ct.stopAndPrint("!!!reverse forward core !!!");
	return ret;
}

int serial_compact_forward(ifstream& ifs)
{
	AdjList adjl;
	{
		OriEdgeList oel;
		read_edges(ifs, oel);
		make_adj_list_backward(oel, adjl);
	}
	CpuTime ct;
	ct.startTimer();
	int ret = compact_forward(adjl);
	ct.stopAndPrint("!!!compact forward core!!!");
	return ret;
}

int serial_forward(ifstream& ifs)
{
	AdjList adjl;
	{
		OriEdgeList oel;
		read_edges(ifs, oel);

		make_adj_list(oel, adjl);
	}
	CpuTime ct;
	ct.startTimer();
	int ret = forward(adjl);
	ct.stopAndPrint("!!!forward core!!!");
	return ret;
}

void serial_test(string& file, int result[], int item, int(*func)(ifstream& ifs) = serial_forward)
{
	cout << "processing file " << file << endl;
	CpuTime ct;
	ct.startTimer();
	ifstream ifs(file, ios::binary);
	if (!ifs){
		cerr << "error open file: " << file << endl;
		exit(1);
	}
	int ret = func(ifs);
	ct.stopAndPrint("triangle counting finished.");
	cout << ret << " triangles" << endl;
	if (ret == result[item]){
		cout << "[[Correct]]. " << endl;
	}
	else{
		cout << "Incorrect." << endl;
	}
	cout << "===============================" << endl;
}
int main()
{
	int result[] = { 667129, 28769868, 177820130 };
	string file_name[] = { "com-amazon.ungraph.txt", "as-skitter.txt", "com-lj.ungraph.txt" };
	//edges 925872 11095298 34681189
	//nodes	334863 1696415  3997962
	string file_path = "f:\\triangle\\";

	int item = 2;
	string file = file_path + file_name[item];
	//serial_forward(file, result, item);
	//cuda_forward(file, result, item);
	vector<int(*)(ifstream& ifs)> functions = { serial_forward, serial_compact_forward
											, serial_reverse_edge_foward, serial_cuda_forward
											, serial_cuda_forward_without_tail, cuda_forward 
											};
	//for (auto& i : functions){
	//	serial_test(file, result, item, i);
	//}
	serial_test(file, result, item, cuda_forward);

}