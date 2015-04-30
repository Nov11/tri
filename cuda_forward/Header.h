#ifndef CUDA_FORWARD_
#define CUDA_FORWARD_

#include <vector>
#include <fstream>

struct OriEdgeList{
	std::vector<std::pair<int, int>> edges;
	int node_num = -1;
	int edge_num = -1;
};

struct AdjList{
	std::vector<std::vector<int>> adj_list;
	int node_num = -1;
	int edge_num = -1;
};

inline void assignOriToAdjl(OriEdgeList& oel, AdjList& adjl)
{
	adjl.edge_num = oel.edge_num;
	adjl.node_num = oel.node_num;
	adjl.adj_list.resize(adjl.node_num);
}

void read_edges(std::ifstream& ifs, OriEdgeList& oel);
void sort_by_degree(std::unordered_map<int, int>& hashmap);
void make_adj_list_cuda(OriEdgeList& oel, AdjList& adjl);
int cuda_forward(std::ifstream& ifs);
int serial_cuda_forward(std::ifstream& ifs);
int serial_cuda_forward_without_tail(std::ifstream& ifs);
#endif