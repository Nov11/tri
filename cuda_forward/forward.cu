#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <string>
#include <fstream>
#include <unordered_map>
#include <algorithm>
#include <iterator>
#include "Header.h"
#include "helper_cuda.h"
#include "timeutility.h"
#include <device_launch_parameters.h>


//
//struct Intersector{
//	typedef thrust::device_vector<thrust::device_vector<int>> ADJLIST;
//	Intersector(ADJLIST& para, int pi) :adjl(para), i(pi){}
//	__device__
//	int operator()(int j){
//		const thrust::device_vector<int>& idev = adjl[i];
//		const thrust::device_vector<int>& jdev = adjl[j];
//		int ret = intersect(idev, jdev);
//		return ret;
//	}
//	typedef thrust::device_vector<int> DEVARRAY;
//	int intersect(const DEVARRAY& i, const DEVARRAY& j)
//	{
//		int ret = 0;
//		for (DEVARRAY::iterator iteri = i.begin(), iterj = j.begin(); iteri != i.end() && iterj != j.end();){
//			if (*iteri == *iterj){
//				ret++;
//				iteri++;
//				iterj++;
//			}
//			else if (*iteri < *iterj){
//				iteri++;
//			}
//			else{
//				iterj++;
//			}
//		}
//		return ret;
//	}
//	ADJLIST& adjl;
//	int i;
//};
//
//struct Op{
//	typedef thrust::device_vector<thrust::device_vector<int>> ADJLIST;
//	Op(ADJLIST& para) :adjl(para){}
//	__device__
//	int operator()(int i){
//		const thrust::device_vector<int>& dref = adjl[i];
//		thrust::device_vector<int> dev(dref.size());
//		thrust::transform(dref.begin(), dref.end(), dev.begin(), Intersector(adjl, i));
//		int ret = thrust::reduce(dev.begin(), dev.end());
//		return ret;
//	}
//	ADJLIST& adjl;
//};
//
//int c_forward(AdjList& adjl)
//{
//	int ret = 0;
//	thrust::device_vector<int> dev(adjl.node_num);
//	thrust::sequence(dev.begin(), dev.end());
//	thrust::device_vector<thrust::device_vector<int>> dev_adjl(adjl.adj_list);
//	//thrust::transform(dev.begin(), dev.end(), dev.begin(), Op(dev_adjl));
//	ret = thrust::reduce(dev.begin(), dev.end());
//	return ret;
//}
//
//int cuda_forward(std::ifstream& ifs)
//{
//	AdjList adjl;
//	{
//		OriEdgeList oel;
//		read_edges(ifs, oel);
//		make_adj_list_cuda(oel, adjl);
//	}
//	int ret = c_forward(adjl);
//	return ret;
//}




void copyEdgeListToDev(OriEdgeList& oel, int*& dev)
{
	cudaStream_t stream0;
	checkCudaErrors(cudaStreamCreate(&stream0));
	checkCudaErrors(cudaMalloc((void**)&dev, oel.edge_num * 2 * sizeof(int) ));
	checkCudaErrors(cudaMemcpyAsync(dev, oel.edges.data(), oel.edge_num * 2 * sizeof(int), cudaMemcpyHostToDevice, stream0));
	checkCudaErrors(cudaDeviceSynchronize());
}

void freeEdgeListFromDev(int*& dev)
{
	checkCudaErrors(cudaFree(dev));
}

void copyAdjListToDev(AdjList& adjl, int*& ptr)
{
	cudaStream_t stream0;
	checkCudaErrors(cudaStreamCreate(&stream0));
	checkCudaErrors(cudaMalloc((void**)&ptr, adjl.adj_list.size() * sizeof(int*) ));
	std::vector<int*> dev_ptr(adjl.adj_list.size());
	for (size_t i = 0; i < adjl.adj_list.size(); i++){
		checkCudaErrors(cudaMalloc((void**)&dev_ptr[i], sizeof(int) * adjl.adj_list[i].size()));
		checkCudaErrors(cudaMemcpyAsync(dev_ptr[i], adjl.adj_list[i].data(), sizeof(int) * adjl.adj_list[i].size(), cudaMemcpyHostToDevice, stream0));
	}
	checkCudaErrors(cudaMemcpyAsync(ptr, dev_ptr.data(), sizeof(int*) * adjl.adj_list.size(), cudaMemcpyHostToDevice, stream0));
	checkCudaErrors(cudaDeviceSynchronize());
}

int cuda_reverse_edge_foward(AdjList& adjl)
{
	int ret = 0;
	return ret;
}

typedef std::vector<int>::iterator Iter;
int intersect_iter(Iter n1beg, Iter n1end, Iter n2beg, Iter n2end)
{
	int ret = 0;
	while (n1beg != n1end && n2beg != n2end){
		if (*n1beg == *n2beg){
			ret++;
			n1beg++;
			n2beg++;
		}
		else if (*n1beg < *n2beg){
			n1beg++;
		}
		else {
			n2beg++;
		}
	}
	return ret;
}

int forward_no_explicit_edge_list(std::vector<int>& index, std::vector<int>& edge_head)
{
	int ret = 0;
	Iter beg = edge_head.begin();
	for (size_t i = 0; i < index.size() - 1; i++){
		for (Iter eh = beg + index[i]; eh != beg + index[i + 1]; eh++){
			int h = *eh;
			ret += intersect_iter(beg + index[i], beg + index[i + 1], beg + index[h], beg + index[h + 1]);
		}
	}
	return ret;
}

int forward_with_edge_list(std::vector<int>& index, std::vector<int>& edge_head, std::vector<int>& tail)
{
	int ret = 0;
	for (size_t i = 0; i < edge_head.size(); i++){
		int node1 = tail[i];
		int node2 = edge_head[i];
		Iter beg = edge_head.begin();
		ret += intersect_iter(beg + index[node1], beg + index[node1 + 1], beg + index[node2], beg + index[node2 + 1]);
	}
	return ret;
}

inline int do_serial_cuda_forward(std::vector<int>& edge_tail_start_index, std::vector<int>& edge_head, std::vector<int>& tail)
{
	CpuTime ct;
	ct.startTimer();
	int ret = forward_no_explicit_edge_list(edge_tail_start_index, edge_head);//cuda_reverse_edge_foward(adjl);
	//int ret = r_e_l_forward(edge_tail_start_index, edge_head, tail);
	ct.stopAndPrint("!!!reverse_edge_list_forward core!!!");
	return ret;
}

__device__ int insect(int* beg1, int* end1, int* beg2, int* end2)
{
	int ret = 0;
	while (beg1 != end1 && beg2 != end2){
		if (*beg1 == *beg2){
			ret++;
			beg1++;
			beg2++;
		}
		else if (*beg1 < *beg2){
			beg1++;
		}
		else{
			beg2++;
		}
	}
	return ret;
}

__global__ void kernel_counting(int* index, size_t index_len, int* head, size_t head_len, int* tail, int* result)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int step = gridDim.x * blockDim.x;
	int h, t;
	for (; idx < head_len; idx += step){
		h = head[idx];
		t = tail[idx];
		result[idx] += insect(head + index[h], head + index[h + 1], head + index[t], head + index[t + 1]);
	}
}

int cuda_count_triangle(int* dev_index, size_t index_len, int* dev_head, size_t head_len, int* dev_tail, int* dev_result)
{
	int ret = 0;
	int thread = 256;
	int block = (head_len - 1) / thread + 1;
	CpuTime ct;
	ct.startTimer();
	kernel_counting<<<block, thread>>>(dev_index, index_len, dev_head, head_len, dev_tail, dev_result);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	ret = thrust::reduce(thrust::device_ptr<int>(dev_result), thrust::device_ptr<int>(dev_result + head_len));
	ct.stopAndPrint("!!!cuda_count_triangle!!!");
	return ret;
}

int para_cuda_forward(std::vector<int>& index, std::vector<int>& head, std::vector<int>& tail)
{
	int ret = 0;
	int* dev_index;
	int* dev_head;
	int* dev_tail;
	int* result;

	cudaStream_t stream1;
	cudaStream_t stream2;
	CpuTime ct;
	ct.startTimer();
	checkCudaErrors(cudaStreamCreate(&stream1));
	checkCudaErrors(cudaStreamCreate(&stream2));
	checkCudaErrors(cudaMalloc((void**)&result, sizeof(int) * (head.size()) ));
	checkCudaErrors(cudaMemsetAsync(result, 0, sizeof(int) * (head.size()), stream1));

	checkCudaErrors(cudaMalloc((void**)&dev_index, sizeof(int) * index.size()));
	checkCudaErrors(cudaMemcpyAsync(dev_index, index.data(), sizeof(int) * index.size(), cudaMemcpyHostToDevice, stream1));
	checkCudaErrors(cudaMalloc((void**)&dev_head, sizeof(int) * head.size()));
	checkCudaErrors(cudaMemcpyAsync(dev_head, head.data(), sizeof(int) * head.size(), cudaMemcpyHostToDevice, stream2));
	checkCudaErrors(cudaMalloc((void**)&dev_tail, sizeof(int) * tail.size()));
	checkCudaErrors(cudaMemcpyAsync(dev_tail, tail.data(), sizeof(int) * tail.size(), cudaMemcpyHostToDevice, stream2));

	checkCudaErrors(cudaDeviceSynchronize());
	ct.stopAndPrint("@@@copy to device done");

	ret = cuda_count_triangle(dev_index, index.size(), dev_head, head.size(), dev_tail, result);
	return ret;
}

void preprocess(std::ifstream& ifs, std::vector<int>& edge_tail_start_index, std::vector<int>& edge_head)
{
	AdjList adjl;
	{
		OriEdgeList oel;
		read_edges(ifs, oel);
		make_adj_list_cuda(oel, adjl);
		//copyEdgeListToDev(oel, dev_edge_list);
		//copyAdjListToDev(adjl, dev_adj_list);//4.39s in kernel 7.969s in userspace for amazon ungraph, it's too slow
		//std::cout << "on device " << dev[0] << " " << dev[1] << std::endl;
		//std::cout << "on host " << oel.edges[0].first << " " << oel.edges[0].second << std::endl;
		//thrust::device_vector<thrust::device_vector<int>> dev_vec(adjl.adj_list.size());
		//for (size_t i = 0; i < adjl.adj_list.size(); i++){
		//	dev_vec[i] = adjl.adj_list[i];
		//}
		//std::cout << "dev_vec size: " << dev_vec.size() << std::endl;
	}
	int edge_count = 0;
	edge_tail_start_index.resize(adjl.adj_list.size() + 1);
	edge_head.reserve(adjl.edge_num);
	for (size_t i = 0; i < adjl.adj_list.size(); i++){
		edge_tail_start_index[i] = edge_count;
		edge_count += adjl.adj_list[i].size();
		std::copy(adjl.adj_list[i].begin(), adjl.adj_list[i].end(), std::back_inserter(edge_head));
	}
	edge_tail_start_index.back() = edge_count;
}

int cuda_forward(std::ifstream& ifs)
{
	
	std::vector<int> edge_tail_start_index;
	std::vector<int> edge_head;
	preprocess(ifs, edge_tail_start_index, edge_head);

	std::vector<int> tail;
	tail.resize(edge_head.size());
	for (size_t i = 0; i < edge_tail_start_index.size() - 1; i++){
		std::fill(tail.begin() + edge_tail_start_index[i], tail.begin() + edge_tail_start_index[i + 1], i);
	}

	int ret = para_cuda_forward(edge_tail_start_index, edge_head, tail);

	checkCudaErrors(cudaDeviceReset());
	return ret;
}

int serial_cuda_forward_without_tail(std::ifstream& ifs)
{
	std::vector<int> edge_tail_start_index;
	std::vector<int> edge_head;
	preprocess(ifs, edge_tail_start_index, edge_head);

	CpuTime ct;
	ct.startTimer();
	int ret = forward_no_explicit_edge_list(edge_tail_start_index, edge_head);
	ct.stopAndPrint("!!!serial cuda forward without tail!!!");
	return ret;
}

int serial_cuda_forward(std::ifstream& ifs)
{
	std::vector<int> edge_tail_start_index;
	std::vector<int> edge_head;
	preprocess(ifs, edge_tail_start_index, edge_head);

	std::vector<int> tail;
	tail.resize(edge_head.size());
	for (size_t i = 0; i < edge_tail_start_index.size() - 1; i++){
		std::fill(tail.begin() + edge_tail_start_index[i], tail.begin() + edge_tail_start_index[i + 1], i);
	}
	CpuTime ct;
	ct.startTimer();
	int ret = forward_with_edge_list(edge_tail_start_index, edge_head, tail);
	ct.stopAndPrint("!!!serial cuda forward with tail!!!");
	return ret;
}
//void cuda_forward(std::string& file, int result[], int item)
//{
//	std::ifstream ifs(file);
//	if (!ifs){
//		std::cerr << "error open file" << std::endl;
//		exit(1);
//	}
//	OriEdgeList oel;
//	read_edges(ifs, oel);
//	thrust::device_vector<std::pair<int, int>> dev_vec(oel.edges.begin(), oel.edges.end());
//}
