#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust\reduce.h>
#include <thrust\device_ptr.h>

#include "Header.h"
#include "helper_cuda.h"
#include "timeutility.h"




__device__ int insect(int* beg1, int* end1, int* beg2, int* end2)
{
	int ret = 0;
	int node1 = *beg1;
	int node2 = *beg2;
	while (beg1 != end1 && beg2 != end2){
		//if (node1 == node2){
		//	ret++;
		//}
		//if (node1 <= node2){
		//	node1 = *++beg1;
		//}
		//if(node1 >= node2){
		//	node2 = *++beg2;
		//}

		//vari
		//if (node1 == node2){
		//	ret++;
		//	node1 = (*++beg1);
		//	node2 = (*++beg2);
		//}
		//else if (node1 < node2){
		//	node1 = (*++beg1);
		//}
		//else{
		//	node2 = (*++beg2);
		//}

		if (node1 < node2){
			node1 = *++beg1;
		}
		else if (node1 > node2){
			node2 = *++beg2;
		}
		else{
			node1 = *++beg1;
			node2 = *++beg2;
			ret++;
		}
	}
	return ret;
}

__device__ int inse(int* less, size_t llen, int* more, size_t mlen)
{
	int ret = 0;

	for (size_t i = 0; i < llen; i++){
		int beg = 0;
		int end = mlen;
		while (beg < end){
			if (more[beg + (end - beg) / 2] == less[i]){
				ret++;
				break;
			}
			else if (more[beg + (end - beg) / 2] < less[i]){
				beg = beg + (end - beg) / 2 + 1;
			}
			else {
				end = beg + (end - beg) / 2;
			}
		}
	}
	return ret;
}

__global__ void kernel_counting(int* index, int* head, size_t head_len, int* tail, int* result, int block_idx_offset = 0)
{
	int idx = blockDim.x * (blockIdx.x + block_idx_offset) + threadIdx.x;
	//int step = gridDim.x * blockDim.x;
	int h, t;
	//for (; idx < head_len; idx += step){
	if (idx >= head_len)return;
	h = head[idx];
	t = tail[idx];
	//ori
	//result[idx] += insect(head + index[h], head + index[h + 1], head + index[t], head + index[t + 1]);//0.199s
	//faster
	result[idx] = insect(head + index[h], head + index[h + 1], head + index[t], head + idx);//0.160s
	//}
	//if (index[h + 1] - index[h] > idx - index[t]){// 2 branch ->0.237s
	//	result[idx] = inse(head + index[t], idx - index[t], head + index[h], index[h + 1] - index[h]);//0.319s on asitter
	//}
	//else{
	//result[idx] = inse(head + index[h], index[h + 1] - index[h], head + index[t], idx - index[t]);//0.189s on asitter
	//}

}

void multikernel(int* dev_index, size_t index_len, int* dev_head, size_t head_len, int* dev_tail, int* dev_result, int block, int thread)
{
	const int kernel_per_stream = 60;
	const int max_stream = 16;
	cudaStream_t stream[max_stream];//max concurrent kernl in compute capability 3.0
	for (size_t i = 0; i < max_stream; i++){
		checkCudaErrors(cudaStreamCreate(&stream[i]));
	}
	block = (block - 1) / max_stream + 1;
	//add
	block = (block - 1) / kernel_per_stream + 1;

	//checkCudaErrors(cudaFuncSetCacheConfig(kernel_counting, cudaFuncCachePreferL1));

	//for (size_t i = 0; i < max_stream; i++){
	//	kernel_counting<<<block, thread, 0, stream[i]>>>(dev_index, dev_head, head_len, dev_tail, dev_result, i * block);
	//}


	//for (size_t i = 0; i < max_stream; i++){
	//	for (size_t j = 0; j < kernel_per_stream; j++){
	//		kernel_counting << <block, thread, 0, stream[i] >> >(dev_index, dev_head, head_len, dev_tail, dev_result, (i * kernel_per_stream + j) * block);
	//	}
	//	//checkCudaErrors(cudaGetLastError());
	//}
	//
	for (size_t j = 0; j < kernel_per_stream; j++){

		for (size_t i = 0; i < max_stream; i++){
			kernel_counting << <block, thread, 0, stream[i] >> >(dev_index, dev_head, head_len, dev_tail, dev_result, (i * kernel_per_stream + j) * block);
		}
		//checkCudaErrors(cudaGetLastError());
	}
	checkCudaErrors(cudaDeviceSynchronize());
	for (size_t i = 0; i < max_stream; i++){
		checkCudaErrors(cudaStreamDestroy(stream[i]));
	}
}

int cuda_count_triangle(int* dev_index, size_t index_len, int* dev_head, size_t head_len, int* dev_tail, int* dev_result)
{
	int ret = 0;
	int thread = 256;//;256
	int block = (head_len - 1) / thread + 1;

	CpuTime ct;
	ct.startTimer();
	cudaEvent_t start, stop;
	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));
	checkCudaErrors(cudaEventRecord(start, 0));
	checkCudaErrors(cudaFuncSetCacheConfig(kernel_counting, cudaFuncCachePreferL1));
	kernel_counting << <block, thread >> >(dev_index, dev_head, head_len, dev_tail, dev_result);
	//multikernel(dev_index, index_len, dev_head, head_len, dev_tail, dev_result, block, thread);

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaEventRecord(stop, 0));
	checkCudaErrors(cudaEventSynchronize(stop));
	float time = 0;
	checkCudaErrors(cudaEventElapsedTime(&time, start, stop));
	std::cout << "start to reduce" << std::endl;
	ret = thrust::reduce(thrust::device_ptr<int>(dev_result), thrust::device_ptr<int>(dev_result + head_len));
	std::cout << "time profiled by cudaEvent: " << time / 1000 << "s" << std::endl;
	ct.stopAndPrint("!!!cuda_count_triangle!!!");
	checkCudaErrors(cudaEventDestroy(start));
	checkCudaErrors(cudaEventDestroy(stop));

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
	//cudaStream_t stream2;
	CpuTime ct;
	ct.startTimer();
	checkCudaErrors(cudaStreamCreate(&stream1));
	//checkCudaErrors(cudaStreamCreate(&stream2));
	checkCudaErrors(cudaMalloc((void**)&result, sizeof(int) * (head.size())));
	checkCudaErrors(cudaMemsetAsync(result, 0, sizeof(int) * (head.size()), stream1));

	checkCudaErrors(cudaMalloc((void**)&dev_index, sizeof(int) * index.size()));
	checkCudaErrors(cudaMemcpyAsync(dev_index, index.data(), sizeof(int) * index.size(), cudaMemcpyHostToDevice, stream1));
	checkCudaErrors(cudaMalloc((void**)&dev_head, sizeof(int) * head.size()));
	checkCudaErrors(cudaMemcpyAsync(dev_head, head.data(), sizeof(int) * head.size(), cudaMemcpyHostToDevice, stream1));
	checkCudaErrors(cudaMalloc((void**)&dev_tail, sizeof(int) * tail.size()));
	checkCudaErrors(cudaMemcpyAsync(dev_tail, tail.data(), sizeof(int) * tail.size(), cudaMemcpyHostToDevice, stream1));
	checkCudaErrors(cudaStreamDestroy(stream1));
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
		CpuTime ct;
		ct.startTimer();
		read_edges(ifs, oel);
		ct.stopAndPrint("read file");
		ct.startTimer();
		make_adj_list_cuda(oel, adjl);
		ct.stopAndPrint("build adj list");
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
	CpuTime ct;
	ct.startTimer();
	int ret = 0;
	std::vector<int> edge_tail_start_index;
	std::vector<int> edge_head;
	preprocess(ifs, edge_tail_start_index, edge_head);

	std::vector<int> tail;
	tail.resize(edge_head.size());
	for (size_t i = 0; i < edge_tail_start_index.size() - 1; i++){
		std::fill(tail.begin() + edge_tail_start_index[i], tail.begin() + edge_tail_start_index[i + 1], i);
	}

	//preprocessOnCuda(ifs);


	ct.stopAndPrint("time to preproc");
	ret = para_cuda_forward(edge_tail_start_index, edge_head, tail);

	checkCudaErrors(cudaDeviceReset());
	return ret;
}



