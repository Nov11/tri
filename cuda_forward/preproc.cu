#include <thrust/sequence.h>
#include <thrust/transform.h>
#include <thrust/adjacent_difference.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/random.h>
#include <thrust/inner_product.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include "Header.h"
#include "helper_cuda.h"
#include "timeutility.h"

//
//void preprocessOnCuda(std::ifstream& ifs)
//{
//	OriEdgeList oel;
//	read_edges(ifs, oel);
//	int* dev_edges;
//	int* dev_e_tmp;
//	size_t edge_size = sizeof(int) * oel.edge_num * 2;
//	checkCudaErrors(cudaMalloc((void**)&dev_edges, edge_size));
//	checkCudaErrors(cudaMalloc((void**)&dev_e_tmp, edge_size));
//	checkCudaErrors(cudaMemcpy(dev_edges, oel.edges.data(), edge_size, cudaMemcpyHostToDevice));
//	checkCudaErrors(cudaMemcpy(dev_e_tmp, dev_edges, edge_size, cudaMemcpyDeviceToDevice));
//	std::cout << "copied edges to dev" << std::endl;
//
//	thrust::device_ptr<int> dptr_beg(dev_e_tmp);
//	thrust::device_ptr<int> dptr_end(dev_e_tmp + oel.edge_num * 2);
//	thrust::sort(dptr_beg, dptr_end);
//
//	//int num = thrust::inner_product(dptr_beg, dptr_end - 1, dptr_beg + 1, 1, thrust::plus<int>(), thrust::not_equal_to<int>());
//	//std::cout << "num of nodes" << num << std::endl;
//	
//	//anonying warning about sort
//	//int* tmp;
//	//checkCudaErrors(cudaMalloc((void**)&tmp, sizeof(int) * 2));
//	//thrust::sort(thrust::device_ptr<int>(tmp), thrust::device_ptr<int>(tmp) + 2);
//	
//	
//	thrust::device_vector<int> d_keys(oel.node_num);
//	thrust::device_vector<int> d_degree(oel.node_num);
//	thrust::reduce_by_key(dptr_beg, dptr_end, thrust::constant_iterator<int>(1), d_keys.begin(), d_degree.begin());
//	thrust::sort_by_key(d_degree.begin(), d_degree.end(), d_keys.begin(), thrust::greater<int>());
//	checkCudaErrors(cudaFree(dev_e_tmp));
//	thrust::fill(d_degree.begin(), d_degree.end(), thrust::counting_iterator<int>(0));
//	//thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple()))
//}