
#include <cuda.h>
#include <vector>
#include <iostream>
#include "itap.hpp"

#define BLOCK_SIZE 512 

namespace itap {

void checkError_t(cudaError_t error, std::string msg) {
    if (error != cudaSuccess) {
        printf("%s: %d\n", msg.c_str(), error);
        std::exit(1);
    }
}

__global__ void partition_gpu(
  int* d_adjp, int* d_adjncy, int* d_adjncy_size, int* d_dep_size,
  int* d_topo_result_gpu,
  int* d_partition_result_gpu,
  int* d_partition_counter_gpu,
  int partition_size,
  int* max_partition_id,
  int read_offset, uint32_t read_size, // [read_offset, read_offset + read_size - 1] are all the frontiers 
  uint32_t* write_size,
  int* d_fu_partition // the future partition this node will be assigned to(if partition not full)
) {

  uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;

  if(tid < read_size) {
    int cur_id = d_topo_result_gpu[read_offset + tid]; // get current task id
    if(d_adjp[cur_id] == -1) {
      return; // if _adjp[cur_id] = -1, that means it has no fanout
    }
    for(int offset=d_adjp[cur_id]; offset<d_adjp[cur_id] + d_adjncy_size[cur_id]; offset++) {
      int neighbor_id = d_adjncy[offset];
      
      atomicMax(&d_fu_partition[neighbor_id], d_partition_result_gpu[cur_id]);

      if(atomicSub(&d_dep_size[neighbor_id], 1) == 1) {
        int position = atomicAdd(write_size, 1); // no need to atomic here...
        d_topo_result_gpu[read_offset + read_size + position] = neighbor_id;        
        int cur_partition = d_fu_partition[neighbor_id]; // get the largest partition id from the parent  
        if(atomicAdd(&d_partition_counter_gpu[cur_partition], 1) < partition_size) { 
          d_partition_result_gpu[neighbor_id] = cur_partition; // no need to atomic here cuz only one thread will access this neighbor here
        }
        else {
          int new_partition_id = atomicAdd(max_partition_id, 1) + 1; // now we have new partition when we find cur_partition is full
                                                                     // we need to store this new partition id locally to the thread
          d_partition_result_gpu[neighbor_id] = new_partition_id;
          d_partition_counter_gpu[new_partition_id]++;  
        }
      }
    }
  }
}



void iTAP::_partition_cuda() {

  _export_csr();   

  std::vector<int> source;
  for(const auto& node : _nodes) {
    if(node._fanins.size() == 0) {
      source.push_back(node._csr_id);
    }
  }

  int* d_adjp; 
  int* d_adjncy; 
  int* d_adjncy_size;
  int* d_dep_size;
  int* d_topo_result_gpu;
  int* d_partition_result_gpu;
  int* d_partition_counter_gpu;
  int read_offset = 0;
  uint32_t read_size = source.size();
  uint32_t* write_size;
  int* max_partition_id; // max_partition id we have currently, initially is source.size() - 1

  unsigned num_nodes = _adjp.size();
  unsigned num_edges = _adjncy.size();
  
  std::vector<int> fu_partition(num_nodes, -1);
  int* d_fu_partition;


  std::vector<int> topo_result_gpu(num_nodes);
  std::vector<int> partition_result_gpu(num_nodes, -1);
  int source_partition_id = 0;
  for(unsigned i=0; i<source.size(); i++) {
    partition_result_gpu[source[i]] = source_partition_id;
    source_partition_id++;
  }
  std::vector<int> partition_counter_gpu(num_nodes, 0);
  for(unsigned i=0; i<source.size(); i++) { // at the beginning, each source corresponds to one partition
    partition_counter_gpu[i]++;
  }

  checkError_t(cudaMalloc(&d_adjp, sizeof(int)*num_nodes), "d_adjp allocation failed");
  checkError_t(cudaMalloc(&d_adjncy, sizeof(int)*num_edges), "d_adjncy allocation failed");
  checkError_t(cudaMalloc(&d_adjncy_size, sizeof(int)*num_nodes), "d_adjncy_size allocation failed");
  checkError_t(cudaMalloc(&d_dep_size, sizeof(int)*num_nodes), "d_dep_size allocation failed");
  checkError_t(cudaMalloc(&d_topo_result_gpu, sizeof(int)*num_nodes), "d_topo_result_gpu allocation failed");
  checkError_t(cudaMalloc(&d_partition_result_gpu, sizeof(int)*num_nodes), "d_partition_result_gpu allocation failed");
  checkError_t(cudaMalloc(&d_partition_counter_gpu, sizeof(int)*num_nodes), "d_partition_counter_gpu allocation failed");
  checkError_t(cudaMalloc(&write_size, sizeof(uint32_t)), "write_size allocation failed");
  checkError_t(cudaMalloc(&max_partition_id, sizeof(int)), "max_partition_id allocation failed");
  checkError_t(cudaMalloc(&d_fu_partition, sizeof(int)*num_nodes), "d_fu_partition allocation failed");

  auto start = std::chrono::steady_clock::now();
  checkError_t(cudaMemcpy(d_adjp, _adjp.data(), sizeof(int)*num_nodes, cudaMemcpyHostToDevice), "d_adjp memcpy failed"); 
  checkError_t(cudaMemcpy(d_adjncy, _adjncy.data(), sizeof(int)*num_edges, cudaMemcpyHostToDevice), "d_adjncy memcpy failed"); 
  checkError_t(cudaMemcpy(d_adjncy_size, _adjncy_size.data(), sizeof(int)*num_nodes, cudaMemcpyHostToDevice), "d_adjncy_size memcpy failed"); 
  checkError_t(cudaMemcpy(d_dep_size, _dep_size.data(), sizeof(int)*num_nodes, cudaMemcpyHostToDevice), "d_dep_size memcpy failed"); 
  checkError_t(cudaMemcpy(d_topo_result_gpu, source.data(), sizeof(int)*source.size(), cudaMemcpyHostToDevice), "d_topo_result_gpu memcpy failed"); 
  checkError_t(cudaMemcpy(d_partition_result_gpu, partition_result_gpu.data(), sizeof(int)*num_nodes, cudaMemcpyHostToDevice), "d_partition_result_gpu memcpy failed"); 
  checkError_t(cudaMemcpy(d_partition_counter_gpu, partition_counter_gpu.data(), sizeof(int)*num_nodes, cudaMemcpyHostToDevice), "d_partition_counter_gpu memcpy failed"); 
  checkError_t(cudaMemset(write_size, 0, sizeof(uint32_t)), "write_size memset failed");
  int max_partition_id_cpu = source.size() - 1;
  checkError_t(cudaMemcpy(max_partition_id, &max_partition_id_cpu, sizeof(int), cudaMemcpyHostToDevice), "max_partition_id memcpy failed"); 
  checkError_t(cudaMemcpy(d_fu_partition, fu_partition.data(), sizeof(int)*num_nodes, cudaMemcpyHostToDevice), "d_fu_partition memcpy failed"); 


  // invoke kernel
  unsigned num_block;
  while(read_size > 0) { 
    num_block = (read_size + BLOCK_SIZE - 1) / BLOCK_SIZE;

    partition_gpu<<<num_block, BLOCK_SIZE>>>(
      d_adjp, d_adjncy, d_adjncy_size, d_dep_size,
      d_topo_result_gpu,
      d_partition_result_gpu,
      d_partition_counter_gpu,
      _partition_size,
      max_partition_id,
      read_offset, read_size, // [read_offset, read_offset + read_size - 1] are all the frontiers 
      write_size,
      d_fu_partition
    );

    // calculate where to read for next iteration
    read_offset += read_size;
    checkError_t(cudaMemcpy(&read_size, write_size, sizeof(uint32_t), cudaMemcpyDeviceToHost), "queue_size memcpy failed");

    // set write_size = 0 for next iteration 
    checkError_t(cudaMemset(write_size, 0, sizeof(uint32_t)), "write_size rewrite failed");
  }

  checkError_t(cudaMemcpy(partition_result_gpu.data(), d_partition_result_gpu, sizeof(int)*num_nodes, cudaMemcpyDeviceToHost), "_partition_result_gpu memcpy failed"); 
  auto end = std::chrono::steady_clock::now();
  partition_time += std::chrono::duration_cast<std::chrono::microseconds>(end-start).count();
  checkError_t(cudaMemcpy(&max_partition_id_cpu, max_partition_id, sizeof(int), cudaMemcpyDeviceToHost), "max_partition_id_cpu memcpy failed"); 
 
  // assign partition IDs to nodes according to partition_result_gpu
  size_t index = 0;
  for(auto& node : _nodes) {
    node._cluster_id = partition_result_gpu[index]; 
    // if(partition_result_gpu[index] == -1) {
    //   std::cerr << "partition_result_gpu wrong...\n";
    //   std::exit(EXIT_FAILURE);
    // }
    ++index;
  }

  // also assign dep_cnt to nodes cuz it is needed in incremental partition
  for(auto& node : _nodes) {
    node._dep_cnt = node._fanins.size();
  } 

  // reset the _cluster_cnt
  _cluster_cnt.resize(max_partition_id_cpu+1, 0); // to avoid reallocation
  for(const auto& node : _nodes) {
    ++_cluster_cnt[node._cluster_id];
  }

  _max_cluster_id = max_partition_id_cpu;

  checkError_t(cudaFree(d_adjp), "d_adjp free failed");
  checkError_t(cudaFree(d_adjncy), "d_adjncy free failed");
  checkError_t(cudaFree(d_adjncy_size), "d_adjncy_size free failed");
  checkError_t(cudaFree(d_dep_size), "d_dep_size free failed");
  checkError_t(cudaFree(d_topo_result_gpu), "d_topo_result_gpu free failed");
  checkError_t(cudaFree(d_partition_result_gpu), "d_partition_result_gpu free failed");
  checkError_t(cudaFree(d_partition_counter_gpu), "d_partition_counter_gpu free failed");
  checkError_t(cudaFree(write_size), "write_size free failed");
  checkError_t(cudaFree(max_partition_id), "max_partition_id free failed");
  checkError_t(cudaFree(d_fu_partition), "fu_partition free failed");
 
}

}  // namespace itap

