#include "itap.hpp"

#define VERSION 1 // 0: G-PASTA, 1: _SA, 2: ML

int main(int argc, char *argv[]) {

  if(argc != 4) {
    std::cerr << "usage: ./example/final_project_test circuit_file matrix_size partition size\n";
    std::cerr << "matrix size means the size of matrix multiplication within each task in the graph\n";
    std::cerr << "partition size means we would like to partition the original graph into how many partition\n";
    std::exit(EXIT_FAILURE);
  }

  std::string filename = argv[1]; 
  int matrix_size = std::atoi(argv[2]);
  size_t partition_size = std::atoi(argv[3]);
  
  // Get object
  itap::iTAP partitioner(filename);

  partitioner.ACA2_partition(matrix_size, partition_size, VERSION);

  // Dump the partitioned graph
  // std::cout << "dumped graph, node_name[partition], check it on GraphvizOnline.\n";
  // partitioner.dump_graph();
  
  // run_graph: show the runtime of simulation of taskflow and itap
  partitioner.run_graph(matrix_size);
}
