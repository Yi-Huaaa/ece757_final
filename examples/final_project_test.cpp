#include "itap.hpp"

int main(int argc, char *argv[]) {

  if(argc != 3) {
    std::cerr << "usage: ./example/final_project_test circuit_file matrix_size\n";
    std::cerr << "matrix size means the size of matrix multiplication within each task in the graph\n";
    std::exit(EXIT_FAILURE);
  }

  std::string filename = argv[1]; 
  int matrix_size = std::atoi(argv[2]);

  itap::iTAP partitioner(filename);

  partitioner.set_partition_size(partitioner.num_nodes());

  std::cout << "task graph size = " << partitioner.num_nodes() << "\n";
  std::cout << "runnning gpu partitioning...\n";
  partitioner.partition();

  if(partitioner.is_partition_valid() == false) {
    std::cerr << "cycle introduced...\n";
    std::exit(EXIT_FAILURE);
  }

  // dump the partitioned graph
  // std::cout << "dumped graph, node_name[partition], check it on GraphvizOnline.\n";
  // partitioner.dump_graph();
  
  // run_graph: show the runtime of simulation of taskflow and itap
  partitioner.run_graph(matrix_size);
}
