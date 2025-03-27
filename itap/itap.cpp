#include "itap.hpp"
#include <cstdlib> 
#include <ctime>
#include <cmath> 
#include <vector>
#include <algorithm>
#include <unordered_map>

namespace itap{

/**
@brief: construct a graph according to a circuit.txt  
*/
iTAP::iTAP(const std::string& filename) {

  std::ifstream infile(filename);
  if (!infile) {
    std::cerr << "Error opening file." << std::endl;
    std::exit(1);
  } 

  int num_nodes;
  infile >> num_nodes; // Read the number of nodes

  std::unordered_map<std::string, Node*> nodes;

  // Read node names and add them to the graph
  std::string node_name;
  for (int i = 0; i < num_nodes; ++i) {
    infile >> node_name;
    // Remove quotes from node name
    node_name = node_name.substr(1, node_name.size() - 3);
    nodes[node_name] = insert_node(node_name);
  }

  // Read edges and add them to the graph
  std::string from, to, arrow;
  while (infile >> from >> arrow >> to) {
    from = from.substr(1, from.size() - 2);
    to = to.substr(1, to.size() - 3);
    insert_edge(nodes[from], nodes[to]);
  } 
}

/**
@brief: insert a new node
*/
Node* iTAP::insert_node(const std::string& name) {

  // Node node(name);
  Node* node_ptr = &(_nodes.emplace_back(name)); 
  node_ptr->_node_satellite = --_nodes.end();

  // add this newly inserted node into _candidates
  _candidates.push_back(node_ptr);
  node_ptr->_is_candidate = true;

  return node_ptr;  
}

/**
@brief: remove a node
*/
void iTAP::remove_node(Node* node) {

  if(_partitioned) {
    if(node->_cluster_id == -1) {
      std::cerr << "remove new nodes...\n";
      std::exit(EXIT_FAILURE);
    }
    if(_cluster_cnt[node->_cluster_id] == 0) {
      std::cerr << "someting wrong with _cluster_cnt...\n";
      std::cerr << "current _cluster_cnt = [";
      for(const auto& c : _cluster_cnt) {
        std::cerr << c << " "; 
      }
      std::cerr << "]\n";
      std::cerr << "node: " << node->_name << ", cluster_id: " << node->_cluster_id << "\n";
      // dump_graph();
      std::exit(EXIT_FAILURE);
    }
    --_cluster_cnt[node->_cluster_id];
  }

  // remove its fanin/fanout edges from _edges
  // remove_edge will erase this edge from node->_fanins/fanouts, so no need to pop_front()
  while(!node->_fanins.empty()) {
    Edge* from = node->_fanins.front();
    remove_edge(from);
  }
  while(!node->_fanouts.empty()) {
    Edge* to = node->_fanouts.front();
    remove_edge(to);
  }
  _nodes.erase(node->_node_satellite);
}

/**
@brief: insert a new edge 
*/
Edge* iTAP::insert_edge(Node* from, Node* to) {

  // check if this is an edge inserted between partitioned nodes
  if(from->_partitioned && to->_partitioned) {
    if(!from->_is_candidate) { 
      // from may be put into candidate before
      _candidates.push_back(from);
      from->_is_candidate = true;
      _frontiers.push_back(from);
      from->_is_frontier = true;
    }
  }

  // update dep_cnt if there is full/incremental partition finished
  if(_partitioned) {
    if(from->_partitioned && to->_partitioned) {
      // if this edge is added between existing nodes, _dep_cnt needs to be added here
      // since we won't do it in incremental partition
      ++to->_dep_cnt; 
    }
  }

  // Edge edge;
  Edge* edge_ptr = &_edges.emplace_back(); 

  edge_ptr->_from = from;
  edge_ptr->_to = to;

  from->_fanouts.push_back(edge_ptr);
  to->_fanins.push_back(edge_ptr);
  
  // tells the index of this edge in _fanouts of from nodes and _fanins of to nodes
  // for remove_edge()
  edge_ptr->_from_satellite = --from->_fanouts.end();
  edge_ptr->_to_satellite = --to->_fanins.end();

  // tells the index of this edge in _fanouts of from nodes and _fanins of to nodes
  // but make the index as a pair with nodes, for remove_node()
  from->_fanout_satellites.push_back(std::make_pair(to, --to->_fanins.end()));
  to->_fanin_satellites.push_back(std::make_pair(from, --from->_fanouts.end()));

  edge_ptr->_satellite = --_edges.end();

  return edge_ptr;
}

/**
@brief: remove an edge 
*/
void iTAP::remove_edge(Edge* edge) {

  Node* from = edge->_from;
  Node* to = edge->_to;

  // update dep_cnt if there is full/incremental partition finished
  if(_partitioned) {
    --to->_dep_cnt;
  }

  // remove edge from _fanouts of from node
  // also remove edge from _fanout_satellites of from node
  // this edge should be in the same index as the edge in _fanouts
  // because they are always inserted and removed at the same time
  auto index = std::distance(from->_fanouts.begin(), edge->_from_satellite);
  auto it_satellite = from->_fanout_satellites.begin();
  std::advance(it_satellite, index);
  from->_fanouts.erase(edge->_from_satellite);
  from->_fanout_satellites.erase(it_satellite);


  // same method applied to to node
  index = std::distance(to->_fanins.begin(), edge->_to_satellite);
  it_satellite = to->_fanin_satellites.begin();
  std::advance(it_satellite, index);
  to->_fanins.erase(edge->_to_satellite);
  to->_fanin_satellites.erase(it_satellite);

  _edges.erase(edge->_satellite);
}

/**
@brief: set partition size 
*/
void iTAP::set_partition_size(const size_t partition_size) {

  _partition_size = partition_size;
}


void iTAP::partition(bool incremental, bool only_handle_edge, const int matrix_size) {
  // yhc: remove the incremental code
  assert(incremental == false && "incremental == false");
  assert(only_handle_edge == false && "only_handle_edge == false");
  
  // printf("bibi, matrix_size = %d\n", matrix_size);
  
  // reset the whole partition
  for(auto& n : _nodes) {
    n._cluster_id = -1;
    n._dep_cnt.store(0);
    // std::cout << "n._name: " << n._name << "\n";
  }
  
  _cluster_cnt.clear(); // clear the cluster cnt in last iteration

  _partition_cuda();

  // reset everything after each partition iteration
  // does not reset dep_cnt cuz incremental partition needs it
  // only reset dep_cnt and cluster_id right before full partition
  for(auto& node : _nodes) {
    node._is_candidate = false;
    node._partitioned = true;
    node._type1 = false;
    node._type2 = false;
    node._type3 = false;
    node._desired_cluster_id = -1;
  }
  _candidates.clear();
  _frontiers.clear();

  _partitioned = true;
}

/**
@brief: cycle-free clustering algorithm 
assign cluster id to node by selecting the largest cluster id from its dependents 
*/
void iTAP::_assign_cluster_id(Node* node_ptr, std::vector<std::atomic<size_t>>& cluster_cnt, std::atomic<int>& max_cluster_id) {
  int desired_cluster_id = node_ptr->_cluster_id; // cluster_id is initialized as -1(excluding source tasks)
  for(auto edge_ptr : node_ptr->_fanins) {
    Node* dep_ptr = edge_ptr->_from; // dependent of node_ptr
    if(dep_ptr->_cluster_id > desired_cluster_id) {
      desired_cluster_id = dep_ptr->_cluster_id;
    }
  }
  if(cluster_cnt[desired_cluster_id].fetch_add(1, std::memory_order_relaxed) < _partition_size) { 
    node_ptr->_cluster_id = desired_cluster_id;
  }
  else {
    int new_cluster_id = max_cluster_id.fetch_add(1, std::memory_order_relaxed) + 1;
    node_ptr->_cluster_id = new_cluster_id;
    cluster_cnt[new_cluster_id]++;
  }
}

/**
@brief: build the partitioned graph by constructing CNode and CEdge 
*/
void iTAP::build_cluster_graph() {

  // clear the original graph
  _cnodes.clear(); 
  _cedges.clear();

  if(_max_cluster_id < 0) {
    std::cerr << "_max_cluster_id is wrong...\n";
    std::exit(EXIT_FAILURE);
  }
  size_t num_clusters = _max_cluster_id + 1;
  // printf("num_clusters = %lu\n", num_clusters);

  // use a 2-D vector to record clusters (cuz it supports constant time random access)
  std::vector<std::vector<Node*>> clusters(num_clusters);
  for(auto& n : _nodes) {
    int cluster = n._cluster_id;
    // std::cout << "name: " << n._name << " cluster: " << cluster << "\n";
    clusters[cluster].push_back(&n);  
  }

  // check results
  // std::cout << "clusters:\n";
  // for(auto c : clusters) {
  //   std::cout << "[";
  //   for(auto n : c) {
  //     std::cout << n->_name << " ";
  //   }
  //   std::cout << "]" <<"\n";
  // }

  // construct CNode
  for(size_t i=0; i<num_clusters; i++) {
    CNode cnode;
    CNode* cnode_ptr = &(_cnodes.emplace_back(cnode));
    cnode_ptr->_satellite = --_cnodes.end();
    for(auto node_ptr : clusters[i]) {
      cnode_ptr->_nodes.emplace_back(node_ptr);
      node_ptr->_cnode_satellite = --_cnodes.end();
      node_ptr->_cnode = cnode_ptr;
    }
  }

  // construct CEdge 
  // e.g., to find fanouts, 
  // 1. traverse nodes within a cluster node
  // 2. for each node, find their fanout nodes
  // 3. find the cluster nodes to which these fanout nodes belong 
  // 4. add edge
  // Note. Redundent edges will be added.
  size_t itr = 0; // to iterate clusters
  for(auto& cnode : _cnodes) {
    // add fanouts  
    for(auto node_ptr : clusters[itr]) {
      for(auto node_fanouts : node_ptr->_fanouts) {
        Node* successor_ptr = node_fanouts->_to;
        // if this node is already in the cluster, ignore it 
        if(successor_ptr->_cluster_id == node_ptr->_cluster_id) {
          continue;
        }
        CNode* to = successor_ptr->_cnode;
        CEdge cedge;
        CEdge* cedge_ptr = &(_cedges.emplace_back(cedge));
        cedge_ptr->_from = &cnode;
        cedge_ptr->_to = to;
        cedge_ptr->_from_node = node_ptr;
        cedge_ptr->_to_node = successor_ptr;
        cnode._fanouts.emplace_back(cedge_ptr);
        to->_fanins.emplace_back(cedge_ptr);
      }
    }

    // add fanins 
    for(auto node_ptr : clusters[itr]) {
      for(auto node_fanins : node_ptr->_fanins) {
        Node* dependent_ptr = node_fanins->_from;
        // if this node is already in the cluster, ignore it 
        if(dependent_ptr->_cluster_id == node_ptr->_cluster_id) {
          continue;
        }
        CNode* from = dependent_ptr->_cnode;
        CEdge cedge;
        CEdge* cedge_ptr = &(_cedges.emplace_back(cedge));
        cedge_ptr->_to = &cnode;
        cedge_ptr->_from = from;
        cedge_ptr->_to_node = node_ptr;
        cedge_ptr->_from_node = dependent_ptr;
        cnode._fanins.emplace_back(cedge_ptr);
        from->_fanouts.emplace_back(cedge_ptr);
      }
    }

    // remove the duplicates
    cnode._fanouts.unique();
    cnode._fanins.unique();

    ++itr;
  }
}

/**
@brief: reset partition by setting _cluster_id of each node as -1 
*/
void iTAP::reset_partition() {
  
  for(auto& n : _nodes) {
    n._cluster_id = -1;
  }

  _partitioned = false;
}

/**
@brief: dump 2 graphs, before and after partitioning, using taskflow 
*/
void iTAP::dump_graph() {
  
  // dump graph before partitioning
  tf::Taskflow taskflow;
  for(auto& node : _nodes) {
    node._task = taskflow.emplace([](){}).name(node._name + "[" + std::to_string(node._cluster_id) + "]");
    // node._task = taskflow.emplace([](){}).name(std::to_string(node._cluster_id));
    // node._task = taskflow.emplace([](){}).name(std::to_string(node._csr_id)+"("+std::to_string(node._fanouts.size())+")");
  } 
  for(auto& node : _nodes) {
    for(auto node_fanout : node._fanouts) {
      node._task.precede(node_fanout->_to->_task); 
    }
  }
  taskflow.dump(std::cout);
  taskflow.clear();

  std::cout << "_cluster_cnt = [";
  for(const auto& c : _cluster_cnt) {
    std::cout << c << " "; 
  }
  std::cout << "]\n";

}

/**
@brief: build cluster graph using taskflow and run it. (used for runtime test)
*/
void iTAP::run_graph(size_t matrix_size) {

  for(const auto& node : _nodes) {
    if(node._cluster_id == -1) {
      std::cerr << "run graph: the graph is not fully partitioned...\n";
      std::exit(EXIT_FAILURE);
    }
  }

  build_cluster_graph();

  tf::Taskflow taskflow;
  tf::Executor executor;

  auto start = std::chrono::steady_clock::now();
  for(auto& node : _nodes) {
    node._task = taskflow.emplace([this, matrix_size]() {
      size_t N = matrix_size;
      size_t M = matrix_size;
      size_t K = matrix_size;
      std::vector<int> A(N*K, 1);
      std::vector<int> B(K*M, 2);
      std::vector<int> C(N*M);
      for(size_t n=0; n<N; n++) {
        for(size_t m=0; m<M; m++) {
          int temp = 0;
          for(size_t k=0; k<K; k++) {
            temp += A[n*K + k] * B[k*M + m];
          }
          C[n*M + m] = temp;
        }
      }
    }); 
  }
  for(auto& node : _nodes) {
    for(auto node_fanout : node._fanouts) {
      node._task.precede(node_fanout->_to->_task); 
    }
  }
  auto end = std::chrono::steady_clock::now();
  size_t origin_taskflow_buildtime = std::chrono::duration_cast<std::chrono::microseconds>(end-start).count();
  start = std::chrono::steady_clock::now();
  executor.run(taskflow).wait(); 
  end = std::chrono::steady_clock::now();
  size_t origin_taskflow_runtime = std::chrono::duration_cast<std::chrono::microseconds>(end-start).count();

  // check partitioned taskflow runtime 
  size_t partitioned_taskflow_buildtime = 0;
  taskflow.clear();
  for(auto& cnode : _cnodes) {
    if(cnode._nodes.size() == 0) {
      continue; // some cnodes are empty (caused by the way I create them)
    }
    // start = std::chrono::steady_clock::now();
    cnode._task = taskflow.emplace([&cnode, matrix_size](){
      for(size_t i=0; i<cnode._nodes.size(); i++) {
        size_t N = matrix_size;
        size_t M = matrix_size;
        size_t K = matrix_size;
        std::vector<int> A(N*K, 1);
        std::vector<int> B(K*M, 2);
        std::vector<int> C(N*M);
        for(size_t n=0; n<N; n++) {
          for(size_t m=0; m<M; m++) {
            int temp = 0;
            for(size_t k=0; k<K; k++) {
              temp += A[n*K + k] * B[k*M + m];
            }
            C[n*M + m] = temp;
          }
        }
      }

    }).name(cnode._get_name());
    // end = std::chrono::steady_clock::now();
    // partitioned_taskflow_buildtime += std::chrono::duration_cast<std::chrono::microseconds>(end-start).count();
  }
  std::vector<size_t> precede_time;
  for(auto& cnode : _cnodes) {
    if(cnode._nodes.size() == 0) {
      continue; // some cnodes are empty (caused by the way I create them)
    }
    start = std::chrono::steady_clock::now();
    for(auto cnode_fanout : cnode._fanouts) {
      cnode._task.precede(cnode_fanout->_to->_task); 
    }
    end = std::chrono::steady_clock::now();
    partitioned_taskflow_buildtime += std::chrono::duration_cast<std::chrono::microseconds>(end-start).count();
    precede_time.push_back(std::chrono::duration_cast<std::chrono::microseconds>(end-start).count());
  }
  start = std::chrono::steady_clock::now();
  executor.run(taskflow).wait(); 
  end = std::chrono::steady_clock::now();
  size_t partitioned_taskflow_runtime = std::chrono::duration_cast<std::chrono::microseconds>(end-start).count();

  // check sequential runtime
  taskflow.clear();
  taskflow.emplace([this, matrix_size] () {
    for(size_t i=0; i<_nodes.size(); i++) {
      size_t N = matrix_size;
      size_t M = matrix_size;
      size_t K = matrix_size;
      std::vector<int> A(N*K, 1);
      std::vector<int> B(K*M, 2);
      std::vector<int> C(N*M);
      for(size_t n=0; n<N; n++) {
        for(size_t m=0; m<M; m++) {
          int temp = 0;
          for(size_t k=0; k<K; k++) {
            temp += A[n*K + k] * B[k*M + m];
          }
          C[n*M + m] = temp;
        }
      }
    }
  });

  start = std::chrono::steady_clock::now();
  executor.run(taskflow).wait(); 
  end = std::chrono::steady_clock::now();
  size_t seq_runtime = std::chrono::duration_cast<std::chrono::microseconds>(end-start).count();

  size_t num_clusters = 0;
  for(const auto& cnode : _cnodes) {
    if(cnode._nodes.size() != 0) {
      ++num_clusters;
    }
  }
  size_t num_deps = 0;
  for(const auto& cnode : _cnodes) {
    if(cnode._nodes.size() != 0) {
      num_deps += cnode._fanouts.size();
    }
  }

  std::cout << "origin taskflow runtime: " << origin_taskflow_runtime << " us"
            << ", partitioned taskflow runtime: " << partitioned_taskflow_runtime << " us\n";

  // std::cout << "origin taskflow runtime: " << origin_taskflow_runtime << " us"
  //           << ", origin taskflow buildtime: " << origin_taskflow_buildtime << " us"
  //           << ", partitioned taskflow runtime: " << partitioned_taskflow_runtime << " us"
  //           << ", partitioned taskflow buildtime: " << partitioned_taskflow_buildtime << " us"
  //           << ", sequential runtime: " << seq_runtime << " us"
  //           // << " graph size: " << _nodes.size() 
  //           << ", partition runtime: " << partition_time << " us"
  //           << ", partition_size: " << _partition_size 
  //           << ", num_partitions: " << num_clusters 
  //           << ", num_deps: " << num_deps 
  //           << ", smallest precede: " << *(std::min_element(precede_time.begin(), precede_time.end()))
  //           << ", largest precede: " << *(std::max_element(precede_time.begin(), precede_time.end()))
  //           << ", max cluster size: " << *(std::max_element(_cluster_cnt.begin(), _cluster_cnt.end())) 
  //           << ", imbalanced_factor: " << get_imbalanced_factor() << "\n";
  taskflow.clear();
  partition_time = 0;

}

// --------------------------- Check functions --------------------------- 

/**
@brief: check if the graph before partitioning has cycle by dfs 
*/
bool iTAP::has_cycle_pre_partition() {

  bool cycle_before_partition = false;
 
  // initialize all node as unvisited
  for(auto& node : _nodes) {
    node._visited = false;
    node._instack = false;
  }

  // Call the recursive helper function
  // to detect cycle in different DFS trees
  for(auto& node : _nodes) {
    if(!node._visited && _isCyclicUtil(&node)) {
      cycle_before_partition = true;
    } 
  }

  return cycle_before_partition;
}

/**
@brief: check if the graph after partitioning has cycle by dfs 
*/
bool iTAP::has_cycle_post_partition() {

  for(const auto& node : _nodes) {
    if(node._cluster_id == -1) {
      std::cerr << "the graph is not fully partitioned...\n";
      std::exit(EXIT_FAILURE);
    }
  }

  build_cluster_graph();

  bool cycle_after_partition = false;
 
  for(auto& cnode : _cnodes) {
    cnode._visited = false;
    cnode._instack = false;
  }

  for(auto& cnode : _cnodes) {
    if(!cnode._visited && _isCyclicUtil(&cnode)) {
      cycle_after_partition = true;
    } 
  }

  return cycle_after_partition;
}

/**
@brief: helper function for checking cycle using dfs 
*/
template <typename T>
bool iTAP::_isCyclicUtil(T node) {
  
  if(!node->_visited) {
    // std::cout << "stucking...\n";

    // Mark the current node as visited
    // and part of recursion stack
    node->_visited = true;
    node->_instack = true;

    // Recur for all the vertices adjacent to this
    // vertex
    for(auto fanout : node->_fanouts) {
      T successor = fanout->_to;
      if(!successor->_visited && _isCyclicUtil(successor)) {
        return true;
      }
      else if(successor->_instack) {
        return true;
      }
    }
  }

  // Remove the vertex from recursion stack
  node->_instack = false;
  return false;
}


void iTAP::check_cluster() const {

  std::cout << "max cluster id = " << *(std::max_element(_cluster_cnt.begin(), _cluster_cnt.end())) << "\n"; 

  std::cout << "_cluster_cnt = [";
  for(auto num : _cluster_cnt) {
    std::cout << num << " ";
  }
  std::cout << "]\n";

}

bool iTAP::is_partition_valid() {

  if(!_partitioned) {
    return false;
  }

  // make sure every node is partitioned
  for(const auto& node : _nodes) {
    if(node._cluster_id == -1) {
      std::cerr << "incomplete partitioning...\n";
      return false;
    }
  }

  // make sure loop invarient maintained
  for(const auto& node : _nodes) {
    for(auto& edge : node._fanins) {
      Node* dependent = edge->_from;
      if(dependent->_cluster_id > node._cluster_id) {
        std::cerr << "loop invarient break...\n";
        return false;
      }
    }
  }

  if(has_cycle_pre_partition()) {
    std::cerr << "pre partition cycle introduced...\n";
    return false;
  }

  if(has_cycle_post_partition()) {
    std::cerr << "post partition cycle introduced...\n";
    return false;
  }

  return true;

}

double iTAP::get_imbalanced_factor() const{
  double largest_cluster_size = static_cast<double>(*(std::max_element(_cluster_cnt.begin(), _cluster_cnt.end()))); 
  double avg_cluster_size = 0.0; 
  size_t num_clusters = 0;;
  for(const auto& size : _cluster_cnt) {
    if(size != 0) {
      avg_cluster_size += size;
      ++num_clusters;
    }
  }
  avg_cluster_size /= num_clusters;
  return (largest_cluster_size - avg_cluster_size) / avg_cluster_size;
}

size_t iTAP::get_max_cluster_size() const{
  size_t largest_cluster_size = *(std::max_element(_cluster_cnt.begin(), _cluster_cnt.end())); 
  return largest_cluster_size;
}

bool iTAP::is_cluster_cnt_valid() const{
  if(_partitioned) {
    // run_graph already built it
    // build_cluster_graph();
    size_t num_clusters = 0;
    for(const auto& c : _cluster_cnt) {
      if(c > 0) {
        ++num_clusters;
      }
    }
    // std::cout << "_cnodes.size(): " << _cnodes.size() << "\n"
    //           << "_cluster_cnt.size(): " << _cluster_cnt.size() << "\n"
    //           << "num_clusters: " << num_clusters << "\n";
    size_t i = 0;
    for(const auto& cnode : _cnodes) {
      if(cnode._nodes.size() != _cluster_cnt[i]) {
        return false;
      }
      ++i;
    }
  }
  return true;
}

void iTAP::_export_csr() {

  // clear csr
  _adjp.clear();
  _adjncy.clear(); 
  _adjncy_size.clear();
  _dep_size.clear();

  // assign id to all the nodes
  size_t id = 0;
  for(auto& node : _nodes) {
    node._csr_id = id++;
  }
  _adjp.push_back(0);
  size_t index = 0;
  size_t count = 0;
  size_t temp;
  for(const auto& node : _nodes) {
    if(count != 0) {
      index += temp;
      if(node._fanouts.size() == 0) {
        _adjp.push_back(-1);
      }
      else {
        _adjp.push_back(index);
      }
    }
    temp = node._fanouts.size();
    ++count;
  }
  for(const auto& node : _nodes) {
    for(const auto& edge : node._fanouts) {
      Node* successor = edge->_to;
      _adjncy.push_back(successor->_csr_id);
    }
  }
  for(const auto& node : _nodes) {
    _adjncy_size.push_back(node._fanouts.size());
  }
  for(const auto& node : _nodes) {
    _dep_size.push_back(node._fanins.size());
  }
  // dump_graph();
  // std::cout << "_adjp = [";
  // for(const auto& n : _adjp) {
  //   std::cout << n << "   ";
  // }
  // std::cout << "]\n";
  // std::cout << "_adjncy = [";
  // for(const auto& n : _adjncy) {
  //   std::cout << n << "   ";
  // }
  // std::cout << "]\n";
  // std::cout << "_adjncy_size = [";
  // for(const auto& n : _adjncy_size) {
  //   std::cout << n << "   ";
  // }
  // std::cout << "]\n";
  // std::cout << "_dep_size = [";
  // for(const auto& n : _dep_size) {
  //   std::cout << n << "   ";
  // }
  // std::cout << "]\n";
}

/**
@brief: randomly remove N nodes and edges, and insert the same amount of node and edges 
*/
void iTAP::random_incre_ops(size_t N) {

  if(2*N >= _nodes.size()) {
    std::cerr << "2*N exceeds the total number of nodes.\n";
    std::exit(EXIT_FAILURE);
  }
  
  std::random_device rd;
  std::mt19937 gen(rd());

  // get N random numbers
  std::vector<size_t> random_nodes = generate_random_nums(_nodes.size(), N);
  std::vector<size_t> random_edges = generate_random_nums(_edges.size(), N);
  std::sort(random_nodes.begin(), random_nodes.end());
  std::sort(random_edges.begin(), random_edges.end());

  // remove N nodes randomly
  auto node_it = _nodes.begin();
  size_t currentIndex = 0;
  auto index_it = random_nodes.begin();
  while (node_it != _nodes.end() && index_it != random_nodes.end()) {
      if (currentIndex == *index_it) {
          // When the current index matches, erase the element
          remove_node(&(*(node_it++))); // first remove the element, then increment node_it
          ++index_it; // Move to the next index
      } else {
          ++node_it; // Move to the next element in the list
      }
      ++currentIndex; // Increment the current index in the list
  }

  // remove N edges randomly
  auto edge_it = _edges.begin();
  currentIndex = 0;
  index_it = random_edges.begin();
  while (edge_it != _edges.end() && index_it != random_edges.end()) {
      if (currentIndex == *index_it) {
          // When the current index matches, erase the element
          remove_edge(&(*(edge_it++))); // first remove the element, then increment edge_it
          ++index_it; // Move to the next index
      } else {
          ++edge_it; // Move to the next element in the list
      }
      ++currentIndex; // Increment the current index in the list
  }

  // add N random edges within the existing nodes
  // this insertion needs to follow topological order
  std::vector<Node*> topo; 
  std::queue<Node*> to_visit;
  for(auto& node : _nodes) {
    node._dep_cnt_random = 0;
  }
  for(auto& node : _nodes) {
    if(node._fanins.size() == 0) {
      to_visit.push(&node);
    }
  }
  while(!to_visit.empty()) {
    Node* cur_node = to_visit.front();
    to_visit.pop();
    topo.push_back(cur_node);
    for(auto& edge : cur_node->_fanouts) {
      Node* successor = edge->_to; 
      if(++successor->_dep_cnt_random == successor->_fanins.size()) {
        to_visit.push(successor);
      }
    }
  }
  if(N >= topo.size()) {
    std::cerr << "N exceeds topo.size()...\n";
    std::exit(EXIT_FAILURE);
  }
  // each time select 2 random indices from topo and connect them
  for(size_t i=0; i<N; i++) {
    std::uniform_int_distribution<> dis(i+1, topo.size() - 1); 
    size_t randomIndex = dis(gen);
    // use name to find if there is already an edge between them
    bool has_edge = false;
    for(auto& edge : topo[i]->_fanouts) {
      if(edge->_to->_name == topo[randomIndex]->_name) {
        has_edge = true;
      } 
    }
    if(!has_edge) {
      insert_edge(topo[i], topo[randomIndex]);
    }
  }
  
  // add N nodes 
  std::vector<Node*> nodes(N);
  for(size_t i=0; i<N; i++) {
    nodes[i] = insert_node("new_"+std::to_string(i)); 
  }

  // add N edges randomly by connectint the new nodes 
  // to the existing nodes as dependents/successors  
  std::vector<size_t> random_nodes1 = generate_random_nums(_nodes.size(), N);
  std::sort(random_nodes1.begin(), random_nodes1.end());
  auto it = random_nodes1.begin();
  size_t listIndex = 0;
  size_t new_node_index = 0;
  for (auto& element : _nodes) {
    if (it != random_nodes1.end() && listIndex == *it) {
      if(*it>=_nodes.size()-N && *it-(_nodes.size()-N) == new_node_index) {
        // to prevent the new node connected to itself
        // so some of the new nodes will not connect to anyone
        continue;
      }
      std::uniform_int_distribution<> dis(0, _nodes.size() - 1 - N); 
      size_t randomIndex = dis(gen);  
      if(randomIndex % 2 == 0) {
        insert_edge(&element, nodes[new_node_index]);
      }
      else {
        insert_edge(nodes[new_node_index], &element);
      }
      ++new_node_index;
      ++it; // Move to the next index
    }
    ++listIndex; // Increment the current index in the list
  }

}

std::vector<size_t> iTAP::generate_random_nums(int N, int count) {

    if(N <= count) {
      std::cerr << "too many random incremental ops...\n";
      std::exit(EXIT_FAILURE);
    }

    std::vector<size_t> numbers(N);
    std::iota(numbers.begin(), numbers.end(), 0); // Fill with 0, 1, ..., N-1

    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(numbers.begin(), numbers.end(), gen);

    numbers.resize(count); // Keep only the first 'count' numbers
    return numbers;
}

// yhc

int iTAP::get_cost(const std::vector<int>& runtime, int partition_size) {
  int idx = partition_size - 1;
  int base = idx * 4;

  int sum = 0;
  for (int i = 0; i < 4; ++i) {
    sum += runtime[base + i];
  }
  return sum / 4;
}


// #define ACA2_DEBUG
size_t iTAP::_SA(const int matrix_size, size_t partition_size) {
#ifdef ACA2_DEBUG
  srand(42); // Set rand seed as const for better debugging 
#else // pure random
  srand(time(NULL));
#endif

  // Edge cases then return 
  if (partition_size > 4096 || matrix_size < 8) {
    return num_nodes();
  }

  // int a = matrix8_rt[0];
  // printf("a = %d\n", a);
  // return partition_size;

  // Init candidates
  int max_partition = (matrix_size < 32) ? 2048 : 128;
  std::vector<int> candidates;
  for (int p = 1; p <= max_partition; ++p) { // Note: 1-based here
    candidates.push_back(p);
  }

  // Runtime data source
  const std::vector<int>* runtime = nullptr;
  if (matrix_size == 8) runtime = &matrix8_rt;
  else if (matrix_size == 16) runtime = &matrix16_rt;
  else if (matrix_size == 32) runtime = &matrix32_rt;
  else if (matrix_size == 64) runtime = &matrix64_rt;
  else return num_nodes(); // Unsupported matrix size fallback, act needs to assert here

  // Sanity check
  if (runtime->size() != (candidates.size()<<2)) {
    std::cerr << "[SA] Runtime size mismatch: got " << runtime->size()
              << ", expected " << candidates.size() * 4 << std::endl;
    return num_nodes();
  }

  // SA init
  int cur = candidates[rand() % candidates.size()];
  int best = cur;
  float T = 100.0;

  // SA
  for (int iter = 0; iter < 1000; ++iter) {
    // Find the position of cur in the candidate vector 
    int pos = cur - 1; 

    // Find left and right neighbors for 3 and 3 as new neighbor candidate
    const int neighbors_starts = ((pos-3) < 0) ? (0) : (pos-3);
    const int neighbors_ends = ((pos+3) >= candidates.size()) ? (candidates.size()) : (pos+3);
    int next_pos = neighbors_starts + rand() % (neighbors_ends - neighbors_starts + 1);
    int next = candidates[next_pos];
    
    int delta = get_cost(*runtime, next) - get_cost(*runtime, cur);
    if (delta < 0 || (rand() % 10000) < 10000 * exp(-delta / T)) {
      cur = next;
      if (get_cost(*runtime, cur) < get_cost(*runtime, best)) {
        best = cur;
      }
    }
    T *= 0.995;
  }
  return best;
}


// yhc

} // namespace itap
