#pragma once

#include <list>
#include <thread>
#include <queue>
#include <utility>
#include <atomic>
#include <iostream>
#include <fstream>
#include <string>
#include <climits>
#include <chrono> // For std::chrono::system_clock
#include "wsq.hpp"
#include "taskflow/taskflow.hpp"
#include "runtime_data.hpp"


namespace itap {

class Node;
class Edge;
class CNode;
class CEdge;

class Node {

  friend class Edge;
  friend class CNode;
  friend class CEdge;
  friend class iTAP;

  public:

    Node(const std::string& name=""): _name(name) {}

    size_t num_fanouts() const {
      return _fanouts.size();
    }

    size_t num_fanins() const {
      return _fanins.size();
    }

    bool has_fanout(Edge* edge) const {
      auto it = std::find(_fanouts.begin(), _fanouts.end(), edge);
      if(it != _fanouts.end()) {
        return true;
      }
      else {
        return false;
      }
    }

    bool has_fanin(Edge* edge) const {
      auto it = std::find(_fanins.begin(), _fanins.end(), edge);
      if(it != _fanins.end()) {
        return true;
      }
      else {
        return false;
      }
    }

  // private:

    // ID for csv
    size_t _csr_id = 0; 

    // for dump graph using taskflow 
    tf::Task _task;

    std::string _name;

    CNode* _cnode = NULL; // specify which cnode(cluster) this node belongs to

    std::atomic<size_t> _dep_cnt{0};
    size_t _dep_cnt_random = 0; // only for random_incre_ops()

    int _cluster_id = -1;
    int _desired_cluster_id = -1; // for incremental partitioning
  
    // when a new node is inserted, it is automatically a candidate
    bool _is_candidate = true; 
    bool _is_frontier = false;

    // to check cycles
    bool _visited {false}; 
    bool _instack {false};

    bool _partitioned = false;

    bool _type1 = false;
    bool _type2 = false;
    bool _type3 = false;

    /*
     * fanouts should not only include which fanout edges this node has(_fanouts) 
     * but also the index of this edge in the fanin edge list of its fanout nodes(_fanout_satellites).
     * so when removing the node, we just need to traverse the _fanout_satellites once and erase the
     * edge of this node from the fanin edge list of fanout nodes.
     * similar method applied to fanins.
     */
    std::list<Edge*> _fanouts;
    std::list<Edge*> _fanins;
    std::list<std::pair<Node*, std::list<Edge*>::iterator>> _fanout_satellites;
    std::list<std::pair<Node*, std::list<Edge*>::iterator>> _fanin_satellites;

    std::list<Node>::iterator _node_satellite;
    std::list<CNode>::iterator _cnode_satellite;

    // helper: to remove nodes in random_incre_ops
    bool _remove = false;
};
 
class Edge {

  friend class iTAP;

  // private:
  public:

    Node* _from;
    Node* _to;

    std::list<Edge*>::iterator _from_satellite; // edge satellite in from node _fanouts
    std::list<Edge*>::iterator _to_satellite; // edge satellite in to node _fanins 

    std::list<Edge>::iterator _satellite;
};

class CNode {

  friend class Node;
  friend class CEdge;
  friend class iTAP;

  private:

    // assign a name by combing all the names of nodes in the cluster
    std::string _get_name() const {
      std::string name = "";
      for(auto node_ptr : _nodes) {
        name += node_ptr->_name + " ";
      }
      return name;
    }

    // to check cycles
    bool _visited {false}; 
    bool _instack {false};

    tf::Task _task;

    std::list<Node*> _nodes;

    std::list<CEdge*> _fanouts;
    std::list<CEdge*> _fanins;
    std::list<CNode>::iterator _satellite;
};

class CEdge{

  friend class iTAP;

  private:

    CNode* _from;
    CNode* _to;
    std::list<CEdge>::iterator _satellite;

    Node* _from_node;
    Node* _to_node;
};
 
class iTAP {

  public:

    // constructor
    iTAP() {}
    iTAP(const std::string& filename); 
   
    // operations
    Node* insert_node(const std::string& name = "");
    Edge* insert_edge(Node* from, Node* to);
    void remove_node(Node* node); 
    void remove_edge(Edge* edge);

    // partition
    void set_partition_size(const size_t partition_size);
    

    // yhc: 
    int get_cost(const std::vector<int>& runtime, int partition_size);
    size_t _SA(const int matrix_size, size_t partition_size);
    void ACA2_partition(const int matrix_size, size_t partition_size, const int version) {
      if (version == 0) {
        printf("Version: G-PASTA\n");
        partition_size = num_nodes();
      } else if (version == 1) { // SA: 
        printf("Version: Simlated Annealing\n");
        partition_size = _SA(matrix_size, partition_size);
      } else if (version == 2) {
        printf("Version: ML (XGBoost)\n");
        const std::vector<int> ML_predict_part_sz = {299, 65, 14, 6};
        if (matrix_size < 8) {
          partition_size = num_nodes();
        } else {
          const int label = std::log2(matrix_size) - 3;
          partition_size = ML_predict_part_sz[label];
        }
      } else {
        printf("No such strategy\n");
        exit(1);
      }
      

      // Set partition size
      set_partition_size(partition_size);
      std::cout << "Picked partition_size = " << partition_size << "\n";
      
      // Do partition
      printf("Runnning gpu partitioning...\n");
      partition(false, false, matrix_size);
      // Check validation
      check_cycle(); 
    }
    // yhc

    void partition(bool incremental = false, 
                  bool only_handle_edge = false, 
                  const int matrix_size = 8);
    void reset_partition();
    void build_cluster_graph();

    size_t partition_time = 0;
    
    void dump_graph();

    void run_graph(size_t matrix_size);

    // ctest checker  
    size_t num_nodes() const {
      return _nodes.size();
    }
    bool has_cycle_pre_partition();
    bool has_cycle_post_partition();

    // incrementality exp
    void random_incre_ops(size_t N);

    // helper: generate "count" different numbers in the range [0, N) 
    std::vector<size_t> generate_random_nums(int N, int count);

    // helper: check cluster_cnt
    void check_cluster() const; 

    // helper: partition validation check
    bool is_partition_valid();

    double get_imbalanced_factor() const;

    size_t get_max_cluster_size() const;

    // check if cluster_cnt is counted correctly after partition
    bool is_cluster_cnt_valid() const;

  // private:

    // G-PASTA
    void _partition_cuda();

    // C-PASTA
    void _partition_cpu();

    std::vector<int> _adjp; // edge offset 
    std::vector<int> _adjncy; // flatterned adjacency list
    std::vector<int> _adjncy_size; // number of fanouts of each node
    std::vector<int> _dep_size; // number of dependents of each node
    void _export_csr();

    // partition size: maximum number of nodes within a cluster
    size_t _partition_size = 1;

    // indicate this graph has been full/incremental partitioned
    bool _partitioned = false;

    // overflow indicates "how many more nodes" a partition can take 
    // only used in incremental partitioning
    size_t _overflow = 0;

    // number of nodes(clusters) after partitioning
    size_t _num_clusters = 0; 

    // cycle-free clustering
    void _assign_cluster_id(Node* node_ptr, std::vector<std::atomic<size_t>>& cluster_cnt, std::atomic<int>& max_cluster_id);

    // helper function for checking cycle using dfs
    template <typename T> bool _isCyclicUtil(T node);

    //RAII 
    std::list<Node> _nodes;
    std::list<Edge> _edges;

    //
    std::list<CNode> _cnodes;
    std::list<CEdge> _cedges;

    // candidates store the newly inserted nodes
    std::vector<Node*> _candidates;
    // frontiers are starting nodes of candidates
    // to do incremental partitioning
    std::vector<Node*> _frontiers;

    // counter the number of nodes in each partition
    // no need to be atomic since we are updating it after full partitioning
    // and we only consider sequential incremental partitioning for now
    std::vector<size_t> _cluster_cnt;
    int _max_cluster_id;

    // get largest partition ID from partitioned dependents;
    // return -1 if no partitioned dependents
    int _get_Lpd(Node* node) {
      int lpd = -1;
      for(auto& edge : node->_fanins) {
        Node* dependent = edge->_from;
        if(dependent->_partitioned) {
          if(dependent->_cluster_id > lpd) {
            lpd = dependent->_cluster_id;
          }
        }
      }
      return lpd;
    }

    // get smallest partition ID from partitioned successors;
    // return -1 if no partitioned successors
    int _get_Sps(Node* node) {
      int Sps = INT_MAX;
      bool all_unpartitioned = true;
      for(auto& edge : node->_fanouts) {
        Node* successor = edge->_to;
        if(successor->_partitioned) {
          all_unpartitioned = false;
          if(successor->_cluster_id < Sps) {
            Sps = successor->_cluster_id;
          }
        }
      } 
      if(all_unpartitioned) {
        return -1;
      }
      else {
        return Sps;
      }
    }

    void check_cycle() {
      if(is_partition_valid() == false) {
        std::cerr << "cycle introduced...\n";
        std::exit(EXIT_FAILURE);
      }      
    }

  private:
    void _raise_error_acaf2() {
      // remvoe incremental
      printf("ACA Final 2: Should not be here\n");
      exit(1);      
    }

}; // class iTAP 
} // namespace itap
