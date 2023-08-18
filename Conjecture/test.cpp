#include <iostream>
#include <igraph/igraph.h>
#include<vector>
#include<fstream>

std::string networkname = "BaN100";

int N_trials = 2000;
int T_relax_parallel  = 100000;
int T_relax_sequential= 10000;
int T_evolve_parallel = 100000;
int T_evolve_sequential = 0;
double J_min = 0.01;
double J_max = 2.5;
int J_res = 100;

// Comment out the paths except the one where you want to save the results:
std::string path = "/home/stijn/Documents/ThesisNew/FullSim/Data/"+networkname+"/";
//std::string path =  "/theia/scratch/brussel/107/vsc10773/Data/"+networkname+"/";
//std::string path = "/kyukon/scratch/gent/459/vsc45915/Data/"+networkname+"/";
std::string pathtemp = path + "temp/";

igraph_matrix_t LoadA(const std::string& filename, int &N){
    igraph_matrix_t adjmatrix;
    std::ifstream iFile(path + filename + "_A.txt");

    // Check if the file opened:
    if (!iFile.is_open()) {
       std::cout << "File unable to open" << std::endl;
    }

    // Get matrix dimensions:
    int m, n;
    iFile >> m >> n;
    std::cout << m << ", " << n << std::endl;
    N = m;
    // Initialize the adjacency matrix:
    igraph_matrix_init(&adjmatrix, m, n);

    if (m <= 0 || n <= 0) {
        std::cout << "Error: Invalid matrix dimensions." << std::endl;
    }

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            iFile >> MATRIX(adjmatrix, i, j);
            //if (!(iFile >> MATRIX(adjmatrix, i, j))) {
                //std::cout << "Error: Unable to read matrix element " << std::endl;
            //}
        }
    }
    iFile.close();

    return adjmatrix;
}

void printA(igraph_matrix_t adjmatrix, int N)
{
    // Print the adjacency matrix
    std::cout << "Adjacency Matrix:" << std::endl;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << MATRIX(adjmatrix, i, j) << " ";
        }
        std::cout << std::endl;
    }
    return;
}

int main() {
    int N; 

    // Create an igraph_t object to hold the graph
    igraph_t graph;
    igraph_empty(&graph, N, IGRAPH_UNDIRECTED);
    igraph_matrix_t adjmatrix(LoadA(networkname,N));
    igraph_adjacency(&graph, &adjmatrix, IGRAPH_ADJ_UNDIRECTED, IGRAPH_NO_LOOPS);

    std::cout << "Number of vertices: " << igraph_vcount(&graph) << std::endl;

    // Print the adjacency matrix
    //printA(adjmatrix,N);

    //Compete edge betweenness centrality:
    igraph_vector_t EBC;
    igraph_vector_init(&EBC, igraph_ecount(&graph));

    igraph_edge_betweenness(&graph,&EBC,0,0);

    //Print EBC:
    for (int i = 0; i < igraph_vector_size(&EBC); i++) {
        printf("Edge %d: %f\n", i, VECTOR(EBC)[i]);
    }

    igraph_integer_t edge_idx = 19; // the edge index you want to query
    igraph_integer_t from, to; // variables to hold the endpoints

    // get the endpoints of the edge
    igraph_edge(&graph, edge_idx, &from, &to);

    // print the endpoints
    std::cout << "Endpoints of edge " << edge_idx << ": " << from+1 << " - " << to +1<< std::endl;



    // Free the memory
    igraph_destroy(&graph);
    //igraph_vector_destroy(&edges);
    //igraph_vector_destroy(&neighbors);

    return 0;

}
