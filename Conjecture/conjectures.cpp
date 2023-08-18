#include<iostream>
#include<igraph/igraph.h>
#include<random>
#include<vector>
#include<armadillo>
#include<cmath>
#include<algorithm>
#include<mpi.h> 
#include<chrono>
#include<thread>

using namespace arma;
using namespace std;

//string path = "/home/stijn/Documents/ThesisNew/FullSim/Data/";

    int networks_to_test = 5;
    double dJ = 0.0025;
    double threshold = 0.8;
    int N_trials = 10;
    int T_relax_parallel  = 100;
    int T_relax_sequential= 10;
    int T_evolve_parallel = 100;
    int T_evolve_sequential = 0;
    double J_min = 0.01;
    int T_evolve = T_evolve_parallel + T_evolve_sequential;
    double T_evolve_double = static_cast<double>(T_evolve)-1.;

void initializeSpins(vector<int>& Spins, int N)   
{   // Uniformly random pick initial spin configuration:
    //Set seed for random generator. 
    random_device rd; // obtain a random number from hardware
    mt19937 gen(rd()); // seed the generator
    uniform_int_distribution<> distr(0, 1); //define the range    

    for(int i =0; i<N; i++){        
        Spins[i] = distr(gen);
    }
    //printSpins(Spins);
    return;
}

void relaxSpins(vector<int>& Spinstate, const vector<vector<int>>& NN,double J, int N)
{
    vector<int> Spins = Spinstate; 

    // PARALLEL UPDATE PART:

    // Setup uniform distribution, to choose uniformely a spin in each time evolution step:
    random_device rd; // obtain a random number from hardware
    mt19937 gen(rd()); // seed the generator
    uniform_real_distribution<> distr2(0.,1.);
    // Initialize spin indices vector:
    vector<int> ind(N);
    iota(ind.begin(), ind.end(), 0);

    // Loop for T1 time discrete steps, updated in parallel! :
    for(int t = 0; t < T_relax_parallel; t++){
        // Random permutation of spin indices
        shuffle(ind.begin(), ind.end(), mt19937{random_device{}()});

        // Loop over all spins out of permutated indices vector:
        for(int j = 0; j < N;j++){
            int spin_chosen = ind[j];
            // Compute probability to flip spin:
            double sumSpinsNeighbouring = 0.;
            for(int i=0;i < static_cast<int>(NN[spin_chosen].size());i++)
            {
                if (Spins[ NN[spin_chosen][i] ] == 1) {
                    sumSpinsNeighbouring++;
                }
                else{sumSpinsNeighbouring--;}
            }
            int spin_chosen_value = -1;
            if(Spins[spin_chosen]==1){spin_chosen_value = 1;}
            double dE = 2.*spin_chosen_value*sumSpinsNeighbouring;
            if(dE<=0.){
                if(spin_chosen_value == 1){Spins[spin_chosen] = 0;}
                else{Spins[spin_chosen] = 1;}  
            }
            else{                
                double p = exp(-J*dE);
                // Generate random number between 0 and 1, flip if below p:
                if(distr2(gen)< p){
                    if(spin_chosen_value == 1){Spins[spin_chosen] = 0;}
                    else{Spins[spin_chosen] = 1;}  
                }
            }            
        }
    }

    // SEQUENTIALLY UPDATE PART:

    // Setup uniform distribution, to choose uniformely a spin in each time evolution step:
    uniform_int_distribution<> distr3(0, N-1); // Updated to uniform int distribution to select spin index
    
    // Loop for T time discrete steps:
    for(int t = 0; t < T_relax_sequential; t++){
        // Randomly choose a spin to update
        int spin_chosen = distr3(gen);

        // Compute probability to flip spin:
        double sumSpinsNeighbouring = 0.;
        for(int i=0;i < static_cast<int>(NN[spin_chosen].size());i++)
        {
            if (Spins[NN[spin_chosen][i]] == 1) {
                sumSpinsNeighbouring++;
            }
            else{sumSpinsNeighbouring--;}
        }
        int spin_chosen_value = -1;
        if(Spins[spin_chosen]==1){spin_chosen_value = 1;}
        double dE = 2.*spin_chosen_value*sumSpinsNeighbouring;
        if(dE<=0.){
            if(spin_chosen_value == 1){Spins[spin_chosen] = 0;}
            else{Spins[spin_chosen] = 1;}  
        }
        else{
            double p = exp(-J*dE);
            // Generate random number between 0 and 1, flip if below p:
            if(distr2(gen)< p){
                if(spin_chosen_value == 1){Spins[spin_chosen] = 0;}
                else{Spins[spin_chosen] = 1;}  
            }
        }
    }

    Spinstate = Spins;

    return;

}

vector<double> a(const Col<int>&  driver, const Col<int>&  target)
{
       
    vector<double> a(8,0.0); //Initialize vector with length total number of states (2^m since spin up or down).
    //Count frequency of each state (each state is labelled by a decimal number, found by taking spinstate as binary label and convert to decimal)
    for(int t = 0; t<T_evolve-1; t++){
        a[target[t+1]*4+target[t]*2+driver[t]]++; //Add 1 to frequency of state labeled bi2de(M[t])        
    }
    for(int j=0;j<8;j++){a[j] = a[j]/T_evolve_double;} //Normalise with total number of measurements.

    return a;
}

vector<double> a(const Col<int>&  driver1,const Col<int>&  driver2, const Col<int>&  target)
{
       
    vector<double> a(16,0.0); //Initialize vector with length total number of states (2^m since spin up or down).
    //Count frequency of each state (each state is labelled by a decimal number, found by taking spinstate as binary label and convert to decimal)
    for(int t = 0; t<T_evolve-1; t++){
        a[target[t+1]*8+target[t]*4+driver1[t]*2+driver2[t]]++; //Add 1 to frequency of state labeled bi2de(M[t])        
    }
    for(int j=0;j<16;j++){a[j] = a[j]/T_evolve_double;} //Normalise with total number of measurements.

    return a;
}

vector<double> b(const Col<int>&  target)
{
       
    vector<double> b(2,0.0); //Initialize vector with length total number of states (2^m since spin up or down).
    //Count frequency of each state (each state is labelled by a decimal number, found by taking spinstate as binary label and convert to decimal)
    for(int t = 0; t<T_evolve-1; t++){
        b[target[t]]++; //Add 1 to frequency of state labeled bi2de(M[t])        
    }
    for(int j=0;j<2;j++){b[j] = b[j]/T_evolve_double;} //Normalise with total number of measurements.

    return b;
}


vector<double> c(const Col<int>&  driver, const Col<int>&  target)
{
       
    vector<double> c(4,0.0); //Initialize vector with length total number of states (2^m since spin up or down).
    //Count frequency of each state (each state is labelled by a decimal number, found by taking spinstate as binary label and convert to decimal)
    for(int t = 0; t<T_evolve-1; t++){
        c[target[t]*2+driver[t]]++; //Add 1 to frequency of state labeled bi2de(M[t])        
    }
    for(int j=0;j<4;j++){c[j] = c[j]/T_evolve_double;} //Normalise with total number of measurements.

    return c;
}

vector<double> c(const Col<int>&  driver1,const Col<int>&  driver2, const Col<int>&  target)
{
       
    vector<double> c(8,0.0); //Initialize vector with length total number of states (2^m since spin up or down).
    //Count frequency of each state (each state is labelled by a decimal number, found by taking spinstate as binary label and convert to decimal)
    for(int t = 0; t<T_evolve-1; t++){
        c[target[t]*4+driver1[t]*2+driver2[t]]++; //Add 1 to frequency of state labeled bi2de(M[t])        
    }
    for(int j=0;j<8;j++){c[j] = c[j]/T_evolve_double;} //Normalise with total number of measurements.

    return c;
}

vector<double> d(const Col<int>&  target)
{
       
    vector<double> d(4,0.0); //Initialize vector with length total number of states (2^m since spin up or down).
    //Count frequency of each state (each state is labelled by a decimal number, found by taking spinstate as binary label and convert to decimal)
    for(int t = 1; t<T_evolve; t++){
        d[target[t]*2+target[t-1]]++;             
    }
    for(int j=0;j<4;j++){d[j] = d[j]/T_evolve_double;} //Normalise with total number of measurements.

    return d;
}

bool found(vector<bool> vector){
    bool temp = false;

    for(int i =0;i < static_cast<int>(vector.size());i++){
        if(vector[i] == false){temp = true;}
    }
    
    return temp;
}

void add(vector<vector<int>> &doublet, int d, int t){
    vector<int> to_be_added = {d,t};
    doublet.push_back(to_be_added);

    return;
}

void add(vector<vector<int>> &triplet, int d1, int d2, int t){
    vector<int> to_be_added = {d1,d2,t};
    triplet.push_back(to_be_added);

    return;
}

vector<vector<double>> skimPeak(vector<double> data, vector<double> J, double max_data, double threshold){
    vector<vector<double>> skimmedPeakPair(2);
    vector<double> skimmedPeak;
    vector<double> skimmedPeakJ;
    double peakBoundary = max_data*threshold;
    bool peakPassed = false;

    for(int i=0;i<static_cast<int>(data.size());i++){
        if(peakPassed == true && data[i] < peakBoundary){break;} //Stop the loop when we passed the peak and the data is smaller than our threshold.
        if(data[i] >= peakBoundary){           
            skimmedPeak.push_back(data[i]);
            skimmedPeakJ.push_back(J[i]);
        }
        if( abs(data[i] - max_data) < 1e-10){peakPassed = true;}
    }

    skimmedPeakPair[0] = skimmedPeak;
    skimmedPeakPair[1] = skimmedPeakJ;

    return skimmedPeakPair;
}

void normalise(vector<double> &data){

    //Normalise
    double sumData = 0.;

    for(int i=0;i<static_cast<int>(data.size());i++){
        sumData += data[i];
    }  
    if(sumData != 0.){
        for(int i=0;i<static_cast<int>(data.size());i++){        
            data[i] = data[i]/sumData;
        } 
    }

    return;
}

void mean(vector<double> PDF, vector<double> J, double &mean){
    mean = 0.;
    for(int i=0;i<static_cast<int>(PDF.size());i++){
        mean += PDF[i]*J[i];
    } 

    return;
}

void variance(vector<double> PDF, vector<double> J, double mean, double &variance){ 
    
    variance = 0.;

    for(int i=0;i<static_cast<int>(PDF.size());i++){
        variance += pow(J[i]-mean,2)*PDF[i];
    } 

    return;
}

void skewness(vector<double> PDF, vector<double> J, double mean, double variance, double &skewness){
    double std = sqrt(variance);

    if(std != 0.){
        skewness = 0.;

        for(int i=0;i<static_cast<int>(PDF.size());i++){
            skewness += pow((J[i]-mean)/std,3)*PDF[i];
        }
    }
    else{
        skewness = -111.; // signals that the triplets/doublet was not found (std ==0);
    }


    return; 
}

void kurtosis(vector<double> PDF, vector<double> J, double mean, double variance, double &kurtosis){
    double std = sqrt(variance);

    if(std != 0.){
        kurtosis = 0.;

        for(int i=0;i<static_cast<int>(PDF.size());i++){
            kurtosis += pow((J[i]-mean)/std,4)*PDF[i];
        }
    }
    else{
        kurtosis = -111.; // signals that the doublet/triplet was not found (std ==0)
    }

    return; 
}




int main(int argc, char *argv[]) {

    //Initialize MPI parallellisation:
    MPI_Init(&argc, &argv);

    int num_procs, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);   
    int chunk_size;
    if(networks_to_test % num_procs == 0){chunk_size = networks_to_test/num_procs;} 
    else{chunk_size = networks_to_test/num_procs+1;} 
    const int start =  rank * chunk_size;
    const int end = start + chunk_size;

    int N=100; 

    int writing = 0; // Signals if the process is writing or not. 

    for(int network=start;network<end;network++){  

        // Create an igraph_t object to hold the graph
        igraph_t graph;
        igraph_empty(&graph, N, IGRAPH_UNDIRECTED);

        // Generate the graph here:

        // For a random tree graph with N nodes:
        //Initialize Prufer sequence
        igraph_vector_int_t PruferSeq;
        igraph_vector_int_init(&PruferSeq,N-2);

        //Set seed for random generator.
        random_device rd; // obtain a random number from hardware
        mt19937 gen(rd()); // seed the generator

        // Uniformly random generate N-2 numbers from 0 to N.
        uniform_int_distribution<> distr(0, N-1); // define the range
        //cout << "PruferSeq: (" ;
        for(int n=0; n<N-2; n++){
            igraph_vector_int_set(&PruferSeq, n , distr(gen)); // generate random numbers
            //cout << igraph_vector_int_get(&PruferSeq,n) <<  ",";
        }
        //cout << endl;

        //Convert randomly generated Prufer sequence to tree graph.
        igraph_from_prufer(&graph, &PruferSeq);
        igraph_vector_int_destroy(&PruferSeq); // Free


        // Get the adjacency matrix of the graph
        igraph_matrix_t adjacency;
        igraph_matrix_init(&adjacency,N,N);
        igraph_get_adjacency(&graph, &adjacency, IGRAPH_GET_ADJACENCY_BOTH,NULL,IGRAPH_NO_LOOPS);

        // Print the adjacency matrix
        //igraph_matrix_print(&adjacency);
        vector<vector<int>> A(N,vector<int>(N,0));
        // Copy adjacency matrix to A
        //cout << "hahahahaha" << endl;
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                A[i][j] = MATRIX(adjacency, i, j);
                //cout << A[i][j] << " ";
            }
            //cout << endl;
        }
        igraph_matrix_destroy(&adjacency); // Free memory

        // Compute neighbouring spins for each spinsite:
        vector<vector<int>> NN(N);
        for(int i=0;i<N;i++){
            for(int j=0;j<N;j++){
                if(A[i][j]==1){NN[i].push_back(j);
                //cout << i<<","<<j<<endl;
                }
            }
        }


        //Compute degrees of each vertix:
        igraph_vector_int_t deg_;
        igraph_vector_int_init(&deg_, N);
        igraph_degree(&graph, &deg_, igraph_vss_all(), IGRAPH_ALL, IGRAPH_NO_LOOPS);
        vector<int> deg(N,0); //Convert to std vector
        for (int i = 0; i < N; i++) {
            deg[i] = static_cast<int>(VECTOR(deg_)[i]);        
        }
        igraph_vector_int_destroy(&deg_); // Free memory

        //Compute vertex betweenness centrality of each vertex:
        igraph_vector_t VBC_;
        igraph_vector_init(&VBC_, N);
        igraph_betweenness(&graph, &VBC_, igraph_vss_all(), false, NULL);
        vector<double> VBC(N,0.); //Convert to std vector
        for (int i = 0; i < N; i++) {
            VBC[i] = VECTOR(VBC_)[i];
        }
        igraph_vector_destroy(&VBC_); // Free memory

        //Compete edge betweenness centrality of each edge:
        igraph_vector_t EBC_;
        igraph_vector_init(&EBC_, igraph_ecount(&graph));
        igraph_edge_betweenness(&graph,&EBC_,false, NULL); 
        vector<double> EBC(igraph_vector_size(&EBC_),0.); //Convert to std vector
        for (int i = 0; i < igraph_vector_size(&EBC_); i++) {
            EBC[i] =  VECTOR(EBC_)[i];
        }
        igraph_vector_destroy(&EBC_); // Free memory

     /*
        //Print degree:
        for (int i = 0; i < igraph_vector_int_size(&deg_); i++) {
            cout << "degree of vertex " << i+1 << ": " << deg[i] << endl;
        }
        //Print VBC:
        for (int i = 0; i < igraph_vector_size(&VBC_); i++) {
            cout << "VBC of vertex " << i+1 << ": " << VBC[i] << endl;
        }
        //Print EBC:
        for (int i = 0; i < igraph_vector_size(&EBC_); i++) {
            cout << "EBC of edge " << i+1 << ": " << EBC[i] << endl;
        }

        //Pint edges:
        for (int i = 0; i < igraph_vector_size(&EBC_); i++) {
            igraph_integer_t from, to; // variables to hold the endpoints
            // get the endpoints of the edge
            igraph_edge(&graph, i, &from, &to);
            cout << "Edge " << i+1 << ": (" << from+1 << "," << to+1 << ")" << endl;
        }   
     */




        //Save all doublets and triplets to compute pairwise TE to in the vector:
        vector<vector<int>> doublets;
        vector<vector<int>> triplets;

        //Procedure 3.1.Y
        int id_3_1_Y_1;
        double max_VBC =-1.;
        double temp = 0.;
        for(int i=0;i<N;i++){
            temp = VBC[i];
            if(temp > max_VBC){
                max_VBC = temp;
                id_3_1_Y_1 = i;
            }
        }

        //Procedure 3.2.Y
        int id_3_2_Y_1;
        double max_deg =-1.;
        temp = 0.;
        for(int i=0;i<N;i++){
            temp = deg[i];
            if(temp > max_deg){
                max_deg = temp;
                id_3_2_Y_1 = i;
            }
        }

        //Procedure 4.X
        int eid_4_X_1;
        double max_EBC = -1.;
        temp = 0.;
        for(int i=0;i<static_cast<int>(EBC.size());i++){
            temp = EBC[i];
            if(temp > max_EBC){
                max_EBC = temp;
                eid_4_X_1 = i;
            }
        }    
        igraph_integer_t vid_4_X_1, vid_4_X_2;
        igraph_edge(&graph, eid_4_X_1, &vid_4_X_1, &vid_4_X_2);

        // Procedure 2

        // Procedure 2.1
            //Find max sums of VBC's:
        double max = -1.;
        double sum = 0.;
        int id_2_1_1,id_2_1_2,neighbor;
        for(int i=0;i<N;i++){
            int number_of_neighbors = static_cast<int>(NN[i].size());
            for(int j=0;j<number_of_neighbors;j++){
                neighbor = NN[i][j];
                sum = VBC[i]+VBC[neighbor];
                if(sum > max){
                    max = sum;
                    if(VBC[i] >= VBC[neighbor]){
                        id_2_1_1 = i;
                        id_2_1_2 = neighbor;
                    }
                    else{
                        id_2_1_1 = neighbor;
                        id_2_1_2 = i;
                    }                
                }           
            }        
        }
        add(doublets,id_2_1_1,id_2_1_2);
        add(doublets,id_2_1_2,id_2_1_1);
        //cout << id_2_1_1 << "," << id_2_1_2 << endl;

        // Procedure 2.2
            //Find max sums of deg's:
        max = -1.;
        sum = 0.;
        int id_2_2_1,id_2_2_2;
        for(int i=0;i<N;i++){
            int number_of_neighbors = static_cast<int>(NN[i].size());
            for(int j=0;j<number_of_neighbors;j++){
                neighbor = NN[i][j];
                sum = deg[i]+deg[neighbor];
                if(sum > max){
                    max = sum;
                    if(deg[i] >= deg[neighbor]){
                        id_2_2_1 = i;
                        id_2_2_2 = neighbor;
                    }
                    else{
                        id_2_2_1 = neighbor;
                        id_2_2_2 = i;
                    } 
                }           
            }        
        }
        add(doublets,id_2_2_1,id_2_2_2);
        add(doublets,id_2_2_2,id_2_2_1);
        // Procedure 2.3
            //Find max sums of deg's:
        max = -1.;
        sum = 0.;
        int id_2_3_1,id_2_3_2;
        for(int i=0;i<N;i++){
            int number_of_neighbors = static_cast<int>(NN[i].size());
            for(int j=0;j<number_of_neighbors;j++){
                neighbor = NN[i][j];
                sum = deg[i]/max_deg+VBC[neighbor]/max_VBC;
                if(sum > max){
                    max = sum;
                    if(deg[i]/max_deg >= VBC[neighbor]/max_VBC){
                        id_2_3_1 = i;
                        id_2_3_2 = neighbor;
                    }
                    else{
                        id_2_3_1 = neighbor;
                        id_2_3_2 = i;
                    } 
                }           
            }        
        }
        add(doublets,id_2_3_1,id_2_3_2);
        add(doublets,id_2_3_2,id_2_3_1);


        // Procedure 3
            //Procedure 3.1.1
        int amount_of_neighbors = static_cast<int>(NN[id_3_1_Y_1].size());
        int id_3_1_1_2,id_3_1_1_3;
        max = -1.;
        sum = 0.;

        for(int i=0;i<amount_of_neighbors;i++){
            for(int j=i+1;j<amount_of_neighbors;j++){
                sum = VBC[NN[id_3_1_Y_1][i]]+VBC[NN[id_3_1_Y_1][j]];
                if(sum > max){
                    max = sum;
                    if(VBC[NN[id_3_1_Y_1][i]] >= VBC[NN[id_3_1_Y_1][j]]){
                        id_3_1_1_2 = NN[id_3_1_Y_1][i];
                        id_3_1_1_3 = NN[id_3_1_Y_1][j];                   
                    }
                    else{
                        id_3_1_1_2 = NN[id_3_1_Y_1][j];
                        id_3_1_1_3 = NN[id_3_1_Y_1][i];
                    }
                }
            }
        }
        //cout << id_3_1_1_2 << "," << id_3_1_1_3 << "," << id_3_1_Y_1 << endl;
        add(triplets,id_3_1_1_2,id_3_1_1_3,id_3_1_Y_1);
        add(triplets,id_3_1_Y_1,id_3_1_1_3,id_3_1_1_2);
        add(triplets,id_3_1_Y_1,id_3_1_1_2,id_3_1_1_3);
        add(doublets,id_3_1_1_2,id_3_1_Y_1);
        add(doublets,id_3_1_Y_1,id_3_1_1_2);
        add(doublets,id_3_1_1_3,id_3_1_Y_1);
        add(doublets,id_3_1_Y_1,id_3_1_1_3);
        add(doublets,id_3_1_1_3,id_3_1_1_2);
        add(doublets,id_3_1_1_2,id_3_1_1_3);

        //Procedure 3.1.2
        int id_3_1_2_2,id_3_1_2_3;
        max = -1.;
        sum = 0.;
        for(int i=0;i<amount_of_neighbors;i++){
            for(int j=i+1;j<amount_of_neighbors;j++){
                sum = deg[NN[id_3_1_Y_1][i]]+deg[NN[id_3_1_Y_1][j]];
                if(sum > max){
                    max = sum;
                    if(deg[NN[id_3_1_Y_1][i]] >= deg[NN[id_3_1_Y_1][j]]){
                        id_3_1_2_2 = NN[id_3_1_Y_1][i];
                        id_3_1_2_3 = NN[id_3_1_Y_1][j];                   
                    }
                    else{
                        id_3_1_2_2 = NN[id_3_1_Y_1][j];
                        id_3_1_2_3 = NN[id_3_1_Y_1][i];
                    }

                }
            }
        }
        add(triplets,id_3_1_2_2,id_3_1_2_3,id_3_1_Y_1);
        add(triplets,id_3_1_Y_1,id_3_1_2_3,id_3_1_2_2);
        add(triplets,id_3_1_Y_1,id_3_1_2_2,id_3_1_2_3);
        add(doublets,id_3_1_2_2,id_3_1_Y_1);
        add(doublets,id_3_1_Y_1,id_3_1_2_2);
        add(doublets,id_3_1_2_3,id_3_1_Y_1);
        add(doublets,id_3_1_Y_1,id_3_1_2_3);
        add(doublets,id_3_1_2_3,id_3_1_2_2);
        add(doublets,id_3_1_2_2,id_3_1_2_3);
    


        //Procedure 3.1.3
        int id_3_1_3_2,id_3_1_3_3;
        max = -1.;
        sum = 0.;
        //Get all incident edges to id_3_1_Y_1
        igraph_vector_int_t edges1;
        igraph_vector_int_init(&edges1, 0);
        igraph_incident(&graph, &edges1, id_3_1_Y_1, IGRAPH_ALL);
        // iterate over edges1 twice
        for (int i = 0; i < igraph_vector_int_size(&edges1); i++) {
            igraph_integer_t eid1 = VECTOR(edges1)[i]; // get edge id
            for (int j = i + 1; j < igraph_vector_int_size(&edges1); j++) {
                igraph_integer_t eid2 = VECTOR(edges1)[j]; // get edge id
                sum = EBC[eid1]+EBC[eid2];
                if(sum > max){
                    max = sum;
                    if(EBC[eid1] >= EBC[eid2]){
                        igraph_integer_t source, target;
                        igraph_edge(&graph, eid1, &source, &target);
                        if(static_cast<int>(source) == id_3_1_Y_1){id_3_1_3_2 = static_cast<int>(target);}
                        else{id_3_1_3_2 = static_cast<int>(source);}
                        igraph_edge(&graph, eid2, &source, &target);
                        if(static_cast<int>(source) == id_3_1_Y_1){id_3_1_3_3 = static_cast<int>(target);}
                        else{id_3_1_3_3 = static_cast<int>(source);}                 
                    }
                    else{
                        igraph_integer_t source, target;
                        igraph_edge(&graph, eid2, &source, &target);
                        if(static_cast<int>(source) == id_3_1_Y_1){id_3_1_3_2 = static_cast<int>(target);}
                        else{id_3_1_3_2 = static_cast<int>(source);}
                        igraph_edge(&graph, eid1, &source, &target);
                        if(static_cast<int>(source) == id_3_1_Y_1){id_3_1_3_3 = static_cast<int>(target);}
                        else{id_3_1_3_3 = static_cast<int>(source);}  
                    }
                }
            }
        }
        add(triplets,id_3_1_3_2,id_3_1_3_3,id_3_1_Y_1);
        add(triplets,id_3_1_Y_1,id_3_1_3_3,id_3_1_3_2);
        add(triplets,id_3_1_Y_1,id_3_1_3_2,id_3_1_3_3);
        add(doublets,id_3_1_3_2,id_3_1_Y_1);
        add(doublets,id_3_1_Y_1,id_3_1_3_2);
        add(doublets,id_3_1_3_3,id_3_1_Y_1);
        add(doublets,id_3_1_Y_1,id_3_1_3_3);
        add(doublets,id_3_1_3_3,id_3_1_3_2);
        add(doublets,id_3_1_3_2,id_3_1_3_3);


        //Procedure 3.1.4
        int id_3_1_4_2,id_3_1_4_3;
        max = -1.;
        sum = 0.;
        for(int i=0;i<amount_of_neighbors;i++){
            for(int j=i+1;j<amount_of_neighbors;j++){
                sum = deg[NN[id_3_1_Y_1][i]]/max_deg+VBC[NN[id_3_1_Y_1][j]]/max_VBC;
                if(sum > max){
                    max = sum;
                    if(deg[NN[id_3_1_Y_1][i]]/max_deg > VBC[NN[id_3_1_Y_1][j]]/max_VBC){
                        id_3_1_4_2 = NN[id_3_1_Y_1][i];
                        id_3_1_4_3 = NN[id_3_1_Y_1][j];                   
                    }
                    else{
                        id_3_1_4_2 = NN[id_3_1_Y_1][j];
                        id_3_1_4_3 = NN[id_3_1_Y_1][i];
                    }
                }
            }
        }
        add(triplets,id_3_1_4_2,id_3_1_4_3,id_3_1_Y_1);
        add(triplets,id_3_1_Y_1,id_3_1_4_3,id_3_1_4_2);
        add(triplets,id_3_1_Y_1,id_3_1_4_2,id_3_1_4_3);
        add(doublets,id_3_1_4_2,id_3_1_Y_1);
        add(doublets,id_3_1_Y_1,id_3_1_4_2);
        add(doublets,id_3_1_4_3,id_3_1_Y_1);
        add(doublets,id_3_1_Y_1,id_3_1_4_3);
        add(doublets,id_3_1_4_3,id_3_1_4_2);
        add(doublets,id_3_1_4_2,id_3_1_4_3);


        //Procedure 3.2.1
        amount_of_neighbors = static_cast<int>(NN[id_3_2_Y_1].size());
        int id_3_2_1_2,id_3_2_1_3;
        max = -1.;
        sum = 0.;

        for(int i=0;i<amount_of_neighbors;i++){
            for(int j=i+1;j<amount_of_neighbors;j++){
                sum = VBC[NN[id_3_2_Y_1][i]]+VBC[NN[id_3_2_Y_1][j]];
                if(sum > max){
                    max = sum;
                    if(VBC[NN[id_3_2_Y_1][i]] >= VBC[NN[id_3_2_Y_1][j]]){
                        id_3_2_1_2 = NN[id_3_2_Y_1][i];
                        id_3_2_1_3 = NN[id_3_2_Y_1][j];                   
                    }
                    else{
                        id_3_2_1_2 = NN[id_3_2_Y_1][j];
                        id_3_2_1_3 = NN[id_3_2_Y_1][i];
                    }
                }
            }
        }

        add(triplets,id_3_2_1_2,id_3_2_1_3,id_3_2_Y_1);
        add(triplets,id_3_2_Y_1,id_3_2_1_3,id_3_2_1_2);
        add(triplets,id_3_2_Y_1,id_3_2_1_2,id_3_2_1_3);
        add(doublets,id_3_2_1_2,id_3_2_Y_1);
        add(doublets,id_3_2_Y_1,id_3_2_1_2);
        add(doublets,id_3_2_1_3,id_3_2_Y_1);
        add(doublets,id_3_2_Y_1,id_3_2_1_3);
        add(doublets,id_3_2_1_3,id_3_2_1_2);
        add(doublets,id_3_2_1_2,id_3_2_1_3);

        //Procedure 3.2.2
        int id_3_2_2_2,id_3_2_2_3;
        max = -1.;
        sum = 0.;
        for(int i=0;i<amount_of_neighbors;i++){
            for(int j=i+1;j<amount_of_neighbors;j++){
                sum = deg[NN[id_3_2_Y_1][i]]+deg[NN[id_3_2_Y_1][j]];
                if(sum > max){
                    max = sum;
                    if(deg[NN[id_3_2_Y_1][i]] >= deg[NN[id_3_2_Y_1][j]]){
                        id_3_2_2_2 = NN[id_3_2_Y_1][i];
                        id_3_2_2_3 = NN[id_3_2_Y_1][j];                   
                    }
                    else{
                        id_3_2_2_2 = NN[id_3_2_Y_1][j];
                        id_3_2_2_3 = NN[id_3_2_Y_1][i];
                    }

                }
            }
        }
        add(triplets,id_3_2_2_2,id_3_2_2_3,id_3_2_Y_1);
        add(triplets,id_3_2_Y_1,id_3_2_2_3,id_3_2_2_2);
        add(triplets,id_3_2_Y_1,id_3_2_2_2,id_3_2_2_3);
        add(doublets,id_3_2_2_2,id_3_2_Y_1);
        add(doublets,id_3_2_Y_1,id_3_2_2_2);
        add(doublets,id_3_2_2_3,id_3_2_Y_1);
        add(doublets,id_3_2_Y_1,id_3_2_2_3);
        add(doublets,id_3_2_2_3,id_3_2_2_2);
        add(doublets,id_3_2_2_2,id_3_2_2_3);
    


        //Procedure 3.2.3
        int id_3_2_3_2,id_3_2_3_3;
        max = -1.;
        sum = 0.;
        //Get all incident edges to id_3_2_Y_1
        igraph_vector_int_t edges2;
        igraph_vector_int_init(&edges2, 0);
        igraph_incident(&graph, &edges2, id_3_2_Y_1, IGRAPH_ALL);
        // iterate over edges2 twice
        for (int i = 0; i < igraph_vector_int_size(&edges2); i++) {
            igraph_integer_t eid1 = VECTOR(edges2)[i]; // get edge id
            for (int j = i + 1; j < igraph_vector_int_size(&edges2); j++) {
                igraph_integer_t eid2 = VECTOR(edges2)[j]; // get edge id
                sum = EBC[eid1]+EBC[eid2];
                if(sum > max){
                    max = sum;
                    if(EBC[eid1] >= EBC[eid2]){
                        igraph_integer_t source, target;
                        igraph_edge(&graph, eid1, &source, &target);
                        if(static_cast<int>(source) == id_3_2_Y_1){id_3_2_3_2 = static_cast<int>(target);}
                        else{id_3_2_3_2 = static_cast<int>(source);}
                        igraph_edge(&graph, eid2, &source, &target);
                        if(static_cast<int>(source) == id_3_2_Y_1){id_3_2_3_3 = static_cast<int>(target);}
                        else{id_3_2_3_3 = static_cast<int>(source);}                 
                    }
                    else{
                        igraph_integer_t source, target;
                        igraph_edge(&graph, eid2, &source, &target);
                        if(static_cast<int>(source) == id_3_2_Y_1){id_3_2_3_2 = static_cast<int>(target);}
                        else{id_3_2_3_2 = static_cast<int>(source);}
                        igraph_edge(&graph, eid1, &source, &target);
                        if(static_cast<int>(source) == id_3_2_Y_1){id_3_2_3_3 = static_cast<int>(target);}
                        else{id_3_2_3_3 = static_cast<int>(source);}  
                    }
                }
            }
        }
        add(triplets,id_3_2_3_2,id_3_2_3_3,id_3_2_Y_1);
        add(triplets,id_3_2_Y_1,id_3_2_3_3,id_3_2_3_2);
        add(triplets,id_3_2_Y_1,id_3_2_3_2,id_3_2_3_3);
        add(doublets,id_3_2_3_2,id_3_2_Y_1);
        add(doublets,id_3_2_Y_1,id_3_2_3_2);
        add(doublets,id_3_2_3_3,id_3_2_Y_1);
        add(doublets,id_3_2_Y_1,id_3_2_3_3);
        add(doublets,id_3_2_3_3,id_3_2_3_2);
        add(doublets,id_3_2_3_2,id_3_2_3_3);


        //Procedure 3.2.4
        int id_3_2_4_2,id_3_2_4_3;
        max = -1.;
        sum = 0.;
        for(int i=0;i<amount_of_neighbors;i++){
            for(int j=i+1;j<amount_of_neighbors;j++){
                sum = deg[NN[id_3_2_Y_1][i]]/max_deg+VBC[NN[id_3_2_Y_1][j]]/max_VBC;
                if(sum > max){
                    max = sum;
                    if(deg[NN[id_3_2_Y_1][i]]/max_deg > VBC[NN[id_3_2_Y_1][j]]/max_VBC){
                        id_3_2_4_2 = NN[id_3_2_Y_1][i];
                        id_3_2_4_3 = NN[id_3_2_Y_1][j];                   
                    }
                    else{
                        id_3_2_4_2 = NN[id_3_2_Y_1][j];
                        id_3_2_4_3 = NN[id_3_2_Y_1][i];
                    }
                }
            }
        }
        add(triplets,id_3_2_4_2,id_3_2_4_3,id_3_2_Y_1);
        add(triplets,id_3_2_Y_1,id_3_2_4_3,id_3_2_4_2);
        add(triplets,id_3_2_Y_1,id_3_2_4_2,id_3_2_4_3);
        add(doublets,id_3_2_4_2,id_3_2_Y_1);
        add(doublets,id_3_2_Y_1,id_3_2_4_2);
        add(doublets,id_3_2_4_3,id_3_2_Y_1);
        add(doublets,id_3_2_Y_1,id_3_2_4_3);
        add(doublets,id_3_2_4_3,id_3_2_4_2);
        add(doublets,id_3_2_4_2,id_3_2_4_3);


        //Procedure 4.1
        int id_4_1_1,id_4_1_2,id_4_1_3;
        if(VBC[vid_4_X_1] >= VBC[vid_4_X_2]){
            id_4_1_1 = static_cast<int>(vid_4_X_1);
            id_4_1_2 = static_cast<int>(vid_4_X_2);
        }
        else{
            id_4_1_1 = static_cast<int>(vid_4_X_2);
            id_4_1_2 = static_cast<int>(vid_4_X_1);
        }

        max = -1.;
        int number_of_neighbors_3 = static_cast<int>(NN[vid_4_X_1].size());
        int number_of_neighbors_4 = static_cast<int>(NN[vid_4_X_2].size());

        for(int i=0; i<number_of_neighbors_3; i++){
            if(VBC[NN[vid_4_X_1][i]] > max){
                if(NN[vid_4_X_1][i] == id_4_1_1 || NN[vid_4_X_1][i] == id_4_1_2){continue;}
                max = VBC[NN[vid_4_X_1][i]];
                id_4_1_3 = NN[vid_4_X_1][i];
            }        
        }
        for(int i=0; i<number_of_neighbors_3; i++){
            if(VBC[NN[vid_4_X_2][i]] > max){
                if(NN[vid_4_X_2][i] == id_4_1_1 || NN[vid_4_X_2][i] == id_4_1_2){continue;}
                max = VBC[NN[vid_4_X_2][i]];
                id_4_1_3 = NN[vid_4_X_2][i];
            }        
        }
        add(triplets,id_4_1_2,id_4_1_3,id_4_1_1);
        add(triplets,id_4_1_1,id_4_1_3,id_4_1_2);
        add(triplets,id_4_1_1,id_4_1_2,id_4_1_3);
        add(doublets,id_4_1_2,id_4_1_1);
        add(doublets,id_4_1_1,id_4_1_2);
        add(doublets,id_4_1_3,id_4_1_1);
        add(doublets,id_4_1_1,id_4_1_3);
        add(doublets,id_4_1_3,id_4_1_2);
        add(doublets,id_4_1_2,id_4_1_3);

        //Procedure 4.2
        int id_4_2_1,id_4_2_2,id_4_2_3;
        if(VBC[vid_4_X_1] >= VBC[vid_4_X_2]){
            id_4_2_1 = static_cast<int>(vid_4_X_1);
            id_4_2_2 = static_cast<int>(vid_4_X_2);
        }
        else{
            id_4_2_1 = static_cast<int>(vid_4_X_2);
            id_4_2_2 = static_cast<int>(vid_4_X_1);
        }

        max = -1.;
        for(int i=0; i<number_of_neighbors_3; i++){
            if(deg[NN[vid_4_X_1][i]] > max){
                if(NN[vid_4_X_1][i] == id_4_1_1 || NN[vid_4_X_1][i] == id_4_1_2){continue;}
                max = deg[NN[vid_4_X_1][i]];
                id_4_2_3 = NN[vid_4_X_1][i];
            }        
        }
        for(int i=0; i<number_of_neighbors_3; i++){
            if(deg[NN[vid_4_X_2][i]] > max){
                if(NN[vid_4_X_2][i] == id_4_1_1 || NN[vid_4_X_2][i] == id_4_1_2){continue;}
                max = deg[NN[vid_4_X_2][i]];
                id_4_2_3 = NN[vid_4_X_2][i];
            }        
        }
        add(triplets,id_4_2_2,id_4_2_3,id_4_2_1);
        add(triplets,id_4_2_1,id_4_2_3,id_4_2_2);
        add(triplets,id_4_2_1,id_4_2_2,id_4_2_3);
        add(doublets,id_4_2_2,id_4_2_1);
        add(doublets,id_4_2_1,id_4_2_2);
        add(doublets,id_4_2_3,id_4_2_1);
        add(doublets,id_4_2_1,id_4_2_3);
        add(doublets,id_4_2_3,id_4_2_2);
        add(doublets,id_4_2_2,id_4_2_3);

        //4.3 missing!


        //Procedure 5.1
        int id_5_1_1,id_5_1_2,id_5_1_3;
        max = -1.;

        for(int i=0;i<N;i++){
            int num_neighbors = static_cast<int>(NN[i].size());
            for(int j= 0; j < num_neighbors; j++){
                for(int k= j+1; k < num_neighbors; k++){
                    igraph_integer_t eid1,eid2;
                    igraph_get_eid(&graph, &eid1, i, NN[i][j], IGRAPH_UNDIRECTED, 0);
                    igraph_get_eid(&graph, &eid2, i, NN[i][k], IGRAPH_UNDIRECTED, 0);
                    sum = EBC[eid1] + EBC[eid2];
                    if(sum > max){
                        max = sum;
                        id_5_1_1 = i;
                        if(VBC[NN[i][j]] >= VBC[NN[i][k]]){
                            id_5_1_2 = NN[i][j];
                            id_5_1_3 = NN[i][k];
                        }
                        else{
                            id_5_1_2 = NN[i][k];
                            id_5_1_3 = NN[i][j];
                        }
                    }
                }
            }
        }
        add(triplets,id_5_1_2,id_5_1_3,id_5_1_1);
        add(triplets,id_5_1_1,id_5_1_3,id_5_1_2);
        add(triplets,id_5_1_1,id_5_1_2,id_5_1_3);
        add(doublets,id_5_1_2,id_5_1_1);
        add(doublets,id_5_1_1,id_5_1_2);
        add(doublets,id_5_1_3,id_5_1_1);
        add(doublets,id_5_1_1,id_5_1_3);
        add(doublets,id_5_1_3,id_5_1_2);
        add(doublets,id_5_1_2,id_5_1_3);

        //Procedure 5.2
        int id_5_2_1,id_5_2_2,id_5_2_3;
        max = -1.;

        for(int i=0;i<N;i++){
            int num_neighbors = static_cast<int>(NN[i].size());
            for(int j= 0; j < num_neighbors; j++){
                for(int k= j+1; k < num_neighbors; k++){
                    if(VBC[i] < VBC[NN[i][j]] || VBC[i] < VBC[NN[i][k]]){continue;}
                    sum = VBC[i] + VBC[NN[i][j]] + VBC[NN[i][k]];
                    if(sum > max){
                        max = sum;
                        id_5_2_1 = i;
                        if(VBC[NN[i][j]] >= VBC[NN[i][k]]){
                            id_5_2_2 = NN[i][j];
                            id_5_2_3 = NN[i][k];
                        }
                        else{
                            id_5_2_2 = NN[i][k];
                            id_5_2_3 = NN[i][j];
                        }
                    }
                }
            }
        }
        add(triplets,id_5_2_2,id_5_2_3,id_5_2_1);
        add(triplets,id_5_2_1,id_5_2_3,id_5_2_2);
        add(triplets,id_5_2_1,id_5_2_2,id_5_2_3);
        add(doublets,id_5_2_2,id_5_2_1);
        add(doublets,id_5_2_1,id_5_2_2);
        add(doublets,id_5_2_3,id_5_2_1);
        add(doublets,id_5_2_1,id_5_2_3);
        add(doublets,id_5_2_3,id_5_2_2);
        add(doublets,id_5_2_2,id_5_2_3);


        //Procedure 5.3
        int id_5_3_1,id_5_3_2,id_5_3_3;
        max = -1.;

        for(int i=0;i<N;i++){
            int num_neighbors = static_cast<int>(NN[i].size());
            for(int j= 0; j < num_neighbors; j++){
                for(int k= j+1; k < num_neighbors; k++){
                    if(deg[i] < deg[NN[i][j]] || deg[i] < deg[NN[i][k]]){continue;}
                    sum = deg[i] + deg[NN[i][j]] + deg[NN[i][k]];
                    if(sum > max){
                        max = sum;
                        id_5_3_1 = i;
                        if(deg[NN[i][j]] >= deg[NN[i][k]]){
                            id_5_3_2 = NN[i][j];
                            id_5_3_3 = NN[i][k];
                        }
                        else{
                            id_5_3_2 = NN[i][k];
                            id_5_3_3 = NN[i][j];
                        }
                    }
                }
            }
        }
        add(triplets,id_5_3_2,id_5_3_3,id_5_3_1);
        add(triplets,id_5_3_1,id_5_3_3,id_5_3_2);
        add(triplets,id_5_3_1,id_5_3_2,id_5_3_3);
        add(doublets,id_5_3_2,id_5_3_1);
        add(doublets,id_5_3_1,id_5_3_2);
        add(doublets,id_5_3_3,id_5_3_1);
        add(doublets,id_5_3_1,id_5_3_3);
        add(doublets,id_5_3_3,id_5_3_2);
        add(doublets,id_5_3_2,id_5_3_3);

        int doublets_size = static_cast<int>(doublets.size());
        int triplets_size = static_cast<int>(triplets.size());

        igraph_destroy(&graph); // Free memory


        //Ising MCMC starts here:
        vector<double> max_TE1(doublets_size,0.);
        vector<double> max_TE2(triplets_size,0.);
        double max_Chi = 0.;

        vector<double> max_TE1_J(doublets_size,0.);
        vector<double> max_TE2_J(triplets_size,0.);
        double max_Chi_J = 0.;

        vector<bool> max_TE1_found(doublets_size,false);
        vector<bool> max_TE2_found(triplets_size,false);
        bool maxChi_found = false;
        bool maxChi_doubled = false;

        double J = -dJ+0.01;
        vector<double> J_vec;
        vector<double> Chi_vec;
        vector<vector<double>> TE1_vec(doublets_size);
        vector<vector<double>> TE2_vec(triplets_size);

        int jj = 0;
        int j_maxChi_found = 0;

        while(found(max_TE1_found) || found(max_TE2_found) || !maxChi_doubled){

            J += dJ; 
            J_vec.push_back(J); 
            //cout << J << endl;
            vector<double> TE1(doublets_size,0.);
            vector<double> TE2(triplets_size,0.);
            double Chi = 0.;

            // Loop over N_trials:        
            #pragma omp parallel for reduction(+:Chi)
            for(int run=0;run<(int)N_trials;run++){            
                Mat<int> spins(T_evolve,N); //Save as matrix.
                vector<int> Spins(N);
                vector<double> M(T_evolve);//Magnetisation
                
                initializeSpins(Spins, N);
                relaxSpins(Spins, NN, J, N);

                random_device rd; // obtain a random number from hardware
                mt19937 gen(rd()); // seed the generator
                uniform_int_distribution<> distr(0, N-1); // uniform int distribution to select spin index
                uniform_real_distribution<> distr2(0.,1.);
                //evolve the spins in parallel and save them every time:

                // Initialize spin indices vector:
                double M_temp;
                vector<int> ind(N);
                iota(ind.begin(), ind.end(), 0);
                for(int t = 0; t < T_evolve_parallel; t++){
                    // Random permutation of spin indices
                    shuffle(ind.begin(), ind.end(), mt19937{random_device{}()});
                    M_temp = 0.;
                    // Loop over all spins out of permutated indices vector:
                    for(int k = 0; k < N;k++){
                        int spin_chosen = ind[k];                    
                        // Compute probability to flip spin:
                        double sumSpinsNeighbouring = 0.;
                        for(int i=0;i < static_cast<int>(NN[spin_chosen].size());i++)
                        {
                            if (Spins[ NN[spin_chosen][i] ] == 1) {
                            sumSpinsNeighbouring++;
                            }
                            else{sumSpinsNeighbouring--;}
                        }
                        int spin_chosen_value = -1;
                        if(Spins[spin_chosen]==1){spin_chosen_value = 1;}
                        double dE = 2.*spin_chosen_value*sumSpinsNeighbouring;                    
                        if(dE<=0.){//Flip spins:
                            if(spin_chosen_value == 1){
                                Spins[spin_chosen] = 0;
                                M_temp--;
                            }
                            else{
                                Spins[spin_chosen] = 1;
                                M_temp++;
                            }  
                        }
                        else{                
                            double p = exp(-J*dE);
                            // Generate random number between 0 and 1, flip if below p:
                            if(distr2(gen)< p){
                                if(spin_chosen_value == 1){Spins[spin_chosen] = 0;
                                M_temp--;
                                }
                                else{Spins[spin_chosen] = 1;
                                M_temp++;
                                }  
                            }
                            else{
                                if(spin_chosen_value==1){M_temp++;}
                                else{M_temp--;}    
                            }
                        }            
                    }     
                            
                    //Save values:
                    spins.row(t) = Row<int>(Spins); 
                    if(M_temp < 0){M_temp = -M_temp;}
                    M[t]=M_temp/static_cast<double>(N);  
                        
                }
                
                //Evolve the spins sequentially and save it every time:           
                
                // Loop for T time discrete steps and sequentially update:
                for(int t = 0; t < T_evolve_sequential; t++){
                    // Randomly choose a spin to update
                    int spin_chosen = distr(gen);

                    // Compute probability to flip spin:
                    double sumSpinsNeighbouring = 0.;
                    for(int i=0;i < static_cast<int>(NN[spin_chosen].size());i++)
                    {
                        if (Spins[NN[spin_chosen][i]] == 1) {
                        sumSpinsNeighbouring++;
                        }
                        else{sumSpinsNeighbouring--;}
                    }
                    int spin_chosen_value = -1;
                    if(Spins[spin_chosen]==1){spin_chosen_value = 1;}
                    double dE = 2.*spin_chosen_value*sumSpinsNeighbouring;                
                    if(dE<=0.){//Flip spins:
                            if(spin_chosen_value == 1){
                                Spins[spin_chosen] = 0;
                                M_temp--;
                            }
                            else{
                                Spins[spin_chosen] = 1;
                                M_temp++;
                            }  
                        }
                        else{                
                            double p = exp(-J*dE);
                            // Generate random number between 0 and 1, flip if below p:
                            if(distr2(gen)< p){
                                if(spin_chosen_value == 1){Spins[spin_chosen] = 0;
                                M_temp--;
                                }
                                else{Spins[spin_chosen] = 1;
                                M_temp++;
                                }  
                            }
                            else{
                                if(spin_chosen_value==1){M_temp++;}
                                else{M_temp--;}    
                            }
                        }
                    //printSpins(Spins);
                    spins.row(t+T_evolve_parallel) = Row<int>(Spins);  
                    if(M_temp < 0){M_temp = -M_temp;}
                    M[t+T_evolve_parallel]=M_temp/static_cast<double>(N);              

                }  // End of evolve loop
                
                // Compute Chi: 
                double mean_M_squared = 0.0;
                double mean_M = 0.0;
                for(int i=0;i < T_evolve;i++){
                    mean_M_squared += pow(M[i],2);
                    mean_M += M[i];
                }
                mean_M_squared = mean_M_squared/T_evolve;
                mean_M = mean_M/T_evolve;   
                Chi += (mean_M_squared - pow(mean_M,2))/static_cast<double>(N_trials);
                //cout << "Chi = " << Chi << endl;
                //spins.save(path+"0spins.mat", raw_ascii);
                //cout << spins << endl;

                //TE1 pairwise between doublets given by conjectures:            
                //#pragma omp parallel for schedule(dynamic)
                for(int count=0;count<doublets_size;count++){  
                    // Handle erros:
                    int driv = doublets[count][0];
                    int tar  = doublets[count][1];
                    if(driv < 0 || driv > N || tar < 0 || tar > N){
                        max_TE1[count] = -111.;
                        max_TE1_J[count] = -111.; // Signals that the indices are out of bound, thus that the corresponding procedure failed to find a doublet or triplet.
                        max_TE1_found[count] = true;
                        continue;
                    }


                    Col<int> driver = spins.col(driv);              
                    Col<int> target = spins.col(tar);  
                    vector<double> a_temp = a(driver,target);               
                    vector<double> b_temp = b(target);
                    vector<double> c_temp = c(driver,target); 
                    vector<double> d_temp = d(target);              

                    // Sum over all states. (use bi2de to get appropriate state label)
                    double TE_local = 0.;
                    for(int s_target1 = 0;s_target1<2;s_target1++){
                        for(int s_target0 = 0;s_target0<2;s_target0++){
                            for(int s_driver0 = 0;s_driver0<2;s_driver0++){
                                double a = a_temp[s_target1*4+s_target0*2+s_driver0];
                                double b = b_temp[s_target0];
                                double c = c_temp[s_target0*2+s_driver0];
                                double d = d_temp[s_target1*2+s_target0];
                                        
                                if(a*b*c*d>0.0){TE_local += a*log2( a*b/(c*d) );}                                                                 
                                        
                            }
                        }
                    } 
                    #pragma omp atomic update
                    TE1[count] += TE_local/static_cast<double>(N_trials);                                                           
                }// End of TE1 loop

                //Compute TE2's (of triplets):
                for(int count=0;count<triplets_size;count++){    
                    int driv1 = triplets[count][0];
                    int driv2 = triplets[count][1];    
                    int tar   = triplets[count][2];     

                    if(driv1 < 0 || driv1 > N || driv2 < 0 || driv2 > N || tar < 0 || tar > N){
                        max_TE2[count] = -111.;
                        max_TE2_J[count] = -111.; // Signals that the indices are out of bound, thus that the corresponding procedure failed to find a doublet or triplet.
                        max_TE2_found[count] = true;
                        continue;
                    }                    

                    Col<int> driver1 = spins.col(driv1); 
                    Col<int> driver2 = spins.col(driv2);                
                    Col<int> target = spins.col(tar);      

                    vector<double> a_temp = a(driver1,driver2,target);                           
                    vector<double> b_temp = b(target);
                    vector<double> c_temp = c(driver1,driver2,target);                
                    vector<double> d_temp = d(target);

                    double TE_local = 0.;
                    // Sum over all states. (use bi2de to get appropriate state label)
                    for(int s_target1 = 0;s_target1<2;s_target1++){
                        for(int s_target0 = 0;s_target0<2;s_target0++){
                            for(int s_driver0_1 = 0;s_driver0_1<2;s_driver0_1++){
                                for(int s_driver0_2 = 0;s_driver0_2<2;s_driver0_2++){
                                    double a = a_temp[s_target1*8+s_target0*4+s_driver0_1*2+s_driver0_2];
                                    double b = b_temp[s_target0];
                                    double c = c_temp[s_target0*4+s_driver0_1*2+s_driver0_2];
                                    double d = d_temp[s_target1*2+s_target0];                                            
                                    if(a*b*c*d>0.0){TE_local += a*log2( a*b/(c*d) );}                                                                                       
                                }
                            }
                        }
                    } 
                    #pragma omp atomic update              
                    TE2[count] += TE_local/static_cast<double>(N_trials);             
                }// End of TE2 loop
    

            }// End of runs loop
            
            // Update new maximums
            for(int count=0;count<doublets_size;count++){  
                TE1_vec[count].push_back(TE1[count]);//Add TE1's to TE1_vec
                if(max_TE1[count] < TE1[count]){            
                    max_TE1[count] = TE1[count];
                    max_TE1_J[count] = J;
                }    
                // Exit while loop when the peaks are found. 
                if(TE1[count] < threshold*max_TE1[count]){max_TE1_found[count] = true;}
            }
            for(int count=0;count<triplets_size;count++){  
                TE2_vec[count].push_back(TE2[count]);//Add TE2's to TE2_vec
                if(max_TE2[count] < TE2[count]){            
                    max_TE2[count] = TE2[count];
                    max_TE2_J[count] = J;
                }    
                // Exit while loop when the peaks are found. 
                if(TE2[count] < threshold*max_TE2[count]){max_TE2_found[count] = true;}
            }
            Chi_vec.push_back(Chi);
            if(max_Chi < Chi){            
                max_Chi = Chi;
                max_Chi_J = J;
                j_maxChi_found = jj;
            }

            jj++;
            if(jj == 2*j_maxChi_found+1){maxChi_doubled = true;}

            //cout << J << " " << jj << " " << j_maxChi_found << " " << maxChi_doubled << endl;
        }//End of J while loop

        // Compute mean, variance, skewness and kurtosis of each peak associated to each doublet, triplet (and Chi). 

        // Normalise our data:

        vector<double> Chi_vec_N = Chi_vec;
        vector<vector<double>> TE1_vec_N = TE1_vec;
        vector<vector<double>> TE2_vec_N = TE2_vec;

        normalise(Chi_vec_N);
        for(int count=0;count<doublets_size;count++){ 
            normalise(TE1_vec_N[count]);
        }
        for(int count=0;count<triplets_size;count++){ 
            normalise(TE2_vec_N[count]);
        }

        // compute means:
        double mean_Chi;
        vector<double> mean_TE1(doublets_size);
        vector<double> mean_TE2(triplets_size);

        mean(Chi_vec_N, J_vec, mean_Chi);
        for(int count=0;count<doublets_size;count++){ 
            mean(TE1_vec_N[count], J_vec, mean_TE1[count]);
        }
        for(int count=0;count<triplets_size;count++){ 
            mean(TE2_vec_N[count], J_vec, mean_TE2[count]);
        }        

        // compute variancese:
        double var_Chi;
        vector<double> var_TE1(doublets_size);
        vector<double> var_TE2(triplets_size);

        variance(Chi_vec_N, J_vec, mean_Chi, var_Chi);
        for(int count=0;count<doublets_size;count++){ 
            variance(TE1_vec_N[count], J_vec, mean_TE1[count], var_TE1[count]);
        }
        for(int count=0;count<triplets_size;count++){ 
            variance(TE2_vec_N[count], J_vec, mean_TE2[count], var_TE2[count]);
        }    

        // compute skewedness:
        double skew_Chi;
        vector<double> skew_TE1(doublets_size);
        vector<double> skew_TE2(triplets_size);

        skewness(Chi_vec_N, J_vec, mean_Chi, var_Chi, skew_Chi);
        for(int count=0;count<doublets_size;count++){ 
            skewness(TE1_vec_N[count], J_vec, mean_TE1[count], var_TE1[count], skew_TE1[count]);
        }
        for(int count=0;count<triplets_size;count++){ 
            skewness(TE2_vec_N[count], J_vec, mean_TE2[count], var_TE2[count], skew_TE2[count]);
        } 

        // compute kurtosis:
        double kurt_Chi;
        vector<double> kurt_TE1(doublets_size);
        vector<double> kurt_TE2(triplets_size);

        kurtosis(Chi_vec_N, J_vec, mean_Chi, var_Chi, kurt_Chi);
        for(int count=0;count<doublets_size;count++){ 
            skewness(TE1_vec_N[count], J_vec, mean_TE1[count], var_TE1[count], kurt_TE1[count]);
        }
        for(int count=0;count<triplets_size;count++){ 
            skewness(TE2_vec_N[count], J_vec, mean_TE2[count], var_TE2[count], kurt_TE2[count]);
        }   

        // Check if a process is writing and if its the case wait before writing, if not the case go ahead and write out cout     

        bool written_out = false;
        int max_writing;

        while(!written_out){
            MPI_Allreduce(&writing, &max_writing, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
            if(max_writing == 0){
                writing = 1; // signals we start writing

                // Write out data:

                //Print J values of maxima:
                cout << "max_Chi_J: " << endl << max_Chi_J << endl;
                cout  << "max_TE1_J: " << endl;
                for(int count=0;count<doublets_size;count++){ 
                    cout << max_TE1_J[count] << " ";
                }
                cout << endl;

                cout << "max_TE2_J: "  << endl;
                for(int count=0;count<triplets_size;count++){ 
                    cout << max_TE2_J[count] << " ";
                }
                cout << endl;

                //Print max values:
                cout << "max_Chi: " << endl << max_Chi << endl;
                cout  << "max_TE1: " << endl;
                for(int count=0;count<doublets_size;count++){ 
                    cout << max_TE1[count] << " ";
                }
                cout << endl;
                cout << "max_TE2: "  << endl;
                for(int count=0;count<triplets_size;count++){ 
                    cout << max_TE2[count] << " ";
                }
                cout << endl;

                //Print vars:
                cout << "var_Chi:" << endl << var_Chi << endl;
                cout  << "var_TE1: " << endl;
                for(int count=0;count<doublets_size;count++){ 
                    cout << var_TE1[count] << " ";
                }
                cout << endl;
                cout << "var_TE2: "  << endl;
                for(int count=0;count<triplets_size;count++){ 
                    cout << var_TE2[count] << " ";
                }
                cout << endl;                

                //Print skews:
                cout << "skew_Chi:" << endl << skew_Chi << endl;
                cout  << "skew_TE1: " << endl;
                for(int count=0;count<doublets_size;count++){ 
                    cout << skew_TE1[count] << " ";
                }
                cout << endl;
                cout << "skew_TE2: "  << endl;
                for(int count=0;count<triplets_size;count++){ 
                    cout << skew_TE2[count] << " ";
                }
                cout << endl; 

                //Print kurts:
                cout << "kurt_Chi:" << endl << kurt_Chi << endl;
                cout  << "kurt_TE1: " << endl;
                for(int count=0;count<doublets_size;count++){ 
                    cout << kurt_TE1[count] << " ";
                }
                cout << endl;
                cout << "kurt_TE2: "  << endl;
                for(int count=0;count<triplets_size;count++){ 
                    cout << kurt_TE2[count] << " ";
                }
                cout << endl; 

                writing = 0; // signals we stopped writing
                written_out = true;
            }
            else{
                // Sleep for 1 second
                this_thread::sleep_for(chrono::seconds(1));
            }
        }  


    
    }//End of network MPI loop.
    MPI_Finalize ();

    return 0;

}
