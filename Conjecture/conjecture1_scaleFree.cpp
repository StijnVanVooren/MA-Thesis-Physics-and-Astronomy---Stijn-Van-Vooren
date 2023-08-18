#include<boost/graph/adjacency_list.hpp>
#include<boost/graph/plod_generator.hpp>
#include<boost/random/linear_congruential.hpp>
#include<random>

#include<boost/graph/betweenness_centrality.hpp>
#include<algorithm>
#include<armadillo>
#include<cmath>
#include<vector>

#include<mpi.h> 

using namespace arma;
using namespace std;
int N_trials = 1500;
int T_relax_parallel  = 50000;
int T_relax_sequential= 10000;
int T_evolve_parallel = 50000;
int T_evolve_sequential = 0;
double dJ = 0.0025;

int T_evolve = T_evolve_parallel + T_evolve_sequential;
double T_evolve_double = static_cast<double>(T_evolve)-1.;

typedef boost::adjacency_list<boost::setS, boost::vecS, boost::undirectedS> Graph;
typedef boost::plod_iterator<boost::minstd_rand, Graph> SFGen;

Graph generate_scale_free_network(int N, double alpha, double beta, bool selfLoops)
{
    std::random_device rd;
    boost::minstd_rand gen(rd());
    Graph g(SFGen(gen, N, alpha, beta, selfLoops), SFGen(), N);
    return g;
}

void printSpins(const vector<int>& Spins)
{
    for (const auto& spin : Spins) {
        cout << spin << " ";
    }
    cout << endl;
    return;
}

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

std::vector<double> a(const arma::Col<int>&  driver, const arma::Col<int>&  target)
{
       
    std::vector<double> a(8,0.0); //Initialize vector with length total number of states (2^m since spin up or down).
    //Count frequency of each state (each state is labelled by a decimal number, found by taking spinstate as binary label and convert to decimal)
    for(int t = 0; t<T_evolve-1; t++){
        a[target[t+1]*4+target[t]*2+driver[t]]++; //Add 1 to frequency of state labeled bi2de(M[t])        
    }
    for(int j=0;j<8;j++){a[j] = a[j]/T_evolve_double;} //Normalise with total number of measurements.

    return a;
}

std::vector<double> b(const arma::Col<int>&  target)
{
       
    std::vector<double> b(2,0.0); //Initialize vector with length total number of states (2^m since spin up or down).
    //Count frequency of each state (each state is labelled by a decimal number, found by taking spinstate as binary label and convert to decimal)
    for(int t = 0; t<T_evolve-1; t++){
        b[target[t]]++; //Add 1 to frequency of state labeled bi2de(M[t])        
    }
    for(int j=0;j<2;j++){b[j] = b[j]/T_evolve_double;} //Normalise with total number of measurements.

    return b;
}


std::vector<double> c(const arma::Col<int>&  driver, const arma::Col<int>&  target)
{
       
    std::vector<double> c(4,0.0); //Initialize vector with length total number of states (2^m since spin up or down).
    //Count frequency of each state (each state is labelled by a decimal number, found by taking spinstate as binary label and convert to decimal)
    for(int t = 0; t<T_evolve-1; t++){
        c[target[t]*2+driver[t]]++; //Add 1 to frequency of state labeled bi2de(M[t])        
    }
    for(int j=0;j<4;j++){c[j] = c[j]/T_evolve_double;} //Normalise with total number of measurements.

    return c;
}


std::vector<double> d(const arma::Col<int>&  target)
{
       
    std::vector<double> d(4,0.0); //Initialize vector with length total number of states (2^m since spin up or down).
    //Count frequency of each state (each state is labelled by a decimal number, found by taking spinstate as binary label and convert to decimal)
    for(int t = 1; t<T_evolve; t++){
        d[target[t]*2+target[t-1]]++;             
    }
    for(int j=0;j<4;j++){d[j] = d[j]/T_evolve_double;} //Normalise with total number of measurements.

    return d;
}


using namespace std;

std::vector<double> scalefree_conjecture_1(int N, double alpha, double beta, bool selfLoops, double threshold)
{



    Graph g = generate_scale_free_network(N, alpha, beta, selfLoops);

    // Transform into nestled vector Nearest neighbours
    vector<vector<int>> NN(N);

    // loop over all vertices in the graph
    for (int i = 0; i < N; i++) {
        // loop over all neighbors of vertex i
        for (auto it = adjacent_vertices(i, g); it.first != it.second; ++it.first) {
            int neighbor_index = *it.first;
            NN[i].push_back(neighbor_index);
        }
    }

    // Define a vector to hold the betweenness centrality values for each vertex
    std::vector<double> centrality(N);

    // Compute the betweenness centrality
    boost::brandes_betweenness_centrality(g, boost::centrality_map(boost::make_iterator_property_map(centrality.begin(), boost::get(boost::vertex_index, g))));



    int max_index = std::distance(centrality.begin(), std::max_element(centrality.begin(), centrality.end()));

    // Find centrality of neighbours and find max neighbors
    double max_centrality = -1.0;
    int max_neighbor = -1;

    for (int neighbor : NN[max_index]) {
        if (centrality[neighbor] > max_centrality) {
            max_centrality = centrality[neighbor];
            max_neighbor = neighbor;
        }
    }
    //cout << max_neighbor << endl;
    int d1 = max_neighbor;
    int t1 = max_index;

    int d2 = max_index;
    int t2 = max_neighbor;

    double max_TE1 = 0.;
    double max_TE2 = 0.;
    double max_Chi = 0.;

    double max_TE1_J = 0.;
    double max_TE2_J = 0.;
    double max_Chi_J = 0.;

    bool maxTE1_found = false;
    bool maxTE2_found = false;
    bool maxChi_found = false;

    double J = -dJ+0.01;
    while(!maxTE1_found || !maxTE2_found || !maxChi_found){

        J += dJ; 
        //std::cout << J << std::endl;
        double TE1 = 0.;
        double TE2 = 0.;
        double Chi = 0.;

        // Loop over N_trials:        
        #pragma omp parallel for reduction(+:Chi)
        for(int run=0;run<(int)N_trials;run++){            
            arma::Mat<int> spins(T_evolve,N); //Save as matrix.
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
                spins.row(t) = arma::Row<int>(Spins); 
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
                spins.row(t+T_evolve_parallel) = arma::Row<int>(Spins);  
                if(M_temp < 0){M_temp = -M_temp;}
                M[t+T_evolve_parallel]=M_temp/static_cast<double>(N);              

            }  
            // Compute Chi: 
            double mean_M_squared = 0.0;
            double mean_M = 0.0;
            for(int i=0;i < T_evolve;i++){
                mean_M_squared += std::pow(M[i],2);
                mean_M += M[i];
            }
            mean_M_squared = mean_M_squared/T_evolve;
            mean_M = mean_M/T_evolve;   
            Chi += mean_M_squared - std::pow(mean_M,2);
            //std::cout << "Chi = " << Chi << std::endl;
            //spins.save(path+"0spins.mat", arma::raw_ascii);
            //std::cout << spins << std::endl;

            //TE pairwise between nodes given by conjecture 1:            
            //#pragma omp parallel for schedule(dynamic)
            for(int count=0;count<2;count++){    
                int driv,tar;
                if(count == 0 ){
                    if(maxTE1_found){continue;}
                    driv = d1;
                    tar = t1;}
                else{
                    if(maxTE2_found){continue;}
                    driv = d2;
                    tar = t2;}

                arma::Col<int> driver = spins.col(driv);              
                arma::Col<int> target = spins.col(tar);  
                std::vector<double> a_temp = a(driver,target);               
                std::vector<double> b_temp = b(target);
                std::vector<double> c_temp = c(driver,target); 
                std::vector<double> d_temp = d(target);              

                // Sum over all states. (use bi2de to get appropriate state label)
                double TE_local = 0.;
                for(int s_target1 = 0;s_target1<2;s_target1++){
                    for(int s_target0 = 0;s_target0<2;s_target0++){
                        for(int s_driver0 = 0;s_driver0<2;s_driver0++){
                            double a = a_temp[s_target1*4+s_target0*2+s_driver0];
                            double b = b_temp[s_target0];
                            double c = c_temp[s_target0*2+s_driver0];
                            double d = d_temp[s_target1*2+s_target0];
                                    
                            if(a*b*c*d>0.0){TE_local += a*std::log2( a*b/(c*d) );}                                                                 
                                    
                        }
                    }
                } 
                if(count == 0){
                    #pragma omp atomic update
                    TE1 += TE_local;
                }
                else{
                    #pragma omp atomic update
                    TE2 += TE_local;                    
                }
                                                           
            }                         
        }
        // Update new maximums
        if(max_TE1 < TE1){            
            max_TE1 = TE1;
            max_TE1_J = J;
        }    
        if(max_TE2 < TE2){            
            max_TE2 = TE2;
            max_TE2_J = J;
        }  
        if(max_Chi < Chi){            
            max_Chi = Chi;
            max_Chi_J = J;
        }
        // Exit while loop when the peaks are found. 
        if(TE1 < threshold*max_TE1){maxTE1_found = true;}
        if(TE2 < threshold*max_TE2){maxTE2_found = true;}
        if(Chi < threshold*max_Chi){maxChi_found = true;}


    }         

    std::vector result = {max_TE1_J,max_TE2_J,max_Chi_J};



    /*
    // Print the betweenness centrality values for each vertex
    cout << "Betweenness centrality values:" << endl;
    for (int i = 0; i < centrality.size(); ++i) {
        cout << "Vertex " << i << ": " << centrality[i] << endl;
    }
    
    std::cout << "Neighbors of node " << max_index << ": ";
    for (auto neighbor : NN[max_index]) {
        std::cout << neighbor << " ";
    }
    std::cout << std::endl;

    cout << "Vertex " << max_index <<  endl;

    // print the NN list
    for (int i = 0; i < N; i++) {
        cout << i << ": ";
        for (int j = 0; j < NN[i].size(); j++) {
            cout << NN[i][j] << " ";
        }
        cout << endl;
    }

    // Print adjacency list
    for (int i = 0; i < N; i++) {
        std::cout << i << ": ";
        for (auto it = adjacent_vertices(i, g); it.first != it.second; ++it.first) {
            std::cout << *it.first << " ";
        }
        std::cout << std::endl;
    }
    */
    return result;
}



int main(int argc, char *argv[])
{


    int N = 100;
    double alpha = 1.5;
    double beta = 1000;
    bool selfLoops = false;
    double threshold = 0.8;
    int networks_to_test = 230;

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

    
    for(int j=start;j<end;j++){  

        std::vector result = scalefree_conjecture_1(N,  alpha,  beta,  selfLoops,  threshold);
        for(int i=0;i<3;i++){
            std::cout << j << " " << i << " " << result[i] << std::endl;
        }

    }


    MPI_Finalize ();
    

    return 0;
}
