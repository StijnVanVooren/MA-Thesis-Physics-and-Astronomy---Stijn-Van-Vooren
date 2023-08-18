#include<vector>
#include<fstream>
#include<iostream>
#include<string>
#include<random>
#include<algorithm>
#include<chrono>
#include<cmath>
#include<mpi.h> 
#include<armadillo>

using namespace arma;
using namespace std;

//std::vector<std::string> networknames = {"Local10","Level11","Local12","Local13","Local14","Local16","Local17"};
std::string networkname = "Lattice2";

int N_trials = 5000;
int T_relax_parallel  = 50000;
int T_relax_sequential= 10000;
int T_evolve_parallel = 60000;
int T_evolve_sequential = 0;
double J_min = 0.01;
double J_max = 1.2;
int J_res = 75;

// Comment out the paths except the one where you want to save the results:
//std::string path = "/home/stijn/Documents/ThesisNew/FullSim/Data/"+networkname+"/";
std::string path =  "/theia/scratch/brussel/107/vsc10773/Data/"+networkname+"/";
//std::string path = "/kyukon/scratch/gent/459/vsc45915/Data/"+networkname+"/";
std::string pathtemp = path + "temp/";

int T_evolve = T_evolve_parallel + T_evolve_sequential;
double T_evolve_double = static_cast<double>(T_evolve)-1.;


vector<vector<int>> LoadA(const string& filename, int &N){
    ifstream iFile(path+filename+"_A.txt");
    // Check if the files opened:
    if(!iFile.is_open()){cout<< "file unable to open";}
    // Get matrix dimensions:
    int m,n;
    iFile >> m >> n;
    N = m;
    //cout << m << ", " << n << endl;
    vector<vector<int>> A(m,vector<int>(n));
    if (m <= 0 || n <= 0) {
            cout << "Error: Invalid matrix dimensions." << endl;
    }
    for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {//cout << "I work untill here" << endl;
                iFile >> A[i][j];
                //if (!(iFile >> A[i][j])) {
                    //cout << "Error: Unable to read matrix element " << endl;
                //}
            }
        }

    vector<vector<int>> NN(N);

    // Compute neighbouring spins for each spinsite:
    for(int i=0;i<N;i++){
        for(int j=0;j<N;j++){
            for(int k=0;k<A[i][j];k++){
                NN[i].push_back(j);            
            }
            //cout << i<<","<<j<<endl;
            
        }
    }

    return NN;
}

void printNN(const vector<vector<int>>& NN) {
    cout << "NN: " << endl;
    for (const auto& row : NN) {
        cout << "Site " << &row - &NN[0] << ": ";
        for (const auto& neighbour : row) {
            cout << neighbour << " ";
        }
        cout << endl;
    }
    return;
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

std::vector<double> a(const arma::Col<int>&  driver1,const arma::Col<int>&  driver2, const arma::Col<int>&  target)
{
       
    std::vector<double> a(16,0.0); //Initialize vector with length total number of states (2^m since spin up or down).
    //Count frequency of each state (each state is labelled by a decimal number, found by taking spinstate as binary label and convert to decimal)
    for(int t = 0; t<T_evolve-1; t++){
        a[target[t+1]*8+target[t]*4+driver1[t]*2+driver2[t]]++; //Add 1 to frequency of state labeled bi2de(M[t])        
    }
    for(int j=0;j<16;j++){a[j] = a[j]/T_evolve_double;} //Normalise with total number of measurements.

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

std::vector<double> c(const arma::Col<int>&  driver1,const arma::Col<int>&  driver2, const arma::Col<int>&  target)
{
       
    std::vector<double> c(8,0.0); //Initialize vector with length total number of states (2^m since spin up or down).
    //Count frequency of each state (each state is labelled by a decimal number, found by taking spinstate as binary label and convert to decimal)
    for(int t = 0; t<T_evolve-1; t++){
        c[target[t]*4+driver1[t]*2+driver2[t]]++; //Add 1 to frequency of state labeled bi2de(M[t])        
    }
    for(int j=0;j<8;j++){c[j] = c[j]/T_evolve_double;} //Normalise with total number of measurements.

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






int main(int argc, char *argv[])
{   
    // Define variables for the simulation:
    int N;
    vector<vector<int>> NN = LoadA(networkname,N); // Load nearest neighbours and N;
    printNN(NN);
        
    string ExtraInfo = "Ran on PC. Metropolis-Hasting.";

    // Write the parameters used to a file with INFO at the end of the name.
    ofstream oFile(path+"INFO.txt");
    // Check if the files opened:
    if(!oFile.is_open()){cout<< "file unable to open";}
    // Write parameters to this file (in python format so we can load easily to plot)
    oFile << "N:" << endl << N << endl << endl;
    oFile << "N_trials:" << endl << N_trials << endl << endl;
    oFile << "T_relax_parallel:" << endl << T_relax_parallel << endl << endl;
    oFile << "T_relax_sequential:" << endl << T_relax_sequential << endl << endl;
    oFile << "T_evolve:" << endl << T_evolve << endl << endl;
    oFile << "T_evolve_parallel:" << endl << T_evolve_parallel << endl << endl;
    oFile << "T_evolve_sequential:" << endl << T_evolve_sequential << endl << endl;
    oFile << "J_min:" << endl << J_min << endl << endl;
    oFile << "J_max:" << endl << J_max << endl << endl;
    oFile << "J_res:" << endl << J_res << endl << endl;

    // Close txt files:
    oFile.close(); 

    //Start of simulations here:        

    double dJ = (J_max-J_min)/(J_res-1.0);
    

    //Loop over J:
    //Initialize MPI parallellisation:
    MPI_Init(&argc, &argv);

    int num_procs, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);   
    int chunk_size;
    if(J_res % num_procs == 0){chunk_size = J_res/num_procs;} 
    else{chunk_size = J_res/num_procs+1;} 
    const int start =  rank * chunk_size;
    const int end = start + chunk_size;

    
    for(int j=start;j<end;j++){  
        if(j>=J_res){continue;} 

        // Get current time (to time the simulation)
        auto start_time = chrono::high_resolution_clock::now();

        Mat<double> TE1(N,N,fill::zeros);
        Cube<double> TE2(N,N,N,fill::zeros);
        
        double J = J_min + j*dJ;
        // Loop over N_trials:
        double Chi = 0.;
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

            //TE1:            
            //#pragma omp parallel for schedule(dynamic)
            for(int tar=0;tar<N;tar++){                
                arma::Col<int> target = spins.col(tar);                
                std::vector<double> b_temp = b(target);
                std::vector<double> d_temp = d(target);
                for(int driv=0;driv<N;driv++){                
                    if(tar==driv){continue;}
                    else{   
                        arma::Col<int> driver = spins.col(driv);                                                         
                        // Compute probabilities    
                        std::vector<double> a_temp = a(driver,target);                        
                        std::vector<double> c_temp = c(driver,target);                        

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
                        #pragma omp atomic update
                        TE1(driv,tar) += TE_local/static_cast<double>(N_trials); 
                        //if(driv == 1 && tar == 3){std::cout << "TE1 = " << "driv1" << TE1(driv,tar) << std::endl; }  
                        //if(driv == 2 && tar == 3){std::cout << "TE1 = " << TE1(driv,tar) << std::endl; }                                            
                    }                         
                }
            } 
            

            //TE2:
            for(int tar=0;tar<N;tar++){                
                arma::Col<int> target = spins.col(tar);                
                std::vector<double> b_temp = b(target);
                std::vector<double> d_temp = d(target);
                
                for(int driv1=0;driv1<N;driv1++){    
                    if(tar==driv1){continue;}
                    arma::Col<int> driver1 = spins.col(driv1);
                    for(int driv2 = driv1+1; driv2<N;driv2++){                                    
                        if(tar==driv2){continue;}                       
                        else{
                            arma::Col<int> driver2 = spins.col(driv2);                                                                
                            // Compute probabilities    
                            std::vector<double> a_temp = a(driver1,driver2,target);                        
                            std::vector<double> c_temp = c(driver1,driver2,target);                        
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
                                            if(a*b*c*d>0.0){TE_local += a*std::log2( a*b/(c*d) );}                                                                                       
                                        }
                                    }
                                }
                            } 
                            #pragma omp atomic update
                            TE2(driv1,driv2,tar) += TE_local/static_cast<double>(N_trials);   
                            #pragma omp atomic update
                            TE2(driv2,driv1,tar) += TE_local/static_cast<double>(N_trials);  
                            //if(driv1 == 1 && driv2 == 2 && tar == 3){std::cout << "TE2 = " << TE2(driv1,driv2,tar) << std::endl; }                                           
                        }                         
                    }
                }    
            }

        }//END OF run LOOP
        
        //Save TE1,TE2
        TE1.save(pathtemp+"TE1_"+std::to_string(j)+".bin");
        TE2.save(pathtemp+"TE2_"+std::to_string(j)+".bin");
        //PID:
        Cube<double> S_T(N,N,N,fill::zeros);
        Cube<double> R_T(N,N,N,fill::zeros);

        for(int d1=0; d1<N; d1++) {
            for(int t=0; t<N; t++) {       
                double te1 = TE1(d1,t);     
                for(int d2=d1+1; d2<N; d2++) {
                    double te2 = TE1(d2,t);
                    double te12 = TE2(d1,d2,t);
                    if(te1<te2){
                        R_T(d1,d2,t) = te1;
                        R_T(d2,d1,t) = te1;
                        S_T(d1,d2,t) = te12-te2;
                        S_T(d2,d1,t) = te12-te2;
                    }
                    else{
                        R_T(d1,d2,t) = te2;
                        R_T(d2,d1,t) = te2;
                        S_T(d1,d2,t) = te12-te1;
                        S_T(d2,d1,t) = te12-te1;
                    }                    
                }
            }
        }
        //Save PID
        S_T.save(pathtemp+"ST_"+std::to_string(j)+".bin");
        R_T.save(pathtemp+"RT_"+std::to_string(j)+".bin");
        //Save Chi
        std::ofstream outfile(pathtemp+"Chi_"+std::to_string(j)+".txt");
        outfile << Chi;
        outfile.close();

        //End benchmark timing:
        // Get current time
        auto end_time = chrono::high_resolution_clock::now();

        // Calculate duration of function call
        auto duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);

        auto runtime = duration.count();
        // Output the duration in milliseconds
        cout << "Duration loop rank = " << rank << ", j = " << j << ": " << runtime << " ms" << endl;          
        
        
    }//END OF j LOOP
    


    MPI_Finalize ();


    
    return 0;
}