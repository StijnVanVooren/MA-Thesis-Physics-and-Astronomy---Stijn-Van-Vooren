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
#include<thread>
#include<chrono>

using namespace arma;
using namespace std;

int max_driver = 7;
int max_target = max_driver;

int N_trials = 1000;
int T_relax_parallel  = 30000;
int T_relax_sequential= 10000;
int T_evolve_parallel = 30000;
int T_evolve_sequential = 0;
double dJ = 0.015;

// Comment out the paths except the one where you want to save the results:
//string path = "/home/stijn/Documents/ThesisNew/FullSim/Data/BinaryTrees/";
string path =  "/theia/scratch/brussel/107/vsc10773/Data/BinaryTrees/";
//string path =  "/kyukon/scratch/gent/459/vsc45915/Data/BinaryTrees/";

int T_evolve = T_evolve_parallel + T_evolve_sequential;
double T_evolve_double = static_cast<double>(T_evolve)-1.;


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

void saveVectorToFile(const vector<double>& data, const string& filename) {
    ofstream outputFile(path+filename);

    if (outputFile.is_open()) {
        for (const auto& element : data) {
            outputFile << element << " ";
        }
        cout << endl;

        outputFile.close();
    } else {
        cerr << "Unable to open file: " << filename << endl;
    }
}

void max(vector<double> vec, double &max_y, double &max_x){

    double max_ytemp = vec[0];
    double max_xtemp = 0; 

    for(int i=0;i < static_cast<int>(vec.size());i++){
        if(vec[i] > max_ytemp){
            max_ytemp = vec[i];
            max_xtemp = i*dJ;
        }
    }

    max_y = max_ytemp;
    max_x = max_xtemp;

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

vector<double> a(const arma::Col<int>&  driver, const arma::Col<int>&  target)
{
       
    vector<double> a(8,0.0); //Initialize vector with length total number of states (2^m since spin up or down).
    //Count frequency of each state (each state is labelled by a decimal number, found by taking spinstate as binary label and convert to decimal)
    for(int t = 0; t<T_evolve-1; t++){
        a[target[t+1]*4+target[t]*2+driver[t]]++; //Add 1 to frequency of state labeled bi2de(M[t])        
    }
    for(int j=0;j<8;j++){a[j] = a[j]/T_evolve_double;} //Normalise with total number of measurements.

    return a;
}

vector<double> a(const arma::Col<int>&  driver1,const arma::Col<int>&  driver2, const arma::Col<int>&  target)
{
       
    vector<double> a(16,0.0); //Initialize vector with length total number of states (2^m since spin up or down).
    //Count frequency of each state (each state is labelled by a decimal number, found by taking spinstate as binary label and convert to decimal)
    for(int t = 0; t<T_evolve-1; t++){
        a[target[t+1]*8+target[t]*4+driver1[t]*2+driver2[t]]++; //Add 1 to frequency of state labeled bi2de(M[t])        
    }
    for(int j=0;j<16;j++){a[j] = a[j]/T_evolve_double;} //Normalise with total number of measurements.

    return a;
}

vector<double> b(const arma::Col<int>&  target)
{
       
    vector<double> b(2,0.0); //Initialize vector with length total number of states (2^m since spin up or down).
    //Count frequency of each state (each state is labelled by a decimal number, found by taking spinstate as binary label and convert to decimal)
    for(int t = 0; t<T_evolve-1; t++){
        b[target[t]]++; //Add 1 to frequency of state labeled bi2de(M[t])        
    }
    for(int j=0;j<2;j++){b[j] = b[j]/T_evolve_double;} //Normalise with total number of measurements.

    return b;
}


vector<double> c(const arma::Col<int>&  driver, const arma::Col<int>&  target)
{
       
    vector<double> c(4,0.0); //Initialize vector with length total number of states (2^m since spin up or down).
    //Count frequency of each state (each state is labelled by a decimal number, found by taking spinstate as binary label and convert to decimal)
    for(int t = 0; t<T_evolve-1; t++){
        c[target[t]*2+driver[t]]++; //Add 1 to frequency of state labeled bi2de(M[t])        
    }
    for(int j=0;j<4;j++){c[j] = c[j]/T_evolve_double;} //Normalise with total number of measurements.

    return c;
}

vector<double> c(const arma::Col<int>&  driver1,const arma::Col<int>&  driver2, const arma::Col<int>&  target)
{
       
    vector<double> c(8,0.0); //Initialize vector with length total number of states (2^m since spin up or down).
    //Count frequency of each state (each state is labelled by a decimal number, found by taking spinstate as binary label and convert to decimal)
    for(int t = 0; t<T_evolve-1; t++){
        c[target[t]*4+driver1[t]*2+driver2[t]]++; //Add 1 to frequency of state labeled bi2de(M[t])        
    }
    for(int j=0;j<8;j++){c[j] = c[j]/T_evolve_double;} //Normalise with total number of measurements.

    return c;
}

vector<double> d(const arma::Col<int>&  target)
{
       
    vector<double> d(4,0.0); //Initialize vector with length total number of states (2^m since spin up or down).
    //Count frequency of each state (each state is labelled by a decimal number, found by taking spinstate as binary label and convert to decimal)
    for(int t = 1; t<T_evolve; t++){
        d[target[t]*2+target[t-1]]++;             
    }
    for(int j=0;j<4;j++){d[j] = d[j]/T_evolve_double;} //Normalise with total number of measurements.

    return d;
}


int main(int argc, char *argv[])
{  


    //Loop over J:
    //Initialize MPI parallellisation:
    MPI_Init(&argc, &argv);

    int num_procs, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);   
    int chunk_size;
    if(max_driver % num_procs == 0){chunk_size = max_driver/num_procs;} 
    else{chunk_size = max_driver/num_procs+1;} 
    const int start =  rank * chunk_size;
    const int end = start + chunk_size;

    int writing = 0; // Signals if the process is writing or not. 

    for(int c1=start; c1<end;c1++){
        for(int c2=c1; c2<max_target;c2++){
            int N = c1+c2+2;
            vector<vector<int>> NN(N);
            NN[0].push_back(1);
            NN[1].push_back(0);
           
            for(int ind1=2;ind1<c1+2;ind1++){
                NN[0].push_back(ind1);
                NN[ind1].push_back(0);
            }
            for(int ind2=c1+2;ind2<c2+c1+2;ind2++){
                NN[1].push_back(ind2);
                NN[ind2].push_back(1);
            }
            vector<double> TE1_;
            vector<double> TE2_;            
            vector<double> Chi_;
            int j=0;
            bool doneTE1 = false;
            bool doneTE2 = false;
            double maxTE1 = 0.;
            double maxTE2 = 0.;
            double J = 0.;
            while(!doneTE1 || !doneTE2 || J < 0.3){  
                j++;
                // Loop over N_trials:
                double Chi = 0.;
                double TE1 = 0.;
                double TE2 = 0.;
                J = j*dJ;

                #pragma omp parallel for reduction(+:Chi,TE1,TE2)
                for(int run=0;run<(int)N_trials;run++){            
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
                    arma::Col<int> spin_site_0(T_evolve);
                    arma::Col<int> spin_site_1(T_evolve);
                    
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
                        spin_site_0[t] = Spins[0];
                        spin_site_1[t] = Spins[1];            
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
                        spin_site_0[t+T_evolve_parallel] = Spins[0];
                        spin_site_1[t+T_evolve_parallel] = Spins[1];  
                        if(M_temp < 0){M_temp = -M_temp;}
                        M[t+T_evolve_parallel]=M_temp/static_cast<double>(N);              

                    }  
                    // Compute Chi: 
                    double mean_M_squared = 0.0;
                    double mean_M = 0.0;
                    for(int i=0;i < T_evolve;i++){
                        mean_M_squared += pow(M[i],2);
                        mean_M += M[i];
                    }
                    mean_M_squared = mean_M_squared/T_evolve;
                    mean_M = mean_M/T_evolve;   
                    Chi += mean_M_squared - pow(mean_M,2);

                    //TE1:        
                    

                    arma::Col<int> target = spin_site_1;                
                    vector<double> b_temp = b(target);
                    vector<double> d_temp = d(target);

                    arma::Col<int> driver = spin_site_0;                                                         
                    // Compute probabilities    
                    vector<double> a_temp = a(driver,target);                        
                    vector<double> c_temp = c(driver,target);                        
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
                    TE1 += TE_local/static_cast<double>(N_trials);                                 

                    //TE2:        
                    

                    target = spin_site_0;                
                    b_temp = b(target);
                    d_temp = d(target);

                    driver = spin_site_1;                                                         
                    // Compute probabilities    
                    a_temp = a(driver,target);                        
                    c_temp = c(driver,target);                        
                    // Sum over all states. (use bi2de to get appropriate state label)
                    TE_local = 0.;
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
                    TE2 += TE_local/static_cast<double>(N_trials);  

                }//END OF run LOOP
                TE1_.push_back(TE1);
                TE2_.push_back(TE2);
                Chi_.push_back(Chi);

                if(TE1 > maxTE1){maxTE1 = TE1;}
                if(TE2 > maxTE2){maxTE2 = TE2;}
                if(TE1 <= maxTE1*0.5){doneTE1 = true;}
                if(TE2 <= maxTE2*0.5){doneTE2 = true;}

            }//End of j loop. 
            
            saveVectorToFile(TE1_,"BinaryTree_TE1_"+to_string(c1)+"_"+to_string(c2));
            saveVectorToFile(TE2_,"BinaryTree_TE2_"+to_string(c1)+"_"+to_string(c2));
            saveVectorToFile(Chi_,"BinaryTree_Chi_"+to_string(c1)+"_"+to_string(c2));
            cout << c1 << " , " << c2 << " finished" << endl;
            /*
            bool written_out = false;
            int max_writing;

            while(!written_out){
                MPI_Allreduce(&writing, &max_writing, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
                if(max_writing == 0){
                    writing = 1; // signals we start writing

                    // Write out data:
                    cout << c1 << " " << c2 << endl;
                    //Print J values of maxima:
                    cout << "max_Chi_J: " << endl << max_Chi_J << endl;
                    cout << "max_Chi: " << endl << max_Chi << endl;
                    cout  << "max_TE1_J: " << endl << max_TE1_J << endl;
                    cout  << "max_TE1: " << endl << max_TE1 << endl;
                    cout  << "max_TE2_J: " << endl << max_TE2_J << endl;
                    cout  << "max_TE2: " << endl << max_TE2 << endl;


                    writing = 0; // signals we stopped writing
                    written_out = true;
                }
                else{
                    // Sleep for 0.5 second
                    this_thread::sleep_for(chrono::milliseconds(500));
                }
            }  
            */

        }        
    }

    return 0;
}