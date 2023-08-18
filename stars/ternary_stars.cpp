#include<vector>
#include<fstream>
#include<iostream>
#include<string>
#include<random>
#include<algorithm>
#include<cmath>
#include<mpi.h> 
#include<thread>


using namespace std;

int N_trials = 1500;
int T_relax_parallel  = 50000;
int T_relax_sequential= 10000;
int T_evolve_parallel = 40000;
int T_evolve_sequential = 0;
double dJ = 0.005;

// Comment out the paths except the one where you want to save the results:
//string path = "/home/stijn/Documents/ThesisNew/stars/Data/TernaryStars/";
string path =  "/dodrio/scratch/projects/starting_2023_066/Data/";
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
        outputFile.close();
    } else {
        cerr << "Unable to open file: " << filename << endl;
    }
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

vector<double> a(const int *driver, const int *target)
{
       
    vector<double> a(8,0.0); //Initialize vector with length total number of states (2^m since spin up or down).
    //Count frequency of each state (each state is labelled by a decimal number, found by taking spinstate as binary label and convert to decimal)
    for(int t = 0; t<T_evolve-1; t++){
        a[target[t+1]*4+target[t]*2+driver[t]]++; //Add 1 to frequency of state labeled bi2de(M[t])        
    }
    for(int j=0;j<8;j++){a[j] = a[j]/T_evolve_double;} //Normalise with total number of measurements.

    return a;
}

vector<double> a(const int *driver1,const int *driver2, const int *target)
{
       
    vector<double> a(16,0.0); //Initialize vector with length total number of states (2^m since spin up or down).
    //Count frequency of each state (each state is labelled by a decimal number, found by taking spinstate as binary label and convert to decimal)
    for(int t = 0; t<T_evolve-1; t++){
        a[target[t+1]*8+target[t]*4+driver1[t]*2+driver2[t]]++; //Add 1 to frequency of state labeled bi2de(M[t])        
    }
    for(int j=0;j<16;j++){a[j] = a[j]/T_evolve_double;} //Normalise with total number of measurements.

    return a;
}

vector<double> b(const int *target)
{
       
    vector<double> b(2,0.0); //Initialize vector with length total number of states (2^m since spin up or down).
    //Count frequency of each state (each state is labelled by a decimal number, found by taking spinstate as binary label and convert to decimal)
    for(int t = 0; t<T_evolve-1; t++){
        b[target[t]]++; //Add 1 to frequency of state labeled bi2de(M[t])        
    }
    for(int j=0;j<2;j++){b[j] = b[j]/T_evolve_double;} //Normalise with total number of measurements.

    return b;
}


vector<double> c(const int *driver, const int *target)
{
       
    vector<double> c(4,0.0); //Initialize vector with length total number of states (2^m since spin up or down).
    //Count frequency of each state (each state is labelled by a decimal number, found by taking spinstate as binary label and convert to decimal)
    for(int t = 0; t<T_evolve-1; t++){
        c[target[t]*2+driver[t]]++; //Add 1 to frequency of state labeled bi2de(M[t])        
    }
    for(int j=0;j<4;j++){c[j] = c[j]/T_evolve_double;} //Normalise with total number of measurements.

    return c;
}

vector<double> c(const int *driver1,const int *driver2, const int *target)
{
       
    vector<double> c(8,0.0); //Initialize vector with length total number of states (2^m since spin up or down).
    //Count frequency of each state (each state is labelled by a decimal number, found by taking spinstate as binary label and convert to decimal)
    for(int t = 0; t<T_evolve-1; t++){
        c[target[t]*4+driver1[t]*2+driver2[t]]++; //Add 1 to frequency of state labeled bi2de(M[t])        
    }
    for(int j=0;j<8;j++){c[j] = c[j]/T_evolve_double;} //Normalise with total number of measurements.

    return c;
}

vector<double> d(const int *target)
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


    int max_driver_1 = 15;
    int max_driver_2 = max_driver_1;
    int max_target = max_driver_1;

    int c1c2=max_driver_1*max_driver_1-rank-1;

    while(c1c2 >= 0){

        int c1 = c1c2/max_driver_1;
        int c2 = c1c2%max_driver_1;
    
            for(int c3=c1; c3<max_target;c3++){

                int N = c1+c2+c3+3;
                vector<vector<int>> NN(N);

                // Connect 0 to 1
                NN[0].push_back(1);
                NN[1].push_back(0);

                // Connect 1 to 2
                NN[1].push_back(2);
                NN[2].push_back(1);
            
                for(int ind1=3;ind1<c1+3;ind1++){
                    NN[0].push_back(ind1);
                    NN[ind1].push_back(0);
                }
                for(int ind2=c1+3;ind2<c2+c1+3;ind2++){
                    NN[1].push_back(ind2);
                    NN[ind2].push_back(1);
                }
                for(int ind3=c2+c1+3;ind3<c3+c2+c1+3;ind3++){
                    NN[2].push_back(ind3);
                    NN[ind3].push_back(2);
                }

                vector<double> TE01_;
                vector<double> TE02_;  
                vector<double> TE10_;
                vector<double> TE12_;  
                vector<double> TE20_;
                vector<double> TE21_;    

                vector<double> TE0_; 
                vector<double> TE1_; 
                vector<double> TE2_; 


                int j=0;

                bool doneTE01 = false;
                bool doneTE02 = false;
                bool doneTE10 = false;
                bool doneTE12 = false;
                bool doneTE20 = false;
                bool doneTE21 = false;

                bool doneTE0 = false;
                bool doneTE1 = false;
                bool doneTE2 = false;

                double maxTE01 = 0.;
                double maxTE02 = 0.;
                double maxTE10 = 0.;
                double maxTE12 = 0.;
                double maxTE20 = 0.;
                double maxTE21 = 0.;

                double maxTE0 = 0.;
                double maxTE1 = 0.;
                double maxTE2 = 0.;
                
                double J = 0.;
                while(!doneTE1 || !doneTE01 || !doneTE02 || !doneTE10 || !doneTE12 || !doneTE20|| !doneTE21 || !doneTE0|| !doneTE1|| !doneTE2){  
                    j++;
                    // Loop over N_trials:
                    double TE01=0.;
                    double TE02=0.;  
                    double TE10=0;
                    double TE12=0.;  
                    double TE20=0.;
                    double TE21=0.;    

                    double TE0=0.; 
                    double TE1=0.; 
                    double TE2=0.;
                    J = j*dJ;

                    #pragma omp parallel for reduction(+:TE01,TE02,TE10,TE12,TE20,TE21,TE0,TE1,TE2)
                    for(int run=0;run<(int)N_trials;run++){            
                        vector<int> Spins(N);
                        
                        initializeSpins(Spins, N);
                        relaxSpins(Spins, NN, J, N);


                        random_device rd; // obtain a random number from hardware
                        mt19937 gen(rd()); // seed the generator
                        uniform_int_distribution<> distr(0, N-1); // uniform int distribution to select spin index
                        uniform_real_distribution<> distr2(0.,1.);
                        //evolve the spins in parallel and save them every time:

                        // Initialize spin indices vector:
                        int spin_site_0[T_evolve];
                        int spin_site_1[T_evolve];
                        int spin_site_2[T_evolve];
                        
 
                        vector<int> ind(N);
                        iota(ind.begin(), ind.end(), 0);
                        for(int t = 0; t < T_evolve_parallel; t++){
                            // Random permutation of spin indices
                            shuffle(ind.begin(), ind.end(), mt19937{random_device{}()});
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
                                    }
                                    else{
                                        Spins[spin_chosen] = 1;                                      
                                    }  
                                }
                                else{                
                                    double p = exp(-J*dE);
                                    // Generate random number between 0 and 1, flip if below p:
                                    if(distr2(gen)< p){
                                        if(spin_chosen_value == 1){Spins[spin_chosen] = 0;                                        
                                        }
                                        else{Spins[spin_chosen] = 1;                                        
                                        }  
                                    }                                    
                                }            
                            }     
                                    
                            //Save values:
                            spin_site_0[t] = Spins[0];
                            spin_site_1[t] = Spins[1];  
                            spin_site_2[t] = Spins[2];  
                             
                                
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
                                    }
                                    else{
                                        Spins[spin_chosen] = 1;                                       
                                    }  
                                }
                                else{                
                                    double p = exp(-J*dE);
                                    // Generate random number between 0 and 1, flip if below p:
                                    if(distr2(gen)< p){
                                        if(spin_chosen_value == 1){Spins[spin_chosen] = 0;                                       
                                        }
                                        else{Spins[spin_chosen] = 1;                                       
                                        }  
                                    }                                  
                                }
                            //printSpins(Spins);
                            spin_site_0[t+T_evolve_parallel] = Spins[0];
                            spin_site_1[t+T_evolve_parallel] = Spins[1];  
                            spin_site_2[t+T_evolve_parallel] = Spins[2];            

                        }  

                        //TE01:     
                              
                        vector<double> b_temp = b(spin_site_1);
                        vector<double> d_temp = d(spin_site_1);

                                                                               
                        vector<double> a_temp = a(spin_site_0,spin_site_1);                        
                        vector<double> c_temp = c(spin_site_0,spin_site_1);                        
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
                        TE01 += TE_local/static_cast<double>(N_trials);                                 

                        //TE02:       

                                      
                        b_temp = b(spin_site_2);
                        d_temp = d(spin_site_2);

                                                                                
                        a_temp = a(spin_site_0,spin_site_2);                        
                        c_temp = c(spin_site_0,spin_site_2);                        
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
                        TE02 += TE_local/static_cast<double>(N_trials); 

                        //TE10:       

                                     
                        b_temp = b(spin_site_0);
                        d_temp = d(spin_site_0);

                                                                                
                        a_temp = a(spin_site_1,spin_site_0);                        
                        c_temp = c(spin_site_1,spin_site_0);                        
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
                        TE10 += TE_local/static_cast<double>(N_trials);

                        //TE12:       

                                        
                        b_temp = b(spin_site_2);
                        d_temp = d(spin_site_2);

                                                                               
                        a_temp = a(spin_site_1,spin_site_2);                        
                        c_temp = c(spin_site_1,spin_site_2);                        
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
                        TE12 += TE_local/static_cast<double>(N_trials);

                        //TE20:       

                                       
                        b_temp = b(spin_site_0);
                        d_temp = d(spin_site_0);

                                                                               
                        a_temp = a(spin_site_2,spin_site_0);                        
                        c_temp = c(spin_site_2,spin_site_0);                        
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
                        TE20 += TE_local/static_cast<double>(N_trials);

                        //TE21:       

                                       
                        b_temp = b(spin_site_1);
                        d_temp = d(spin_site_1);

                                                                              
                        a_temp = a(spin_site_2,spin_site_1);                        
                        c_temp = c(spin_site_2,spin_site_1);                        
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
                        TE21 += TE_local/static_cast<double>(N_trials); 


                        // Calculate TE0 (doublet as source):
            
                        b_temp = b(spin_site_0);
                        d_temp = d(spin_site_0);                                                               
                           
                        a_temp = a(spin_site_1,spin_site_2,spin_site_0);                        
                        c_temp = c(spin_site_1,spin_site_2,spin_site_0); 

                        TE_local = 0.;

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
                        TE0 += TE_local/static_cast<double>(N_trials);

                        // Calculate TE1 (doublet as source):
            
                        b_temp = b(spin_site_1);
                        d_temp = d(spin_site_1);                                                              
                           
                        a_temp = a(spin_site_0,spin_site_2,spin_site_1);                        
                        c_temp = c(spin_site_0,spin_site_2,spin_site_1); 

                        TE_local = 0.;

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
                        TE1 += TE_local/static_cast<double>(N_trials);

                        // Calculate TE2 (doublet as source):
                 
                        b_temp = b(spin_site_2);
                        d_temp = d(spin_site_2);                                                             
                           
                        a_temp = a(spin_site_0,spin_site_1,spin_site_2);                        
                        c_temp = c(spin_site_0,spin_site_1,spin_site_2); 

                        TE_local = 0.;

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
                        TE2 += TE_local/static_cast<double>(N_trials);                        

                    }//END OF run LOOP


                    TE01_.push_back(TE01);
                    TE02_.push_back(TE02);
                    TE10_.push_back(TE10);
                    TE12_.push_back(TE12);
                    TE20_.push_back(TE20);
                    TE21_.push_back(TE21);

                    TE0_.push_back(TE0);
                    TE1_.push_back(TE1);
                    TE2_.push_back(TE2);                   

                    if(TE01 > maxTE01){maxTE01 = TE01;}                    
                    if(TE01 <= maxTE01*0.7){doneTE01 = true;}
                    if(TE02 > maxTE02){maxTE02 = TE02;}                    
                    if(TE02 <= maxTE02*0.7){doneTE02 = true;}
                    if(TE10 > maxTE10){maxTE10 = TE10;}                    
                    if(TE10 <= maxTE10*0.7){doneTE10 = true;}
                    if(TE12 > maxTE12){maxTE12 = TE12;}                    
                    if(TE12 <= maxTE12*0.7){doneTE12 = true;}
                    if(TE20 > maxTE20){maxTE20 = TE20;}                    
                    if(TE20 <= maxTE20*0.7){doneTE20 = true;}
                    if(TE21 > maxTE21){maxTE21 = TE21;}                    
                    if(TE21 <= maxTE21*0.7){doneTE21 = true;}

                    if(TE0 > maxTE0){maxTE0 = TE0;}                    
                    if(TE0 <= maxTE0*0.7){doneTE0 = true;}
                    if(TE1 > maxTE1){maxTE1 = TE1;}                    
                    if(TE1 <= maxTE1*0.7){doneTE1 = true;}
                    if(TE2 > maxTE2){maxTE2 = TE2;}                    
                    if(TE2 <= maxTE2*0.7){doneTE2 = true;}

                    //cout << doneTE01 << doneTE02 << doneTE10 << doneTE12 << doneTE20 << doneTE21  << doneTE0 << doneTE1 <<doneTE2 << endl;
                    //cout << TE01 << TE02 << TE10 << TE12 << TE20 << TE21  << TE0 << TE1 <<TE2 << endl;
                    cout << "rank: " << rank << " c1: " <<  c1 << " c2: " << c2 << " c3: " << c3 << " finished with j = " << j << endl;
                    

                }//End of j loop. 
                
                saveVectorToFile(TE01_,"TernaryTree_TE_01_"+to_string(c1)+"_"+to_string(c2)+"_"+to_string(c3));
                saveVectorToFile(TE02_,"TernaryTree_TE_02_"+to_string(c1)+"_"+to_string(c2)+"_"+to_string(c3));
                saveVectorToFile(TE10_,"TernaryTree_TE_10_"+to_string(c1)+"_"+to_string(c2)+"_"+to_string(c3));
                saveVectorToFile(TE12_,"TernaryTree_TE_12_"+to_string(c1)+"_"+to_string(c2)+"_"+to_string(c3));
                saveVectorToFile(TE20_,"TernaryTree_TE_20_"+to_string(c1)+"_"+to_string(c2)+"_"+to_string(c3));
                saveVectorToFile(TE21_,"TernaryTree_TE_21_"+to_string(c1)+"_"+to_string(c2)+"_"+to_string(c3));

                saveVectorToFile(TE0_,"TernaryTree_TE_0_"+to_string(c1)+"_"+to_string(c2)+"_"+to_string(c3));
                saveVectorToFile(TE1_,"TernaryTree_TE_1_"+to_string(c1)+"_"+to_string(c2)+"_"+to_string(c3));
                saveVectorToFile(TE2_,"TernaryTree_TE_2_"+to_string(c1)+"_"+to_string(c2)+"_"+to_string(c3));
                
                cout << "rank: " << rank << " c1: " <<  c1 << " c2: " << c2 << " c3: " << c3 << " finished." << endl;

            }// End of c3 loop
        
        c1c2 -= num_procs;

    } //End of c1c2 while loop


    // Finalise MPI
    MPI_Finalize ();


    return 0;
}