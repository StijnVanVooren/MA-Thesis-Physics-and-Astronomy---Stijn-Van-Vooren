#include<vector>
#include<fstream>
#include<iostream>
#include<string>
#include<armadillo>

using namespace arma;
using namespace std;

std::string networkname = "RandomTreeN60";

// Comment out the paths except the one where you want to save the results:
std::string path = "/home/stijn/Documents/ThesisNew/FullSim/Data/"+networkname+"/";
//std::string path =  "/theia/scratch/brussel/107/vsc10773/Data/"+networkname+"/";
//std::string path = "/kyukon/scratch/gent/459/vsc45915/Data/"+networkname+"/";
std::string pathtemp = path + "temp/";




void readInfoFile(int& N, int& N_trials, int& T_relax_parallel, int& T_relax_sequential, int& T_evolve, int& T_evolve_parallel, int& T_evolve_sequential, double& J_min, double& J_max, int& J_res)
{
    std::ifstream infile(path+"INFO.txt");
    std::string line;

    while (std::getline(infile, line))
    {
        if (line == "N:")
        {
            infile >> N;
        }
        else if (line == "N_trials:")
        {
            infile >> N_trials;
        }
        else if (line == "T_relax_parallel:")
        {
            infile >> T_relax_parallel;
        }
        else if (line == "T_relax_sequential:")
        {
            infile >> T_relax_sequential;
        }
        else if (line == "T_evolve:")
        {
            infile >> T_evolve;
        }
        else if (line == "T_evolve_parallel:")
        {
            infile >> T_evolve_parallel;
        }
        else if (line == "T_evolve_sequential:")
        {
            infile >> T_evolve_sequential;
        }
        else if (line == "J_min:")
        {
            infile >> J_min;
        }
        else if (line == "J_max:")
        {
            infile >> J_max;
        }
        else if (line == "J_res:")
        {
            infile >> J_res;
        }
    }
}





int main(int argc, char *argv[])
{   
    int N, N_trials, T_relax_parallel, T_relax_sequential, T_evolve, T_evolve_parallel, T_evolve_sequential, J_res;
    double J_min, J_max;
    readInfoFile(N, N_trials, T_relax_parallel, T_relax_sequential, T_evolve, T_evolve_parallel, T_evolve_sequential, J_min, J_max, J_res);
    //printInfo(N, N_trials, T_relax_parallel, T_relax_sequential, T_evolve, T_evolve_parallel, T_evolve_sequential, J_min, J_max, J_res);  

    std::vector<double> Chi(J_res);
    std::vector<arma::Mat<double>> TE1(J_res,arma::Mat<double>(N,N));
    std::vector<arma::Cube<double>> TE2(J_res,arma::Cube<double>(N,N,N));
    std::vector<arma::Cube<double>> ST(J_res,arma::Cube<double>(N,N,N));
    std::vector<arma::Cube<double>> RT(J_res,arma::Cube<double>(N,N,N));

    // Load data from temp folder:
    for(int j=0;j<J_res;j++){   
        std::ifstream ifileChi(pathtemp+"Chi"+std::to_string(j)+".txt");
        ifileChi >> Chi[j];
        ifileChi.close();


        TE1[j].load(pathtemp+"TE1_"+std::to_string(j)+".bin");
        TE2[j].load(pathtemp+"TE2_"+std::to_string(j)+".bin");
        ST[j].load(pathtemp+"ST_"+std::to_string(j)+".bin");
        RT[j].load(pathtemp+"RT_"+std::to_string(j)+".bin");
    }

    // Set correct values to zero:
    for(int d1=0; d1<N; d1++) {
        for(int t=0; t<N; t++) {     
            for(int d2=0; d2<N; d2++) {
                if(d1 == t || d2 == t){
                    for(int j=0;j<J_res;j++){
                        ST[j](d1,d2,t) = 0;
                        RT[j](d1,d2,t) = 0;
                    }
                }
                if(d1 == d2){
                    for(int j=0;j<J_res;j++){
                        ST[j](d1,d2,t) = 0;
                        RT[j](d1,d2,t) = TE1[j](d1,t);                        
                    }
                }
            }
        }
    }

    //Save Chi:
    std::ofstream oFileChi(path+"Chi.txt");
    for(int j=0;j<J_res;j++){
        oFileChi << Chi[j] << " " ;
    }
    oFileChi << std::endl;
    oFileChi.close();

    //Save TE1:
    std::ofstream oFileTE1(path+"TE1.txt");
    // Write N_trials_cumulative to first line.
    for(int d=0;d<N;d++){
        for(int t=0;t<N;t++){
            oFileTE1 << d << " " << t << std::endl; 
            for(int j=0;j<J_res;j++){
                oFileTE1 << TE1[j](d,t) << " ";                
            }
            oFileTE1 << std::endl;
        }
    }   
    oFileTE1.close();

    //Save TE2
    std::ofstream oFileTE2(path+"TE2.txt");
    
    for(int d1=0; d1<N; d1++) {
        for(int d2=0; d2<N; d2++) {
            for(int t=0; t<N; t++) { 
                oFileTE2 << d1 << " " << d2 << " " << t << std::endl; 
                for(int j=0;j<J_res;j++){
                    oFileTE2 << TE2[j](d1,d2,t) << " ";                
                }
                oFileTE2 << std::endl;
            }
        }
    }
    oFileTE2.close();

    //Save ST
    std::ofstream oFileST(path+"ST.txt");
    
    for(int d1=0; d1<N; d1++) {
        for(int d2=0; d2<N; d2++) {
            for(int t=0; t<N; t++) { 
                oFileST << d1 << " " << d2 << " " << t << std::endl; 
                for(int j=0;j<J_res;j++){
                    oFileST << ST[j](d1,d2,t) << " ";                
                }
                oFileST << std::endl;
            }
        }
    }
    oFileST.close();

    //Save RT
    std::ofstream oFileRT(path+"RT.txt");
    
    for(int d1=0; d1<N; d1++) {
        for(int d2=0; d2<N; d2++) {
            for(int t=0; t<N; t++) { 
                oFileRT << d1 << " " << d2 << " " << t << std::endl; 
                for(int j=0;j<J_res;j++){
                    oFileRT << RT[j](d1,d2,t) << " ";                
                }
                oFileRT << std::endl;
            }
        }
    }
    oFileRT.close();
    
    return 0;
}