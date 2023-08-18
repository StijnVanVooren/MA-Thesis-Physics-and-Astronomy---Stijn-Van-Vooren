#include<fstream>
#include<string>
#include<iostream>
#include <vector>
#include<armadillo>

std::string networkname = "CompleteTreeN31";

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

arma::Cube<double> readTextFile(const std::string& filename, int N, int J_res)
{
    std::ifstream file(path+filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return {};
    }

    arma::Cube<double> data(N,N,J_res);
    
    int d_read,t_read;
        for(int d=0; d<N; d++) {
            for(int t=0; t<N; t++) {
                // Read indices line
                file >> d_read >> t_read; 
                // Read TE values for each j corresponding to TE[d_read][t_read][j]
                for(int j=0; j<J_res; j++) {
                    file >> data(d_read,t_read,j);                    
                }
            }
        }

    file.close();

    return data;
}

std::vector<arma::Cube<double>> readTextFile2(const std::string& filename, int N, int J_res)
{
    std::ifstream file(path+filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return {};
    }

    std::vector<arma::Cube<double>>  data(J_res,arma::Cube<double>(N,N,N));
    
    int d1_read,d2_read,t_read;
        for(int d1=0; d1<N; d1++) {
            for(int d2 = 0;d2<N;d2++){
                for(int t=0; t<N; t++) {
                    // Read indices line
                    file >> d1_read >> d2_read >> t_read; 
                    // Read TE values for each j corresponding to TE[d_read][t_read][j]
                    for(int j=0; j<J_res; j++) {
                        file >> data[j](d1_read,d2_read,t_read);                    
                    }
                }
            }
        }

    file.close();

    return data;
}

int main(int argc, char *argv[])
{   
    // Load and print file INFO:
    int N, N_trials, T_relax_parallel, T_relax_sequential, T_evolve, T_evolve_parallel, T_evolve_sequential, J_res;
    double J_min, J_max;
    readInfoFile(N, N_trials, T_relax_parallel, T_relax_sequential, T_evolve, T_evolve_parallel, T_evolve_sequential, J_min, J_max, J_res);
    //printInfo(N, N_trials, T_relax_parallel, T_relax_sequential, T_evolve, T_evolve_parallel, T_evolve_sequential, J_min, J_max, J_res);    

    
    std::ifstream file(path+"Chi.txt");
    if (!file) {
        std::cerr << "Failed to open file." << std::endl;
        return 1;
    }


    std::vector<double> Chi(J_res);
    for (int i = 0; i < J_res; ++i) {
        file >> Chi[i];
    }

    file.close();

    for (int j = 0; j < J_res; ++j) {
        std::string filename = "Chi_" + std::to_string(j) + ".txt";
        std::ofstream file2(pathtemp+filename);
        if (!file2) {
            std::cerr << "Failed to open file: " << filename << std::endl;
            
        }
        file2 << Chi[j];
        file2.close();
    }

    arma::Cube<double> TE1 = readTextFile("TE1.txt", N ,J_res);

    for(int j=0;j<J_res;j++){
        arma::Mat<double> slice = TE1.slice(j);
        slice.save(pathtemp+"TE1_"+std::to_string(j)+".bin");
    }

    std::vector<arma::Cube<double>> TE2 = readTextFile2("TE2.txt", N ,J_res);

    for(int j=0;j<J_res;j++){
        TE2[j].save(pathtemp+"TE2_"+std::to_string(j)+".bin");
    }

    std::vector<arma::Cube<double>> ST = readTextFile2("S_T.txt", N ,J_res);

    for(int j=0;j<J_res;j++){
        ST[j].save(pathtemp+"ST_"+std::to_string(j)+".bin");
    }

    std::vector<arma::Cube<double>> RT = readTextFile2("R_T.txt", N ,J_res);

    for(int j=0;j<J_res;j++){        
        RT[j].save(pathtemp+"RT_"+std::to_string(j)+".bin");
    }
    

    return 0;
}