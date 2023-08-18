#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>

using namespace std;

int main()
{
    // Open the input file
    ifstream infile("conjecture1_randomTree_1.txt");
    if (!infile)
    {
        cerr << "Error: Could not open input file." << endl;
        return 1;
    }

    // Open the output file
    ofstream outfile("output.txt");
    if (!outfile)
    {
        cerr << "Error: Could not open output file." << endl;
        return 1;
    }

    // Loop through each line of the input file
    string line;
    while (getline(infile, line))
    {
        // Check if the line contains the substrings we want to remove
        if (line.find(":") != string::npos || line.find("Vertex") != string::npos)
        {
            // Skip this line
            continue;
        }

        // Write the line to the output file
        outfile << line << endl;
    }

    // Close the input and output files
    infile.close();
    outfile.close();

    return 0;
}
