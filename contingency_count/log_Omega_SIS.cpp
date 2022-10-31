// This code may be compiled as g++ -std=c++17 -o log_Omega_SIS log_Omega_SIS.cpp

#include <getopt.h>
#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <random>
#include <chrono>
#include <sstream>
#include <vector>
#include <iterator>
#include <random>
#include <vector>
#include <iomanip>
#include <map>

//*********************** GLOBAL VARIABLES *******************************************************************


//*********************** FUNCTION DECLARATIONS **************************************************************

// Sampling functions
void sampleTable(std::vector<std::vector<int>>& table, double& mLogProb, std::vector<int> rs, std::vector<int> cs, double **lnHVals,double ***lnGVals);
void sampleTableColumn(std::vector<int>& column, double& mLogProb,std::vector<int>& rrs, int& colNum, int& c, double **lnHVals, double ***lnGVals);

// Precomputation Helpers
double alphaC(std::vector<int>& cs, int j, int m);

// String splitting (for parsing input files)
template <typename Out> void split(const std::string &s, char delim, Out result);
std::vector<std::string> split(const std::string &s, char delim);

// Debugging functions
void print_array (int arg[], int length);
void print_array (double arg[], int length);
void print_vector (std::vector<int> vec);
void print_vector (std::vector<double> vec);
void print_vector (std::vector<std::string> vec);
void print_table(std::vector<std::vector<int>>& table); // Useful to check on the state of the table

//*********************** MAIN PROGRAM ***********************************************************************

int main (int argc, char *argv[]) {
  // Read command-line arguments
  std::string inputFilename = ""; //i
  std::string outputFilename = ""; //o
  bool writing = true; //w
  int iterations = 1; //t
  int save_freq = 1; //S -> convert this to the seed for mt
  int maxTime = 60; //T 
  std::string mode = "EC"; //M

  int opt;
  // take command line arguments
  while ((opt = getopt(argc, argv, "i:o:w:t:S:T:M:")) != -1) {
    switch (opt) {
      case 'i':
        inputFilename = optarg;
        break;
      case 'o':
        outputFilename = optarg;
        break;
      case 't':
        iterations = atoi(optarg);
        break;
      case 'T':
        maxTime = atoi(optarg);
        break;
      case 'S':
        save_freq = atoi(optarg);
        break;
      case 'w':
        if(optarg == "F"){
          writing = false;
        }
        break;
      case 'M':
        mode = optarg;
        break;
      default:
        exit(EXIT_FAILURE);
    }
  }

  // Read Data from inputFilename
  // File should contain two rows with the row sums followed by the column sums
  std::ifstream inputFile(inputFilename); 
  std::ofstream outFile;
  if (inputFile.is_open()){
    if(writing){
      std::ofstream outFile (outputFilename); // Reset file
    }

    // Start timer 
    auto start = std::chrono::high_resolution_clock::now();
    if(writing){
      outFile << "Iteration,Omega,sigmaOmega,maxOmega,cv2,time" << std::endl;
    }
    // Parse input
    std::string line;
    std::getline(inputFile,line);
    // std::cout << inputFilename << " " << line << std::endl;
    std::vector<std::string> lineSplit = split(line, ' ');
    std::vector<int> rs;
    for(std::string s: lineSplit){
      rs.push_back(std::stoi(s));
    }
    std::getline(inputFile,line);
    lineSplit = split(line, ' ');
    std::vector<int> cs;
    for(std::string s: lineSplit){
      cs.push_back(std::stoi(s));
    }
    // Sort the columns cs
    std::sort(cs.begin(),cs.end());
    std::reverse(cs.begin(),cs.end());
    // std::cout << "rs: ";
    // print_vector(rs);
    // std::cout << "cs: ";
    // print_vector(cs);
    int m = rs.size();
    int n = cs.size();
    int N = 0;
    for(int val: rs){
      N += val;
    }
    double mLogProb;
    std::vector<double> mLogProbs;
    std::vector<double> tableHashes;
    int countApps = 0;
    double eNum = 0;
    std::srand(1);

    // Precomputing alphas, lnHVals
    std::vector<double> alphas;
    for(int j = 0; j < n; j++){
      if(mode=="EC"){
        alphas.push_back(alphaC(cs,j,m));
      }else{
        if(j<cs.size() - 1){
          alphas.push_back(cs.size() - (j+1));
        }else{
          alphas.push_back(1);
        }
      }
    }
    // std::cout << "alphas:" << std::endl;
    // print_vector(alphas);
    int maxR = 0;
    for(int r: rs){
      if(r > maxR){
        maxR = r;
      }
    }

    // Instantiate lnGVals, lnHVals
    double **lnHVals; // Represents the transition probabilities 
    lnHVals = new double *[n];
    for(int j = 0; j < n; j++){
      lnHVals[j] = new double[maxR + 1];
      for(int rr = 0; rr < maxR + 1; rr++){
        lnHVals[j][rr] = std::lgamma(rr + alphas[j]) - std::lgamma(alphas[j]) - std::lgamma(rr + 1);
        // std::cout << "alpha "<< alphas[j] <<" lnHVals["<<j<<","<<rr<<"]="<<lnHVals[j][rr]<<std::endl;
      }
    }

    // Instantiate lnHVals
    int maxC = 0;
    for(int c: cs){
      if(c > maxC){
        maxC = c;
      }
    }
    double ***lnGVals;
    lnGVals = new double **[m+1];
    for(int i = 0; i < m+1; i++){
      lnGVals[i] = new double *[maxC+1];
      for(int sl = 0; sl < maxC+1; sl++){
        lnGVals[i][sl] = new double[maxC + 1];
        for(int sc = 0; sc < maxC + 1; sc++){
          lnGVals[i][sl][sc] = 0;
        }
      }
    }
    for(int t = 1; t <= iterations; t++){
      mLogProb = 0;
      std::vector<std::vector<int>> table;
      sampleTable(table,mLogProb,rs,cs,lnHVals,lnGVals);

      // Optionally print the sampled tables
      // std::cout << "Table sampled with probability " << prob<< " -> "<< (1/prob)<< std::endl;
      // std::cout << "Table sampled with -ln p " << mLogProb << std::endl;
      // print_table(table);
      mLogProbs.push_back(mLogProb);
      auto stop = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::seconds>(stop - start);
      if(int(std::sqrt(t)) == std::sqrt(t) || t == iterations || duration.count() >= maxTime){
        double max = *std::max_element(mLogProbs.begin(),mLogProbs.end());
        std::vector<double> amounts;
        for(double lP : mLogProbs){
          amounts.push_back(std::exp(lP - max));
        }
        double sum = std::accumulate(amounts.begin(), amounts.end(), 0.0);
        double mean = sum / amounts.size();

        std::vector<double> diff(amounts.size());
        std::transform(amounts.begin(), amounts.end(), diff.begin(), [mean](double x) { return x - mean; });
        double sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
        double stdev = std::sqrt(sq_sum / amounts.size());
        double dShift = std::floor((max+std::log(mean))/std::log(10));
        double maxDShift = std::floor(max/std::log(10));

        double cv2 = 0;
        for(double am : amounts){
          cv2 += (am - mean)*(am - mean)/(amounts.size() - 1);
        }
        cv2 /= mean*mean;
        if(writing){
          std::ofstream outFile (outputFilename, std::ios_base::app);
        }
        // Get ending timepoint
        if(writing){
          outFile << t <<  "," << mean*std::exp(max - dShift*std::log(10)) << "E10+"<< dShift << ","<< stdev/std::sqrt(amounts.size())*std::exp(max - dShift*std::log(10)) << "E10+"<< dShift <<","<<std::exp(max - maxDShift*std::log(10))<<"E+"<<maxDShift<<"," << cv2 <<","<<duration.count()<< std::endl;  
        }
        if(t == iterations || duration.count() >= maxTime){
          std::cout <<"{"<< "\"iterations\":"<<t<<",";
          std::cout << "\"value\":\"" << mean*std::exp(max - dShift*std::log(10)) << "E10+"<< dShift << "\",";
          std::cout << "\"error\":\"" << stdev/std::sqrt(amounts.size())*std::exp(max - dShift*std::log(10)) << "E10+"<< dShift <<"\",";
          std::cout << "\"max\":\"" <<std::exp(max - maxDShift*std::log(10))<<"E+"<<maxDShift<<"\",";
          std::cout << "\"cv2\":"  << cv2 <<",";
          std::cout << "\"time\":" <<duration.count()<< "}"<<std::endl;  
          break;
        }
      }
    }
    // auto stop = std::chrono::high_resolution_clock::now();
    // auto duration = std::chrono::duration_cast<std::chrono::seconds>(stop - start);
    // std::cout << t <<  "," << mean*std::exp(max - dShift*std::log(10)) << "E10+"<< dShift << ","<< stdev/std::sqrt(amounts.size())*std::exp(max - dShift*std::log(10)) << "E10+"<< dShift <<","<<std::exp(max - maxDShift*std::log(10))<<"E+"<<maxDShift<<"," << cv2 <<","<<duration.count()<< std::endl;  
  }else{
    std::cout << "File not found at " << inputFilename << std::endl;
  }
}


void sampleTableColumn(std::vector<int>& column, double& mLogProb,std::vector<int>& rrs, int& colNum, int& c, double **lnHVals, double ***lnGVals){
  // Compute lnGVals recursively
  int m = rrs.size();
  double gSumShift = 0; // For floating point precision, when performing the sums in g_i(sl,sc) = h_i(sl,sc) sum_sn(g_{i+1}(sc,sn)) we shift the g_{i+1}(sc,sn)
  for(int i = m - 1; i >= 0; i--){
    if(i == m - 1){ // The bottom row has g_i = h_i
      for(int sl = std::max(0,c - rrs[m-1]); sl <= c; sl++){
        lnGVals[m - 1][sl][c] = lnHVals[colNum][rrs[m-1] - c + sl];
        // std::cout << "colNum "<<colNum<< ": gVal["<< i << "," << sl << "," << c << "] = " << std::exp(lnGVals[m - 1][sl][c]) << std::endl;
      }
    }else if(i > 0){
      for(int sc = 0; sc <= c; sc++){ // For each possible sc, compute the sum sum_sn(g_{i+1}(sc,sn)) 
                                      // and calculate g_i(sl,sc) = h_i(sl,sc) sum_sn(g_{i+1}(sc,sn)) for the valid sl
        double gSum = 0; // gSum = \sum_i exp(\ln x_i - shift)
        double gSumShift = 0; // we sum the exponents as \ln(\sum_i x_i) = \ln(\sum_i exp(\ln x_i)) =  \ln(\sum_i exp(\ln x_i - shift)) + shift
                              // shift = max_i \ln x_i
        if(i == m - 2){ // If on the second-to-last row, there is only one nonzero g_{m-1}(sc,sn), sn = c
          if(sc >= c - rrs[m-1]){ // Also needs to be a valid transition
            gSumShift = lnGVals[i+1][sc][c];
            gSum = 1; // exp(0)
          }
        }else{
          for(int sn = sc; sn <= std::min(c,sc+rrs[i+1]); sn++){ // Nonzero g_{i+1}(sc,sn) satisfy sn >= sc, sn <= c, sn <= sc + x[colNum,i+1] <= sc + rrs[i+1]
            if(lnGVals[i+1][sc][sn] > gSumShift){ // find max in range for shifting
              gSumShift = lnGVals[i+1][sc][sn];
            }
          }
          for(int sn = sc; sn <= std::min(c,sc+rrs[i+1]); sn++){ // calculate gSum = \sum_i exp(\ln x_i - shift) over range
            gSum += std::exp(lnGVals[i+1][sc][sn] - gSumShift);
          }
        }
        // Evaluate as g_i(sl,sc) = h_i(sl,sc) sum_sn(g_{i+1}(sc,sn)) for valid sl
        for(int sl = std::max(0,sc - rrs[i]); sl <= sc; sl++){ // Require 0 <= sl, sl <= sc, sl = sc - x[colNum,i] >= sc - rrs[i]
          // if(sl == 0 && sc == 0){
          //     std::cout << "colNum "<<colNum<<" i "<<i<<" gSum " << gSum <<std::endl;
          // }
          lnGVals[i][sl][sc] = lnHVals[colNum][rrs[i] - sc + sl] + gSumShift + std::log(gSum);
        }
      }

    }else{
      for(int sc = 0; sc <= std::min(c,0+rrs[i]); sc++){
        double gSum = 0;
        for(int sn = sc; sn <= std::min(c,sc+rrs[i+1]); sn++){
          if(lnGVals[i+1][sc][sn] > gSumShift){
            gSumShift = lnGVals[i+1][sc][sn];
          }
        }
        for(int sn = sc; sn <= std::min(c,sc+rrs[i+1]); sn++){
          gSum += std::exp(lnGVals[i+1][sc][sn] - gSumShift);
        }
        lnGVals[i][0][sc] = lnHVals[colNum][rrs[i] - sc + 0] + gSumShift + std::log(gSum);
        // std::cout << "colNum "<<colNum<< ": gVal["<< i << "," << 0 << "," << sc << "] = " << std::exp(lnGVals[i][0][sc]) << std::endl;
      }
    } 
  }
  // std::cout << "colNum " << colNum << " c " << c << " " << lnGVals[0][0][2] << std::endl;
  // Sample the column using the computed lnGVals
  int sc = 0; // Current sum
  for(int i = 0; i < m; i++){
    double totWeight = 0;
    if(i == m - 1){ // Last entry is forced
      column.push_back(c - sc);
      rrs[i] -= c - sc;
    }else{
      gSumShift = 0;
      for(int sn = sc; sn <= std::min(c,sc+rrs[i]); sn++){
        if(lnGVals[i][sc][sn] > gSumShift){
          gSumShift = lnGVals[i][sc][sn];
        }
      }
      for(int sn = sc; sn <= std::min(c,sc+rrs[i]); sn++){ // Iterate over possible next sums
        // std::cout << i << " " << sc << " -> " << sn << " " << std::exp(lnGVals[i][sc][sn]) << " " << std::endl;
        totWeight += std::exp(lnGVals[i][sc][sn] - gSumShift);
      }
      int sn = sc;
      float r = totWeight*(static_cast <float> (std::rand()) / static_cast <float> (RAND_MAX));
      for(int psn = sc; psn <= std::min(c,sc+rrs[i]); psn++){
        if(r <= std::exp(lnGVals[i][sc][psn] - gSumShift)){
          sn = psn;
          break;
        }else{
          r -= std::exp(lnGVals[i][sc][psn] - gSumShift);
        }
      }
      //std::cout << sn << std::endl;
      column.push_back(sn - sc);
      rrs[i] -= sn - sc;
      mLogProb += std::log(totWeight) - lnGVals[i][sc][sn] + gSumShift;
      // std::cout <<"Chosen: "<<  i << " " << sc << " -> " << sn << " " << std::exp(lnGVals[i][sc][sn] - gSumShift)/totWeight << " " << std::endl;
      // if(std::isnan(std::exp(lnGVals[i][sc][sn] - gSumShift)/totWeight)){
      //   for(int sn = sc; sn <= std::min(c,sc+rrs[i]); sn++){ // Iterate over possible next sums
      //     std::cout << i << " " << sc << " -> " << sn << " " << lnGVals[i][sc][sn] << " " << std::endl;
      //   }
      //   std::cout << sc << " " << std::min(c,sc+rrs[i]) << " " << totWeight << std::endl;
      //   std::cout <<"Chosen: "<<  i << " " << sc << " -> " << sn << " " << std::exp(lnGVals[i][sc][sn] - gSumShift)/totWeight << " " << std::endl;
      // }
      sc = sn;
    }
  }
}

// Update the given table and logProb with a sample and probability
void sampleTable(std::vector<std::vector<int>>& table, double& mLogProb, std::vector<int> rs, std::vector<int> cs, double **lnHVals,double ***lnGVals){
  std::vector<int> rrs;
  for(int r : rs){ // Make deep copy of rs and cs to be manipulated by reference by the samplers
    rrs.push_back(r);
  }
  std::vector<int> rcs;
  for(int c : cs){
    rcs.push_back(c);
  }
  for(int j = 0; j < cs.size(); j++){
    // std::cout << "sampling column " << j << std::endl;
    table.push_back(std::vector<int>());
    sampleTableColumn(table[j],mLogProb,rrs,j,rcs[j],lnHVals,lnGVals);
    // print_table(table);
  }
}

// alphaC for EC estimate calculate for columns [j+1:n]
double alphaC(std::vector<int>& cs, int j, int m){
  float N = std::accumulate(cs.begin() + j + 1, cs.end(), 0); // Sum of remaining columns
  if(N == 0){
    return 1;
  }
  if(N == cs.size() - (j+1)){ // Should only occur for cs = (1,1,...,1)
    return 1000;
  }
  double cDNorm = 0;// ||c/N||^2
  for(int jt = j +1; jt < cs.size(); jt++){
    cDNorm += cs[jt]*cs[jt]/(N*N);
  }
  return ((1 - 1/N)+(1 - cDNorm)/m)/(cDNorm-1/N);
}

// Splitting string by token
template <typename Out>
void split(const std::string &s, char delim, Out result) {
    std::istringstream iss(s);
    std::string item;
    while (std::getline(iss, item, delim)) {
        *result++ = item;
    }
}
std::vector<std::string> split(const std::string &s, char delim) {
    std::vector<std::string> elems;
    split(s, delim, std::back_inserter(elems));
    return elems;
}

// Functions for printing various objects (helpful for debugging)
void print_array (int arg[], int length) {
  for (int n=0; n<length; ++n)
    std::cout << arg[n] << ' ';
  std::cout << '\n';
}
void print_array (double arg[], int length) {
  for (int n=0; n<length; ++n)
    std::cout << arg[n] << ' ';
  std::cout << '\n';
}
void print_vector (std::vector<int> vec) {
  for (int n=0; n<vec.size(); ++n)
    std::cout << vec[n] << ' ';
  std::cout << '\n';
}
void print_vector (std::vector<double> vec) {
  for (int n=0; n<vec.size(); ++n)
    std::cout << vec[n] << ' ';
  std::cout << '\n';
}
void print_vector (std::vector<std::string> vec) {
  for (int n=0; n<vec.size(); ++n)
    std::cout << vec[n] << ' ';
  std::cout << '\n';
}
void print_table(std::vector<std::vector<int>>& table){
  for(int j = 0; j < table[0].size(); j++){
    for(int i = 0; i < table.size(); i++){
      if(j < table[i].size()){
        std::cout << table[i][j] << " ";
      }
    }
    std::cout << std::endl;
  }
}
