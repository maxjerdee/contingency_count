/*  Given marginalization counts n1 and n2, estimate the number of possible 0-1 or non-neg int tables
* (and also generate samples along the way)
*/

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

// Print out arrays (for debugging)
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

// Return the indices perm which label the row sums from largest to smallest
template<typename T>
std::vector<int> argsort(const std::vector<T> &array) {
    std::vector<int> indices(array.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(),
              [&array](int left, int right) -> bool {
                  // sort indices according to corresponding array element
                  return array[left] > array[right];
              });

    return indices;
}

// Return log(exp(a) + exp(b)), while shifting to avoid floating point problems
double logSumExp(double a, double b){
  if(a == -INFINITY && b == -INFINITY){
    return -INFINITY;
  }
  return std::log(std::exp(a - std::max(a,b)) + std::exp(b - std::max(a,b))) + std::max(a,b);
}

// Update the given information with a sampled column
void sampleTableColumn(std::vector<int>& column, double& mLogProb,std::vector<int>& rrs, int cRemain, int& c, double *lnHVals, double ***lnGVals, int *GRBound, bool verbose){
  // std::cout <<"Remaing Columns "<< cRemain << std::endl;
  // std::cout <<"Remaing Row Sum ";
  // print_vector(rrs);
  int m = rrs.size();
  std::vector<int> rPerm = argsort(rrs); // (length m) vector of the indices which give the elements of rrs from largest to smallest
  // All rows will be referenced through this permutation
  // for(int i = 0; i < m; i++){
  //   std::cout << rrs[rPerm[i]]<<" " << rPerm[i] << std::endl;
  // }
  // Calculate sBound[i], the lower bounds on the partial sum s[i]
  std::vector<int>sBound;
  int totalR = 0;
  for(int i = 0; i < m; i++){
    totalR += rrs[rPerm[i]];
    sBound.push_back(std::max(0,totalR - GRBound[i]));
  }
  // std::cout<<"sorted rrs ";
  // for(int i = 0; i < m; i++){
  //   std::cout<<" "<<rrs[rPerm[i]];
  // }
  // std::cout << ", c="<<c<<std::endl;
  // std::cout << "sBound: ";
  // print_vector(sBound);
  double gSumShift = 0;
  // Compute lnGVals recursively
  for(int i = m - 1; i >= 0; i--){
    for(int sc = 0; sc <= c; sc++){
        // Reset the lnGVals to -INFINITY
        if(sc - 1 >= 0){
          lnGVals[i][sc - 1][sc] = -INFINITY;
        }
        lnGVals[i][sc][sc] = -INFINITY;
        lnGVals[i][sc][sc+1] = -INFINITY;
      }
    if(i == m - 1){ // The bottom row has g_i = h_i
      // Only valid combinations are sc = c, sl = c - 1, sl = c
      // lnGVals[m - 1][sl][c] = lnHVals[colNum][rrs[m-1] - c + sl]; 
      // std::cout<<(c - 1)<<sBound[m-2]<<std::endl;
      if(c - 1 >= 0 && rrs[rPerm[m-1]] > 0 && c - 1 >= sBound[m-2]){ // x[m-1] = 1 only if c - 1 >= 0, r[m-1] > 0, c - 1 >= sBound[m-2]
        lnGVals[m - 1][c - 1][c] = lnHVals[rrs[rPerm[m-1]] - 1];
      }
      if(rrs[rPerm[m-1]] < cRemain &&  c >= sBound[m-2]){ // x[m-1] = 0 only if r[m-1] < cRemain, c >= sBound[m-2]
        lnGVals[m - 1][c][c] = lnHVals[rrs[rPerm[m-1]]];
      }
    }else if(i > 0){
      for(int sc = sBound[i]; sc <= c; sc++){ // sc >= sBound[i]
        // For each possible sc, compute the sum sum_sn(g_{i+1}(sc,sn)) 
        double logGSum;
        if(sc < c){
          logGSum = logSumExp(lnGVals[i+1][sc][sc],lnGVals[i+1][sc][sc+1]);
        }else{
          logGSum = lnGVals[i+1][sc][sc];
        }
        // std::cout<<"A "<<i<<" "<<sc<<" "<<sBound[i-1]<<" "<<logGSum<<std::endl;
        // Evaluate as g_i(sl,sc) = h_i(sl,sc) sum_sn(g_{i+1}(sc,sn)) for valid sl
        if(sc - 1 >= 0 && rrs[rPerm[i]] > 0 && sc - 1 >= sBound[i - 1]){ // x[i] = 1, sl = sc - 1 works if sl >= 0, r[i] > 0, sl >= sBound[i-1], sc >= sBound[i]
            lnGVals[i][sc - 1][sc] = lnHVals[rrs[rPerm[i]] - 1] + logGSum;
        }
        if(rrs[rPerm[i]] < cRemain && sc >= sBound[i - 1]){ // x[i] = 0, sl = sc works if r[i] < cRemain, sl >= sBound[i-1], sc >= sBound[i]
          lnGVals[i][sc][sc] = lnHVals[rrs[rPerm[i]]] + logGSum;
        }
      }

    }else{ // i = 0
      // sl = s[0] = 0, sc = s[1] can be 0 or 1
      if(rrs[rPerm[0]] > 0 && 1 >= sBound[0]){ // x[0] = sc = 1 if r[i] > 0, sc >= sBound[0] 
        lnGVals[0][0][1] = lnHVals[rrs[rPerm[0]] - 1] + logSumExp(lnGVals[1][1][1],lnGVals[1][1][2]);
      }
      if(rrs[rPerm[0]] < cRemain && 0 >= sBound[0]){ // x[0] = sc = 0 if r[i] < cRemain, sc >= sBound[0]
        lnGVals[0][0][0] = lnHVals[rrs[rPerm[0]]] + logSumExp(lnGVals[1][0][0],lnGVals[1][0][1]);
      }
    } 
  } 
  if(verbose){
    for(int i = 0; i < m; i++){
      for(int sc = 0; sc <= c; sc++){
        for(int sn = sc; sn <= sc + 1; sn++){
          if(sn <= c){
            std::cout <<"gVal["<<i<<","<<sc<<","<<sn<<"]="<<lnGVals[i][sc][sn]<<std::endl; 
          }
        }
      }
    }
  }
  // Sample the column using the computed lnGVals
  int sc = 0; // Current sum
  int* sampledColumn = new int[m];
  for(int i = 0; i < m; i++){
    if(i == m - 1){ // Last entry is forced
      sampledColumn[rPerm[i]] = c - sc;
      rrs[rPerm[i]] -= c - sc;
    }else{
      int sn;
      if(sc == c){ // If the culmulative sum is already maximum, next choice is forced
        sn = sc;
      }else{
        double shift = std::max(lnGVals[i][sc][sc],lnGVals[i][sc][sc+1]);
        double totWeight = std::exp(lnGVals[i][sc][sc]-shift)+std::exp(lnGVals[i][sc][sc+1]-shift);
        // std::cout<<totWeight<<std::endl;
        float r = totWeight*(static_cast <float> (std::rand()) / static_cast <float> (RAND_MAX));
        if(verbose){
          std::cout << (std::exp(lnGVals[i][sc][sc] - shift))<<" "<<totWeight << std::endl;
        }
        if(r < std::exp(lnGVals[i][sc][sc] - shift)){
          sn = sc;
          mLogProb += std::log(totWeight) - lnGVals[i][sc][sc] + shift;
        }else{
          sn = sc + 1;
          mLogProb += std::log(totWeight) - lnGVals[i][sc][sc+1] + shift;
        }
      }
      sampledColumn[rPerm[i]] = sn - sc;
      rrs[rPerm[i]] -= sn - sc;
      sc = sn;
    }
    // std::cout<<i<<" "<<mLogProb<<std::endl;
  }
  if(verbose){
    print_array(sampledColumn,m);
    std::cout << std::exp(-mLogProb) << std::endl;
  }
  for(int i = 0; i < m; i++){
    // if(cRemain == 3){
    //   std::cout<<"Column 3, row "<<i<<" "<<sampledColumn[i]<<std::endl;
    // }
    column.push_back(sampledColumn[i]);
  }
}

int zeroOneTableHash(std::vector<std::vector<int>>& table){
  int pow = 1;int res = 0;
  for(int j = 0; j < table.size(); j++){
    for(int i = 0; i < table.size(); i++){
      res += table[i][j]*pow;
      pow *= 2;
    }
  }
  return res;
}

// Update the given table and logProb with a sample and probability
void sampleTable(std::vector<std::vector<int>>& table, double& mLogProb, std::vector<int> rs, std::vector<int> cs, double **lnHVals,double ***lnGVals, int **GRBound, bool verbose){
  std::vector<int> rrs;
  for(int r : rs){ // Make deep copy of rs and cs to be manipulated by reference by the samplers
    rrs.push_back(r);
  }
  std::vector<int> rcs;
  for(int c : cs){
    rcs.push_back(c);
  }
  for(int j = 0; j < cs.size(); j++){
    if(verbose){
      std::cout << "sampling column " << j << std::endl;
    }
    table.push_back(std::vector<int>());
    sampleTableColumn(table[j],mLogProb,rrs,cs.size() - j,rcs[j],lnHVals[j],lnGVals,GRBound[j], verbose);
    if(verbose){
      std::cout << "done sampling column " << j << " -log(p) "<<mLogProb<<std::endl;
      // print_table(table);
    }
  }
}

// alphaC for EC estimate calculate for columns [j+1:n]
double alphaC(std::vector<int>& cs, int j, int m){
  float N = std::accumulate(cs.begin() + j + 1, cs.end(), 0); // Sum of remaining columns
  if(N == 0){
    return 1;
  }
  double cDNorm = 0;// ||c/N||^2
  for(int jt = j +1; jt < cs.size(); jt++){
    cDNorm += cs[jt]*cs[jt]/(N*N);
  }
  if((cDNorm-1/N)==0){ // Should only occur for cs = (1,1,...,1)
    return 5000;
  }
  if(std::abs(((1 - 1/N)-(1 - cDNorm)/m)/(cDNorm-1/N))>1000){
    return 5000;
  }
  return ((1 - 1/N)-(1 - cDNorm)/m)/(cDNorm-1/N);
}

int main (int argc, char *argv[]) {
  // Read command-line arguments
  std::string inputFilename = ""; //i
  std::string outputFilename = ""; //o
  int iterations = 1; //t
  int save_freq = 1; //S
  std::string mode = "EC";

  double x = std::log(0);
  // std::cout << "log(0) = " << x << std::endl;
  double y = std::exp(x);
  // std::cout << "exp(log(0)) = " << y << std::endl;
  int opt;

  // take command line arguments
  while ((opt = getopt(argc, argv, "i:o:t:S:H:")) != -1) {
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
      case 'S':
        save_freq = atoi(optarg);
        break;
      case 'H':
        mode = optarg;
        break;
      default:
        exit(EXIT_FAILURE);
    }
  }

  // Read Data from file
  
  std::ifstream inputFile(inputFilename); // File should contain two rows with n1 followed by n2
  if (inputFile.is_open()){
    std::ofstream outFile (outputFilename); // Reset file
    auto start = std::chrono::high_resolution_clock::now();
 
    bool verbose = false;
    outFile << "Iteration,Omega,sigmaOmega,maxOmega,cv2,time" << std::endl;

    // Parse input:
    std::string line;
    std::getline(inputFile,line);
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
    std::sort(cs.begin(),cs.end());
    std::reverse(cs.begin(),cs.end());
    std::cout << "rs: ";
    print_vector(rs);
    std::cout << "cs (sorted): ";
    print_vector(cs);
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
      alphas.push_back(alphaC(cs,j,m));
    }
    std::cout << "alphas:" << std::endl;
    print_vector(alphas);
    int maxR = 0;
    for(int r: rs){
      if(r > maxR){
        maxR = r;
      }
    }
    // int **array;
    // array = new int *[10];
    // for(int i = 0; i <10; i++)
    //   array[i] = new int[10];
    // double lnHVals[n][m][maxR + 1];
    double **lnHVals;
    lnHVals = new double *[n];
    for(int j = 0; j < n; j++){
      double Nt = 0;
      double Cs = 0;
      for(int k = j + 1; k < n; k++){
        Nt += cs[k];
      }
      for(int k = j + 1; k < n; k++){
        Cs += (cs[k]-Nt/(n - 1 - j))*(cs[k]-Nt/(n - 1 - j));
      }
      double lambda = Nt/(m*(n - 1 - j));
      double A = lambda*(1-lambda)/2;
      double eta;
      if(j < n - 1){
        eta = (1-Cs/(2*A*m*(n - 1 - j)))/(2*A*m*(n - 1 - j));
      }else{
        eta = 0;
      }
      // std::cout<<eta<<" "<<Nt<<std::endl;
      lnHVals[j] = new double[maxR + 1];
      for(int rr = 0; rr < maxR + 1; rr++){
        if(mode == "EC"){
          lnHVals[j][rr] = std::lgamma(alphas[j]+1) - std::lgamma(rr + 1) - std::lgamma(alphas[j] - rr + 1); // EC-based
        }else if(mode == "GC"){
          lnHVals[j][rr] = std::lgamma((n-1-j)+1) - std::lgamma(rr + 1) - std::lgamma((n-1-j) - rr + 1); // EC-based
        }else if(mode == "CGM"){
          lnHVals[j][rr] = std::lgamma((n-1-j)+1) - std::lgamma(rr+1) - std::lgamma((n-1-j)-rr+1) + 0.5*eta*(rr - Nt/m)*(rr - Nt/m); // CGM-Based
        }else{
          std::cout << "Unrecognized mode "<<mode<<std::endl;
          break;
        }
        // lnHVals[j][rr] = std::lgamma(alphas[j]+1) - std::lgamma(rr + 1) - std::lgamma(alphas[j] - rr + 1); // EC-based
        
        // std::cout<<std::lgamma((n-1-j)+1)<<" "<<std::lgamma(rr+1)<<" "<<std::lgamma((n-1-j)-rr+1)<<" "<<1/2*eta*(rr - Nt/m)*(rr - Nt/m)<<" "<<(rr - Nt/m)*(rr - Nt/m)<<std::endl;
        if(verbose){
          std::cout<<"hVals["<<j<<","<< rr<<"]="<<lnHVals[j][rr]<<std::endl;
        }
      }
    }
    // Precompute the Gale-Ryser Bound (gr[colNum,i] = \sum_{l=1}^i \sum_{j=colNum+1}^n 1{c_j >= l})
    int **GRBound;
    GRBound = new int *[n];
    for(int colNum = 0; colNum < n; colNum++){
      GRBound[colNum] = new int[m];
      GRBound[colNum][0] = 0;
      for(int j = colNum+1; j < n; j++){ // \sum_{j=colNum+1}^n 1{c_j >= 1}
        if(cs[j] >= 1){
          GRBound[colNum][0]++;
        }
      }
      for(int l = 1; l < m; l++){
        GRBound[colNum][l] = GRBound[colNum][l-1];
        for(int j = colNum+1; j < n; j++){ // \sum_{j=colNum+1}^n 1{c_j >= l}
          if(cs[j] >= l + 1){
            GRBound[colNum][l]++;
          }
        }
      }
    }
    // Instantiate lnGVals
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
          lnGVals[i][sl][sc] = -INFINITY;
        }
      }
    }
    int num_count = 0;
    for(int t = 1; t <= iterations; t++){
      mLogProb = 0;
      std::vector<std::vector<int>> table;
      // for(int i = 0; i < m+1; i++){
      //   for(int sl = 0; sl < maxC+1; sl++){
      //     for(int sc = 0; sc < maxC + 1; sc++){
      //       lnGVals[i][sl][sc] = -INFINITY;
      //     }
      //   }
      // }
      sampleTable(table,mLogProb,rs,cs,lnHVals,lnGVals,GRBound,verbose);
      // if(t >= 1212){
      //   std::cout << "Iteration "<<t<<std::endl;
      //   sampleTable(table,mLogProb,rs,cs,lnHVals,lnGVals,GRBound,true);
      // }else{
      //   sampleTable(table,mLogProb,rs,cs,lnHVals,lnGVals,GRBound,false);
      // }
      // if(table[0][3] == 1){
      //   num_count++;
      // }
      // std::cout << "Table sampled with probability " << prob<< " -> "<< (1/prob)<< std::endl;
      if(verbose){
        std::cout << "Table sampled with -ln(p) = " << mLogProb << std::endl;
        print_table(table);
      }
      // std::cout<< "Hash: "<< zeroOneTableHash(table) << std::endl;
      // tableHashes.push_back(zeroOneTableHash(table));
      mLogProbs.push_back(mLogProb);
      if(int(std::sqrt(t)) == std::sqrt(t)){
        // print_table(table);
        // std::cout << mLogProb << std::endl;
        // double pSum = std::accumulate(mLogProbs.begin(), mLogProbs.end(), 0.0);
        // double pMean = pSum / mLogProbs.size();
        // double dOff = std::floor(pMean/std::log(10));
        // std::vector<double> amounts;
        // for(double lP : mLogProbs){
        //   amounts.push_back(std::exp(lP - dOff*std::log(10)));
        // }
        // double sum = std::accumulate(amounts.begin(), amounts.end(), 0.0);
        // double mean = sum / amounts.size();

        // std::vector<double> diff(amounts.size());
        // std::transform(amounts.begin(), amounts.end(), diff.begin(), [mean](double x) { return x - mean; });
        // double sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
        // double stdev = std::sqrt(sq_sum / amounts.size());
        // std::cout << "Iteration: " << t <<  " avg: " << mean << "+/-"<< stdev/std::sqrt(amounts.size()) << " x 10^" << dOff << std::endl;
        
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
        std::ofstream outFile (outputFilename, std::ios_base::app);
        // Get ending timepoint
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(stop - start);
        outFile << t <<  "," << mean*std::exp(max - dShift*std::log(10)) << "E10+"<< dShift << ","<< stdev/std::sqrt(amounts.size())*std::exp(max - dShift*std::log(10)) << "E10+"<< dShift <<","<<std::exp(max - maxDShift*std::log(10))<<"E+"<<maxDShift<<"," << cv2 <<","<<duration.count()<< std::endl;  
        if(int(std::sqrt(t)) == int(std::sqrt(iterations))){
          std::cout <<"Final Result: "<< t <<  "," << mean*std::exp(max - dShift*std::log(10)) << "E10+"<< dShift << ","<< stdev/std::sqrt(amounts.size())*std::exp(max - dShift*std::log(10)) << "E10+"<< dShift <<","<<std::exp(max - maxDShift*std::log(10))<<"E+"<<maxDShift<<"," << cv2 <<","<<duration.count()<< std::endl;  
        }
        // sum = std::accumulate(mLogProbs.begin(), mLogProbs.end(), 0.0);
        // mean = sum / mLogProbs.size();

        // std::vector<double> pdiff(mLogProbs.size());
        // std::transform(mLogProbs.begin(), mLogProbs.end(), pdiff.begin(), [mean](double x) { return x - mean; });
        // sq_sum = std::inner_product(pdiff.begin(), pdiff.end(), pdiff.begin(), 0.0);
        // stdev = std::sqrt(sq_sum / mLogProbs.size());
        // std::cout << "Iteration: " << t <<  " avg: " << mean << "+/-"<< stdev/std::sqrt(mLogProbs.size()) << std::endl;
      }
    }
    // Hash checking
    // std::vector<int> uniqueHashes;
    // std::vector<double> uniqueHashProbs;
    // for(int i = 0; i < tableHashes.size(); i++){
    //   if (std::find(uniqueHashes.begin(), uniqueHashes.end(), tableHashes[i]) == uniqueHashes.end()) {
    //     uniqueHashes.push_back(tableHashes[i]);
    //     uniqueHashProbs.push_back(std::exp(-mLogProbs[i]));
    //   }
    // }
    // for(int i = 0; i < uniqueHashes.size(); i++){
    //   double probFound = 0;
    //   for(int hash2: tableHashes){
    //     if(uniqueHashes[i] == hash2){
    //       probFound++;
    //     }
    //   }
    //   probFound /= iterations;
    //   std::cout << uniqueHashes[i] << " " << probFound << " " <<uniqueHashProbs[i]<<std::endl;
    // }
    // print_vector(uniqueHashes);
  }
}


