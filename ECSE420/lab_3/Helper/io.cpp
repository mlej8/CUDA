#include <iostream>
#include <fstream>
#include <string>

using namespace std;

void ReadCSV(const char *filename, char* data) {
  // open file for reading
  ifstream input_file(filename);
  const string delimiter = ",";
  if (input_file.is_open()) {
    string line;
    int counter = 0;
    while (getline(input_file, line)) {
      size_t comma_index;
      while ((comma_index = line.find(delimiter)) != string::npos) {
        data[counter] = stoi(line.substr(0, comma_index));
        counter++;
        line.erase(0, comma_index + delimiter.length());
      }
      // at the end there is still a number left
      data[counter] = stoi(line);
      counter++;
    }
    input_file.close();
  } else {
    cout << "Unable to open file" << endl;
  }
  cout << "Done reading " << filename << endl;
}

void WriteOutput(const char *output_filename, const int* data, const int file_length) { 
  ofstream output_fp(output_filename);
  string result;

  // write the file length as the first line
  result += to_string(file_length) + "\n";

  for (int i = 0; i < file_length; i++) {
    result += to_string(data[i]) + "\n";
  }
  output_fp << result;
  output_fp.close();
}