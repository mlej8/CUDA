#ifndef LAB_1_IO_H_
#define LAB_1_IO_H_

// read CSV file
void ReadCSV(const char *filename, char* data);

// write program output to a file
void WriteOutput(const char *output_filename, const char* data, const int file_length);

#endif