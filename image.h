#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string>
#include <functional>
#include <dirent.h>
#include <bits/stdc++.h>
#include <vector>
struct image{
        int h;
        int w;
        int c;
	    int bs;
        float *data;
};

using namespace std;

extern std::vector<int> labelList;
extern std::vector<image> imageList;
extern std::vector<string> AllFileNames;
extern std::vector<string> LabelEncode;
extern std::map<string,int> LabelMap;

image make_empty_image(int w, int h, int c);

image make_empty_batch_image(int w, int h, int c, int b);

void free_image(image m);

float get_pixel(image m, int x, int y, int c);

void set_pixel(image m, int x, int y, int c, float val);

void add_pixel(image m, int x, int y, int c, float val);

image make_image(int w, int h, int c);

image make_bach_image(int w, int h, int c, int b);

image resize_image(image im, int w, int h);

image load_image_stb(string filename, int channels);

image load_image(string filename, int w, int h, int c);

image readImage(string imageName, string mainDirectory);

image GetBatch(int index, int batchSize, string mainDirectory);

//void ListLabels(const std::string &path, string LabelsNames, std::function<void(const std::string &)> Lb);

void ListFiles(const std::string &path, string subDirectory, std::function<void(const std::string &)> cb);