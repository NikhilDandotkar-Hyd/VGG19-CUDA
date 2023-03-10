#include "image.h"


#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>


using namespace std;

std::vector<int> labelList;
std::vector<image> imageList;
std::vector<string> AllFileNames;
std::vector<string>LabelEncode;
std::map<string,int> LabelMap; 


image make_empty_image(int w, int h, int c)
{
    image out;
    out.data = 0;
    out.h = h;
    out.w = w;
    out.c = c;
    return out;
}
image make_empty_batch_image(int w, int h, int c, int b)
{
    image out;
    out.data = 0;
    out.h = h;
    out.w = w;
    out.c = c;
    out.bs = b;
    return out;
}
void free_image(image m)
{
    if (m.data) {
        free(m.data);
    }
}

float get_pixel(image m, int x, int y, int c)
{
    assert(x < m.w && y < m.h && c < m.c);
    return m.data[c*m.h*m.w + y*m.w + x];
}
void set_pixel(image m, int x, int y, int c, float val)
{
    if (x < 0 || y < 0 || c < 0 || x >= m.w || y >= m.h || c >= m.c) return;
    assert(x < m.w && y < m.h && c < m.c);
    m.data[c*m.h*m.w + y*m.w + x] = val;
}

void add_pixel(image m, int x, int y, int c, float val)
{
    assert(x < m.w && y < m.h && c < m.c);
    m.data[c*m.h*m.w + y*m.w + x] += val;
}

image make_image(int w, int h, int c)
{
    image out = make_empty_image(w, h, c);
    out.data = (float *)calloc(h*w*c, sizeof(float));
    return out;
}

image make_bach_image(int w, int h, int c, int b)
{
    image out = make_empty_batch_image(w, h, c, b);
    out.data = (float *)calloc(h*w*c*b, sizeof(float));
    return out;
}


image resize_image(image im, int w, int h)
{
    image resized = make_image(w, h, im.c);
    image part = make_image(w, im.h, im.c);
    int r, c, k;
    float w_scale = (float)(im.w - 1) / (w - 1);
    float h_scale = (float)(im.h - 1) / (h - 1);
    for (k = 0; k < im.c; ++k) {
        for (r = 0; r < im.h; ++r) {
            for (c = 0; c < w; ++c) {
                float val = 0;
                if (c == w - 1 || im.w == 1) {
                    val = get_pixel(im, im.w - 1, r, k);
                }
                else {
                    float sx = c*w_scale;
                    int ix = (int)sx;
                    float dx = sx - ix;
                    val = (1 - dx) * get_pixel(im, ix, r, k) + dx * get_pixel(im, ix + 1, r, k);
                }
                set_pixel(part, c, r, k, val);
            }
        }
    }
    for (k = 0; k < im.c; ++k) {
        for (r = 0; r < h; ++r) {
            float sy = r*h_scale;
            int iy = (int)sy;
            float dy = sy - iy;
            for (c = 0; c < w; ++c) {
                float val = (1 - dy) * get_pixel(part, c, iy, k);
                set_pixel(resized, c, r, k, val);
            }
            if (r == h - 1 || im.h == 1) continue;
            for (c = 0; c < w; ++c) {
                float val = dy * get_pixel(part, c, iy + 1, k);
                add_pixel(resized, c, r, k, val);
            }
        }
    }

    free_image(part);
    return resized;
}

image load_image_stb(string filename, int channels)
{
    int w, h, c;
    unsigned char *data = stbi_load(filename.c_str(), &w, &h, &c, channels);
    if (!data) {
        fprintf(stderr, "Cannot load image \"%s\"\nSTB Reason: %s\n", filename.c_str(), stbi_failure_reason());
        exit(0);
    }
    if (channels) c = channels;
    int i, j, k;
    image im = make_image(w, h, c);
    for (k = 0; k < c; ++k) 
    {
        for (j = 0; j < h; ++j)
        {
            for (i = 0; i < w; ++i)
            {
                int dst_index = i + w*j + w*h*k;
                int src_index = k + c*i + c*w*j;
                im.data[dst_index] = (float)data[src_index] / 255.;
            }
        }
    }
    free(data);
    return im;
}

image load_image(string filename, int w, int h, int c)
{
    image out = load_image_stb(filename, c);
    if ((h && w) && (h != out.h || w != out.w))
    {
        image resized = resize_image(out, w, h);
        free_image(out);
        out = resized;
    }
    return out;
}

image readImage(string imageName, string mainDirectory)
{
    image image_t;
    string totalFilePath = mainDirectory + imageName;
    image_t = load_image(totalFilePath, 224, 224, 3);
    return image_t;
}

image GetBatch(int index, int batchSize, string mainDirectory)
{
    labelList.clear();
    int start = index;
    int end = index + batchSize;
    if(end > AllFileNames.size())
    {
        end = AllFileNames.size();
        
    }
    image batchImages;
    int bc=0;
    image output = make_bach_image(224, 224, 3, 32);
    for (int it = start; it < end; it++)
    {   
        //cout<< "Done.Training dataset size" <<AllFileNames.size()<<endl;
        string totalName = AllFileNames[it];
        int seperatorIndex = totalName.find("_", 0);
        string label = totalName.substr(0, seperatorIndex);
        string imageName = totalName.substr(seperatorIndex+1,totalName.length() - seperatorIndex - 1);
        string imageSubFolder = label + "/" + imageName;
        batchImages = readImage(imageSubFolder,mainDirectory);
		if(LabelMap.find(label) == LabelMap.end())
		{
				continue;
		}
		for (int k = 0; k < batchImages.c; ++k)
        {
            for (int j = 0; j < batchImages.h; ++j)
            {
                for (int i = 0; i < batchImages.w; ++i) 
                {
                    int dst_index = i + batchImages.w*j + batchImages.w*batchImages.h*k + batchImages.w *batchImages.h *batchImages.c * bc;
                    int src_index = i + batchImages.w*j + batchImages.w*batchImages.h*k;
                    output.data[dst_index] = batchImages.data[src_index];
                    //cout << "Element at x["<< bc << "][" << k << "][" << j << "][" << i << "] = " << output.data[i + batchImages.w*j + batchImages.w*batchImages.h*k + batchImages.w *batchImages.h *batchImages.c * 1] << endl; 
                }
            }
        }
		labelList.push_back(LabelMap[label]);
		bc = bc+1;
        free(batchImages.data);
    }
    return output;
}
void ListFiles(const std::string &path, string subDirectory, std::function<void(const std::string &)> cb) 
{
    if (auto dir = opendir(path.c_str()))
    {
        while (auto f = readdir(dir))
        {
            if (!f->d_name || f->d_name[0] == '.') continue;
            if (f->d_type == DT_DIR){
				if(LabelMap.find(f->d_name) == LabelMap.end())
				{
					LabelEncode.push_back(f->d_name);
					LabelMap[f->d_name] = LabelEncode.size()-1;
					ListFiles(path + f->d_name + "/",f->d_name, cb);	 
                }
           }
                
            else if (f->d_type == DT_REG)
            {
                // Get the last subFolder of this
                cb(subDirectory+"_"+f->d_name);
            }
        }
    closedir(dir);
    }
}
/*void ListLabels(const std::string &path, string LabelsNames, std::function<void(const std::string &)> Lb) 
{
    if (auto dir = opendir(path.c_str()))
    {
        while (auto f = readdir(dir))
        {
            if (!f->d_name || f->d_name[0] == '.') continue;
            if (f->d_type == DT_DIR)
                labelList.push_back(f->d_name);
                //cout << (f->d_name) << endl;
                
        }
    closedir(dir);
    }
}*/
