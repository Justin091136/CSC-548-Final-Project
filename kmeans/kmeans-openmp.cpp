/* K-means with OpenMP */
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <string>
#include <regex>
#include <limits>
#include <chrono>
#include <iomanip>
#include <sys/stat.h>
#include <sys/types.h>

using namespace std;

struct Point
{
    vector<double> coords;
    int cluster = -1;
    Point() : cluster(-1) {}
    Point(const vector<double> &c) : coords(c), cluster(-1) {}
};

struct Centroid
{
    vector<double> coords;
};

// Parse k from filename like: data_k3.csv
int extract_k_from_filename(const string &filename)
{
    smatch match;
    regex pattern("k(\\d+)");
    if (regex_search(filename, match, pattern))
    {
        return stoi(match[1]);
    }
    else
    {
        cerr << "Error: Cannot extract k from filename: " << filename << endl;
        exit(1);
    }
}

// Load CSV
vector<Point> load_csv(const string &filename)
{
    vector<Point> points;
    ifstream file(filename);
    if (!file.is_open())
    {
        cerr << "Error: Failed to open input file: " << filename << endl;
        exit(1);
    }
    string line;
    while (getline(file, line))
    {
        stringstream ss(line);
        string val;
        vector<double> coords;
        while (getline(ss, val, ','))
        {
            coords.push_back(stod(val));
        }
        points.emplace_back(coords);
    }
    return points;
}

// Compute squared distance between point and centroid
inline double compute_distance(const Point &p, const Centroid &c)
{
    double sum = 0.0;
    for (int i = 0; i < p.coords.size(); ++i)
    {
        double diff = p.coords[i] - c.coords[i];
        sum += diff * diff;
    }
    return sum;
}

void assign_clusters(vector<Point> &points, const vector<Centroid> &centroids)
{
#pragma omp parallel for
    for (int i = 0; i < points.size(); ++i)
    {
        double min_dist = numeric_limits<double>::max();
        int best_cluster = -1;
        for (int j = 0; j < centroids.size(); ++j)
        {
            double dist = compute_distance(points[i], centroids[j]);
            if (dist < min_dist)
            {
                min_dist = dist;
                best_cluster = j;
            }
        }
        points[i].cluster = best_cluster;
    }
}

void update_centroids(vector<Point> &points, vector<Centroid> &centroids, int k)
{
    int n = points.size();
    int dim = points[0].coords.size();

    vector<vector<double>> sum_coords(k, vector<double>(dim, 0.0));
    vector<int> count(k, 0);

#pragma omp parallel
    {
        vector<vector<double>> local_sum(k, vector<double>(dim, 0.0));
        vector<int> local_count(k, 0);

#pragma omp for nowait
        for (int i = 0; i < n; ++i)
        {
            int cid = points[i].cluster;
            for (int d = 0; d < dim; ++d)
            {
                local_sum[cid][d] += points[i].coords[d];
            }
            local_count[cid]++;
        }

#pragma omp critical
        {
            for (int c = 0; c < k; ++c)
            {
                for (int d = 0; d < dim; ++d)
                {
                    sum_coords[c][d] += local_sum[c][d];
                }
                count[c] += local_count[c];
            }
        }
    }

    for (int c = 0; c < k; ++c)
    {
        if (count[c] > 0)
        {
            for (int d = 0; d < dim; ++d)
            {
                centroids[c].coords[d] = sum_coords[c][d] / count[c];
            }
        }
    }
}

bool has_converged(const vector<Centroid> &old_centroids, const vector<Centroid> &new_centroids, double epsilon = 1e-4)
{
    for (int i = 0; i < old_centroids.size(); ++i)
    {
        if (compute_distance(Point{old_centroids[i].coords}, new_centroids[i]) > epsilon)
        {
            return false;
        }
    }
    return true;
}

void run_kmeans(vector<Point> &points, int k, int max_iters = 100)
{
    int n = points.size();
    int dim = points[0].coords.size();
    vector<Centroid> centroids(k, Centroid{vector<double>(dim, 0.0)});

    // srand(42);
    //  Random initialization
    for (int i = 0; i < k; ++i)
    {
        int rand_idx = rand() % n;
        centroids[i].coords = points[rand_idx].coords;
    }

    for (int iter = 0; iter < max_iters; ++iter)
    {
        vector<Centroid> prev_centroids = centroids;

        assign_clusters(points, centroids);
        update_centroids(points, centroids, k);

        if (has_converged(prev_centroids, centroids))
        {
            break;
        }
        if (iter == max_iters - 1)
        {
            ;
        }
    }
}

void print_debug_summary(const vector<Point> &points, int k)
{
    vector<int> count(k, 0);
    cout << "\n--- Cluster Counts ---\n";
    for (const auto &p : points)
        count[p.cluster]++;
    for (int i = 0; i < k; ++i)
        cout << "Cluster " << i << ": " << count[i] << " points\n";
}

// Helper to create a directory if it doesn't exist
void create_dir_if_not_exists(const string &dir_path)
{
    struct stat info;
    if (stat(dir_path.c_str(), &info) != 0)
    {
        // Directory does not exist
        if (mkdir(dir_path.c_str(), 0755) != 0)
        {
            cerr << "Error: Failed to create directory: " << dir_path << endl;
            exit(1);
        }
    }
    else if (!(info.st_mode & S_IFDIR))
    {
        cerr << "Error: '" << dir_path << "' exists but is not a directory.\n";
        exit(1);
    }
}

// Save execution times to CSV under "results/runtime_csv" directory
void save_execution_times(const vector<double> &times, const string &version_name)
{
    string results_dir = "results";
    string runtime_csv_dir = results_dir + "/runtime_csv";

    create_dir_if_not_exists(results_dir);
    create_dir_if_not_exists(runtime_csv_dir);

    string output_file = runtime_csv_dir + "/execution_times_" + version_name + ".csv";
    ofstream ofs(output_file);
    if (!ofs.is_open())
    {
        cerr << "Error: Failed to open output file: " << output_file << endl;
        exit(1);
    }

    ofs << "trial,time_ms\n";
    for (int i = 0; i < times.size(); ++i)
    {
        ofs << (i + 1) << "," << fixed << setprecision(4) << times[i] << "\n";
    }
    ofs.close();
}

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        cerr << "Usage: ./kmeans <filename>\n";
        return 1;
    }

    string filename = argv[1];
    int k = extract_k_from_filename(filename);
    vector<Point> original_points = load_csv(filename);

    const int trials = 100;
    double total_time = 0;
    vector<double> trial_times; // Store individual execution times

    for (int t = 0; t < trials; ++t)
    {
        vector<Point> points = original_points;
        auto start = chrono::high_resolution_clock::now();
        srand(42 + t);

        run_kmeans(points, k);

        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double, milli> elapsed = end - start;
        total_time += elapsed.count();
        trial_times.push_back(elapsed.count());

        if ((t == 0) && (points.size() <= 500))
        {
            print_debug_summary(points, k);
        }
    }

    cout << "Average time over " << trials << " runs: " << fixed << setprecision(3)
         << total_time / trials << " ms" << endl;

    save_execution_times(trial_times, "openmp");

    return 0;
}
