/* kmeans.cpp: sequential K-means*/
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
    Point(int dim) : coords(dim, 0.0), cluster(-1) {}
    Point(const vector<double> &c) : coords(c), cluster(-1) {}
};

struct Centroid
{
    vector<double> coords;
};

// Extract k from a filename like: data_k3.csv
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

// Reads a CSV file where each line has one or more comma-separated floating values.
// All lines must have the same number of columns, which will be the dimension.
vector<Point> load_csv(const string &filename)
{
    ifstream file(filename);
    if (!file.is_open())
    {
        cerr << "Error: Failed to open file: " << filename << endl;
        exit(1);
    }

    vector<Point> points;
    string line;
    int dimension = -1;

    while (getline(file, line))
    {
        if (line.empty())
            continue; // skip any empty lines
        stringstream ss(line);
        string val;
        vector<double> coordsInLine;

        // Split line by comma
        while (getline(ss, val, ','))
        {
            coordsInLine.push_back(stod(val));
        }
        // Determine dimension if not yet known
        if (dimension < 0)
        {
            dimension = static_cast<int>(coordsInLine.size());
        }
        else
        {
            // Check if each line has consistent number of columns
            if (static_cast<int>(coordsInLine.size()) != dimension)
            {
                cerr << "Error: Inconsistent column count in CSV." << endl;
                exit(1);
            }
        }
        // Create a Point with these coords
        points.emplace_back(coordsInLine);
    }
    return points;
}

// Compute the squared distance between a Point and a Centroid
inline double compute_distance(const Point &p, const Centroid &c)
{
    double dist_sq = 0.0;
    // Assuming p.coords.size() == c.coords.size()
    for (size_t d = 0; d < p.coords.size(); d++)
    {
        double diff = p.coords[d] - c.coords[d];
        dist_sq += diff * diff;
    }
    return dist_sq;
}

// Assign each point to the closest centroid
void assign_clusters(vector<Point> &points, const vector<Centroid> &centroids)
{
    for (auto &p : points)
    {
        double min_dist = numeric_limits<double>::max();
        int best_cluster = -1;
        for (int i = 0; i < static_cast<int>(centroids.size()); i++)
        {
            double dist_sq = compute_distance(p, centroids[i]);
            if (dist_sq < min_dist)
            {
                min_dist = dist_sq;
                best_cluster = i;
            }
        }
        p.cluster = best_cluster;
    }
}

// Update centroids based on the points' cluster assignments
void update_centroids(const vector<Point> &points, vector<Centroid> &centroids, int k)
{
    if (points.empty())
        return;
    int dim = static_cast<int>(points[0].coords.size());

    // sum_coords[c][d] accumulates coordinate sums for cluster c in dimension d
    vector<vector<double>> sum_coords(k, vector<double>(dim, 0.0));
    vector<int> count(k, 0);

    // Accumulate sums per cluster
    for (const auto &p : points)
    {
        int cid = p.cluster;
        if (cid < 0 || cid >= k)
            continue; // safety check
        for (int d = 0; d < dim; d++)
        {
            sum_coords[cid][d] += p.coords[d];
        }
        count[cid]++;
    }

    // Compute the average for each cluster
    for (int c = 0; c < k; c++)
    {
        // If no points are assigned to this cluster, we skip updating
        if (count[c] > 0)
        {
            for (int d = 0; d < dim; d++)
            {
                centroids[c].coords[d] = sum_coords[c][d] / count[c];
            }
        }
    }
}

// Check if the centroids have converged within a given threshold epsilon (using squared distance)
bool has_converged(const vector<Centroid> &old_centroids, const vector<Centroid> &new_centroids, double epsilon = 1e-4)
{
    if (old_centroids.size() != new_centroids.size())
        return false;
    for (size_t i = 0; i < old_centroids.size(); i++)
    {
        double dist_sq = 0.0;
        if (old_centroids[i].coords.size() != new_centroids[i].coords.size())
            return false;
        for (size_t d = 0; d < old_centroids[i].coords.size(); d++)
        {
            double diff = old_centroids[i].coords[d] - new_centroids[i].coords[d];
            dist_sq += diff * diff;
        }
        if (dist_sq > epsilon * epsilon)
        {
            return false;
        }
    }
    return true;
}

// Run K-means with maximum iterations
void run_kmeans(vector<Point> &points, int k, int max_iters = 200)
{
    if (points.empty())
        return;
    int n = static_cast<int>(points.size());
    int dim = static_cast<int>(points[0].coords.size());

    // Allocate centroids
    vector<Centroid> centroids(k);
    for (int c = 0; c < k; c++)
    {
        centroids[c].coords.resize(dim, 0.0);
    }

    // Initialize centroids by randomly picking points

    for (int i = 0; i < k; ++i)
    {
        int rand_idx = rand() % n;
        for (int d = 0; d < dim; d++)
        {
            centroids[i].coords[d] = points[rand_idx].coords[d];
        }
    }

    // Main loop
    for (int iter = 0; iter < max_iters; ++iter)
    {
        vector<Centroid> prev_centroids = centroids;

        assign_clusters(points, centroids);
        update_centroids(points, centroids, k);

        if (has_converged(prev_centroids, centroids))
        {
            // Uncomment if you want to see the iteration count
            // cout << "Converged after " << iter + 1 << " iterations." << endl;
            break;
        }
    }
}

// Print a summary of how many points ended up in each cluster
void print_debug_summary(const vector<Point> &points, int k)
{
    vector<int> count(k, 0);
    for (const auto &p : points)
    {
        if (p.cluster >= 0 && p.cluster < k)
        {
            count[p.cluster]++;
        }
    }
    cout << "\n--- Cluster Counts ---" << endl;
    for (int i = 0; i < k; ++i)
    {
        cout << "Cluster " << i << ": " << count[i] << " points" << endl;
    }
}

// Helper to create a directory if it doesn't exist
void create_dir_if_not_exists(const string &dir_path)
{
    struct stat info;
    if (stat(dir_path.c_str(), &info) != 0)
    {
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

string get_clean_test_name(const string &filename)
{
    // Remove path
    size_t last_slash = filename.find_last_of("/\\");
    string base = (last_slash == string::npos) ? filename : filename.substr(last_slash + 1);

    // Remove .csv
    size_t dot_pos = base.rfind('.');
    if (dot_pos != string::npos)
    {
        base = base.substr(0, dot_pos);
    }
    return base;
}

// Save execution times to CSV under "results/runtime_csv"
void save_execution_times(const vector<double> &times, const string &test_name, const string &version_name)
{
    string results_dir = "results";
    string runtime_csv_dir = results_dir + "/runtime_csv";
    string test_dir = runtime_csv_dir + "/" + test_name;

    create_dir_if_not_exists(results_dir);
    create_dir_if_not_exists(runtime_csv_dir);
    create_dir_if_not_exists(test_dir);

    string output_file = test_dir + "/execution_times_" + version_name + ".csv";
    std::ofstream ofs(output_file);
    if (!ofs.is_open())
    {
        std::cerr << "Error: Failed to open output file: " << output_file << std::endl;
        exit(1);
    }

    ofs << "trial,time_ms\n";
    for (int i = 0; i < (int)times.size(); ++i)
    {
        ofs << (i + 1) << "," << std::fixed << std::setprecision(3) << times[i] << "\n";
    }
    ofs.close();
}

bool save_result = false;
int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        cerr << "Usage: ./kmeans <filename>\n";
        return 1;
    }

    string filename = argv[1];
    int k = extract_k_from_filename(filename);

    const int trials = 100;
    double total_time_ms = 0.0;
    vector<double> trial_times;

    for (int t = 0; t < trials; ++t)
    {
        vector<Point> points = load_csv(filename);

        srand(42 + t);
        auto start = chrono::high_resolution_clock::now();
        run_kmeans(points, k);
        auto end = chrono::high_resolution_clock::now();

        chrono::duration<double, milli> elapsed = end - start;
        total_time_ms += elapsed.count();
        trial_times.push_back(elapsed.count());

        if (t == 0 && points.size() <= 500)
        {
            print_debug_summary(points, k);
        }
    }

    cout << "Average time over " << trials << " runs: "
         << (total_time_ms / trials) << " ms" << endl;

    if (save_result)
    {
        string test_name = get_clean_test_name(filename);
        save_execution_times(trial_times, test_name, "kmeans_seq");
    }

    return 0;
}