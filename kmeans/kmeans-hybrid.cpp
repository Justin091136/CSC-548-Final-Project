/* K-means with MPI+OpenMP */
#include <mpi.h>
#include <omp.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <limits>
#include <string>
#include <regex>
#include <iomanip>
#include <chrono>
#include <iomanip>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>

using namespace std;

// =========================
// Structure Definitions
// =========================
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

// Global dimension
int dims = 0;

// =========================
// Extract k from filename (e.g., data_k3.csv => k=3)
// =========================
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

// =========================
// Load CSV and set global dims
// =========================
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
        vector<double> row;
        // Read each column
        while (getline(ss, val, ','))
        {
            row.push_back(stod(val));
        }
        // Set dims on the first row
        if (dims == 0)
        {
            dims = row.size();
        }
        points.emplace_back(row);
    }
    return points;
}

// =========================
// Compute squared Euclidean distance between a Point and a Centroid
// =========================
inline double compute_distance(const Point &p, const Centroid &c)
{
    double sum = 0.0;
    for (int d = 0; d < dims; ++d)
    {
        double diff = p.coords[d] - c.coords[d];
        sum += diff * diff;
    }
    return sum;
}

// =========================
// Assign clusters (parallel with OpenMP)
// =========================
void assign_clusters(vector<Point> &points, const vector<Centroid> &centroids)
{
#pragma omp parallel for
    for (int i = 0; i < (int)points.size(); ++i)
    {
        double min_dist = numeric_limits<double>::max();
        int best_cluster = -1;
        for (int j = 0; j < (int)centroids.size(); ++j)
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

// =========================
// Update centroids (MPI + OpenMP)
// =========================
void update_centroids(const vector<Point> &points, vector<Centroid> &centroids, int k, MPI_Comm comm)
{
    // local_combined[c*(dims+1) + d] => sum for dimension d of cluster c
    // local_combined[c*(dims+1) + dims] => count of cluster c
    vector<double> local_combined(k * (dims + 1), 0.0);

// ===== OpenMP thread-local accumulation =====
#pragma omp parallel
    {
        vector<double> thread_combined(k * (dims + 1), 0.0);

#pragma omp for nowait
        for (int i = 0; i < (int)points.size(); ++i)
        {
            int cid = points[i].cluster;
            if (cid < 0 || cid >= k)
                continue;

            int offset = cid * (dims + 1);
            for (int d = 0; d < dims; ++d)
            {
                thread_combined[offset + d] += points[i].coords[d];
            }
            thread_combined[offset + dims] += 1.0;
        }

#pragma omp critical
        {
            for (int c = 0; c < k; ++c)
            {
                int offset = c * (dims + 1);
                for (int d = 0; d < dims; ++d)
                {
                    local_combined[offset + d] += thread_combined[offset + d];
                }
                local_combined[offset + dims] += thread_combined[offset + dims];
            }
        }
    }

    // ===== MPI Allreduce (only once) =====
    vector<double> global_combined(k * (dims + 1), 0.0);
    MPI_Allreduce(local_combined.data(), global_combined.data(), k * (dims + 1), MPI_DOUBLE, MPI_SUM, comm);

    // ===== Update centroids =====
    for (int c = 0; c < k; ++c)
    {
        int offset = c * (dims + 1);
        double count = global_combined[offset + dims];
        if (count > 0)
        {
            centroids[c].coords.resize(dims);
            for (int d = 0; d < dims; ++d)
            {
                centroids[c].coords[d] = global_combined[offset + d] / count;
            }
        }
    }
}

// =========================
// Check for convergence
// =========================
bool has_converged(const vector<Centroid> &old_c, const vector<Centroid> &new_c, double epsilon = 1e-4)
{
    for (int i = 0; i < (int)old_c.size(); i++)
    {
        double sum = 0.0;
        for (int d = 0; d < dims; d++)
        {
            double diff = old_c[i].coords[d] - new_c[i].coords[d];
            sum += diff * diff;
        }
        if (sum > epsilon * epsilon)
        {
            return false;
        }
    }
    return true;
}

// =========================
// Run k-means algorithm
// =========================
void run_kmeans(vector<Point> &local_points, vector<Centroid> &centroids, int k, int max_iter, MPI_Comm comm)
{
    int rank;
    MPI_Comm_rank(comm, &rank);
    for (int iter = 0; iter < max_iter; iter++)
    {
        vector<Centroid> old_c = centroids;

        assign_clusters(local_points, centroids);
        update_centroids(local_points, centroids, k, comm);

        bool local_ok = has_converged(old_c, centroids);
        int local_val = local_ok ? 1 : 0;
        int global_ok = 0;
        MPI_Allreduce(&local_val, &global_ok, 1, MPI_INT, MPI_LAND, comm);

        if (global_ok)
        {
            break;
        }
    }
}

// =========================
// Gather results to rank 0 using Gatherv
// =========================
vector<Point> gather_results_to_rank0(const vector<Point> &local_points, int total_points,
                                      int rank, int size, MPI_Comm comm)
{
    int local_n = (int)local_points.size();

    // Flatten each point (dims + 1) = coordinates + cluster
    vector<double> send_buf(local_n * (dims + 1));
    for (int i = 0; i < local_n; i++)
    {
        for (int d = 0; d < dims; d++)
        {
            send_buf[i * (dims + 1) + d] = local_points[i].coords[d];
        }
        send_buf[i * (dims + 1) + dims] = (double)local_points[i].cluster;
    }

    vector<int> recv_counts, displs;
    if (rank == 0)
    {
        recv_counts.resize(size);
        displs.resize(size);
        int base = total_points / size;
        int remain = total_points % size;
        int prev = 0;
        for (int i = 0; i < size; ++i)
        {
            int n = base + (i < remain ? 1 : 0);
            recv_counts[i] = n * (dims + 1);
            displs[i] = prev;
            prev += recv_counts[i];
        }
    }

    vector<double> gathered_buf;
    if (rank == 0)
    {
        gathered_buf.resize(total_points * (dims + 1));
    }

    // Gatherv
    MPI_Gatherv(send_buf.data(), send_buf.size(), MPI_DOUBLE,
                gathered_buf.data(),
                (rank == 0 ? recv_counts.data() : nullptr),
                (rank == 0 ? displs.data() : nullptr),
                MPI_DOUBLE, 0, comm);

    vector<Point> all_points;
    if (rank == 0)
    {
        all_points.resize(total_points);
        for (int i = 0; i < total_points; i++)
        {
            vector<double> c(dims);
            for (int d = 0; d < dims; d++)
            {
                c[d] = gathered_buf[i * (dims + 1) + d];
            }
            int cluster = (int)gathered_buf[i * (dims + 1) + dims];
            all_points[i] = Point(c);
            all_points[i].cluster = cluster;
        }
    }
    return all_points;
}

void print_debug_summary(const vector<Point> &points, int k)
{
    vector<int> count(k, 0);
    for (auto &p : points)
    {
        count[p.cluster]++;
    }
    cout << "\n--- Cluster Counts ---\n";
    for (int i = 0; i < k; i++)
    {
        cout << "Cluster " << i << ": " << count[i] << " points\n";
    }
}

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
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    char hostname[256];
    gethostname(hostname, sizeof(hostname));

    /*

#ifdef _OPENMP
#pragma omp parallel
    {
        int omp_rank = omp_get_thread_num();
        int omp_size = omp_get_num_threads();
        if (omp_rank == 0)
        { // only one thread prints per process
            cout << "Rank " << rank << "/" << size
                 << " running on " << hostname
                 << ", PID " << getpid()
                 << ", OpenMP threads: " << omp_size
                 << endl;
        }
    }
#else
    if (rank == 0)
    {
        cout << "Compiled without OpenMP support." << endl;
    }
#endif
    */

    if (argc < 2)
    {
        if (rank == 0)
        {
            cerr << "Usage: mpirun -np <n> ./kmeans_hybrid_multi <filename>\n";
        }
        MPI_Finalize();
        return 1;
    }

    string filename = argv[1];
    // rank=0 extracts k
    int k = 0;
    if (rank == 0)
    {
        k = extract_k_from_filename(filename);
    }
    MPI_Bcast(&k, 1, MPI_INT, 0, MPI_COMM_WORLD);

    vector<Point> full_data;
    vector<double> send_buf;
    int total_points = 0;

    // rank=0 loads data & flattens
    if (rank == 0)
    {
        full_data = load_csv(filename);
        total_points = (int)full_data.size();

        send_buf.resize(total_points * dims);
        for (int i = 0; i < total_points; i++)
        {
            for (int d = 0; d < dims; d++)
            {
                send_buf[i * dims + d] = full_data[i].coords[d];
            }
        }
    }

    // Broadcast total_points and dims
    MPI_Bcast(&total_points, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&dims, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Determine how many points this rank is responsible for
    int base = total_points / size;
    int remain = total_points % size;
    int local_n = base + (rank < remain ? 1 : 0);

    // Scatterv
    vector<int> counts(size), displs(size);
    if (rank == 0)
    {
        int prev = 0;
        for (int i = 0; i < size; i++)
        {
            int n = base + (i < remain ? 1 : 0);
            counts[i] = n * dims;
            displs[i] = prev;
            prev += counts[i];
        }
    }

    vector<double> recv_buf(local_n * dims);
    MPI_Scatterv(send_buf.data(), counts.data(), displs.data(), MPI_DOUBLE,
                 recv_buf.data(), recv_buf.size(), MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    // Reconstruct local points from received buffer
    vector<Point> local_data(local_n);
    for (int i = 0; i < local_n; i++)
    {
        vector<double> tmp(dims);
        for (int d = 0; d < dims; d++)
        {
            tmp[d] = recv_buf[i * dims + d];
        }
        local_data[i] = Point(tmp);
    }

    // Backup original local data for reuse across multiple trials
    vector<Point> original_local_data = local_data;

    // Multiple trials
    const int trials = 100;
    double total_time = 0.0;
    vector<double> trial_times;

    for (int t = 0; t < trials; t++)
    {
        // 1) Reset local data => clear previous cluster assignment
        local_data = original_local_data;

        // 2) Randomly initialize centroids (only by rank 0)
        vector<Centroid> centroids(k, Centroid{});
        srand(42 + t);
        if (rank == 0)
        {
            // srand(42 + t);  // Use different seed for each trial
            for (int i = 0; i < k; i++)
            {
                centroids[i].coords.resize(dims);
                int idx = rand() % total_points;
                for (int d = 0; d < dims; d++)
                {
                    centroids[i].coords[d] = send_buf[idx * dims + d];
                }
            }
        }

        // Broadcast initial centroids
        for (int i = 0; i < k; i++)
        {
            if ((int)centroids[i].coords.size() < dims)
            {
                centroids[i].coords.resize(dims, 0.0);
            }
            MPI_Bcast(centroids[i].coords.data(), dims, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        }

        MPI_Barrier(MPI_COMM_WORLD);
        double start_time = MPI_Wtime();

        // 3) Run K-means
        run_kmeans(local_data, centroids, k, 100, MPI_COMM_WORLD);

        MPI_Barrier(MPI_COMM_WORLD);
        double end_time = MPI_Wtime();
        double elapsed_ms = (end_time - start_time) * 1000.0;
        total_time += elapsed_ms;

        // 4) Optionally gather and print result of this trial
        //    Here we only print trial 0 or the last trial

        if (t == 0)
        {
            vector<Point> all_pts = gather_results_to_rank0(local_data, total_points, rank, size, MPI_COMM_WORLD);
            if (rank == 0)
            {

                if ((int)all_pts.size() <= 500)
                {
                    print_debug_summary(all_pts, k);
                }
            }
        }

        if (rank == 0)
        {
            trial_times.push_back(elapsed_ms);
        }
    }

    // Print average time of all trials
    if (rank == 0)
    {
        cout << "Average time over " << trials << " runs: "
             << (total_time / trials) << " ms" << endl;
        save_execution_times(trial_times, "kmeans_hybrid");
    }

    MPI_Finalize();
    return 0;
}
