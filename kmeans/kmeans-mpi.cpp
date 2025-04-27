/* K-means with MPI */
#include <mpi.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <limits>
#include <string>
#include <iomanip>
#include <regex>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>

using namespace std;

// Each Point holds a dynamic-dimensional coordinate vector plus the assigned cluster
struct Point
{
    vector<double> coords;
    int cluster = -1;
    Point() : cluster(-1) {}
    Point(const vector<double> &c) : coords(c), cluster(-1) {}
};

// Each Centroid holds a dynamic-dimensional coordinate vector
struct Centroid
{
    vector<double> coords;
};

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

// This global variable will store the dimension count once we parse the first line
int dims = 0;

// Load CSV that may contain multiple columns (dimensions)
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

    // Parse each line, split by comma, and store it as a vector<double>
    while (getline(file, line))
    {
        stringstream ss(line);
        string val;
        vector<double> row;
        while (getline(ss, val, ','))
        {
            row.push_back(stod(val));
        }
        // The first parsed line sets the global dims
        if (dims == 0)
        {
            dims = row.size();
        }
        points.emplace_back(row);
    }
    return points;
}

// Compute squared distance between a Point and a Centroid across all dimensions
inline double compute_distance(const Point &p, const Centroid &c)
{
    double sum = 0.0;
    for (int d = 0; d < dims; d++)
    {
        double diff = p.coords[d] - c.coords[d];
        sum += diff * diff;
    }
    return sum;
}

// Assign each point to the nearest centroid
void assign_clusters(vector<Point> &points, const vector<Centroid> &centroids)
{
    for (auto &p : points)
    {
        double min_dist = numeric_limits<double>::max();
        int best_cluster = -1;
        for (int i = 0; i < (int)centroids.size(); ++i)
        {
            double dist = compute_distance(p, centroids[i]);
            if (dist < min_dist)
            {
                min_dist = dist;
                best_cluster = i;
            }
        }
        p.cluster = best_cluster;
    }
}

// Update centroids by averaging all points assigned to each cluster
void update_centroids(const vector<Point> &points, vector<Centroid> &centroids, int k, MPI_Comm comm)
{
    // local_combined stores both coordinate sums and the count for each cluster:
    // For cluster c, the segment c*(dims+1)...c*(dims+1)+(dims-1) is used for coordinate sums,
    // and the position c*(dims+1)+dims holds the count of points in that cluster.
    vector<double> local_combined(k * (dims + 1), 0.0);
    vector<double> global_combined(k * (dims + 1), 0.0);

    // 1) Accumulate local sums and counts
    for (const auto &p : points)
    {
        int cid = p.cluster;
        // If there's a possibility that cid can be -1 or invalid, check here:
        // if (cid < 0 || cid >= k) continue;

        // Add up coordinates
        for (int d = 0; d < dims; d++)
        {
            local_combined[cid * (dims + 1) + d] += p.coords[d];
        }
        // Increment the count (the last slot of each cluster block)
        local_combined[cid * (dims + 1) + dims] += 1.0;
    }

    // 2) Perform a single Allreduce to combine both sums and counts
    MPI_Allreduce(local_combined.data(),
                  global_combined.data(),
                  k * (dims + 1),
                  MPI_DOUBLE,
                  MPI_SUM,
                  comm);

    // 3) Recompute the centroid positions using the global sums and counts
    for (int i = 0; i < k; ++i)
    {
        double count = global_combined[i * (dims + 1) + dims];
        if (count > 0)
        {
            for (int d = 0; d < dims; d++)
            {
                centroids[i].coords[d] =
                    global_combined[i * (dims + 1) + d] / count;
            }
        }
    }
}

// Check if centroids have converged
bool has_converged(const vector<Centroid> &old_centroids, const vector<Centroid> &new_centroids, double epsilon = 1e-4)
{
    for (int i = 0; i < (int)old_centroids.size(); ++i)
    {
        double dist = 0.0;
        for (int d = 0; d < dims; d++)
        {
            double diff = old_centroids[i].coords[d] - new_centroids[i].coords[d];
            dist += diff * diff;
        }
        if (dist > epsilon * epsilon)
        {
            return false;
        }
    }
    return true;
}

// Gather the final (x, y, cluster) data on rank 0; extended to dims as well
// but we only need to gather coords and cluster
vector<Point> gather_results_to_rank0(const vector<Point> &local_points, int total_points, int rank, int size, MPI_Comm comm)
{
    int local_n = (int)local_points.size();

    // Flatten local points to store: [coord0, coord1, ..., coord(dims-1), cluster]
    // That means each point is dims + 1 in length
    vector<double> send_buf(local_n * (dims + 1));
    for (int i = 0; i < local_n; ++i)
    {
        for (int d = 0; d < dims; d++)
        {
            send_buf[i * (dims + 1) + d] = local_points[i].coords[d];
        }
        send_buf[i * (dims + 1) + dims] = (double)local_points[i].cluster;
    }

    // Prepare receive counts and displacements (for Gatherv)
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

    // Allocate buffer to gather all results (only on rank 0)
    vector<double> gathered_buf;
    if (rank == 0)
    {
        gathered_buf.resize(total_points * (dims + 1));
    }

    // Gather all flattened results to rank 0
    MPI_Gatherv(send_buf.data(), (int)send_buf.size(), MPI_DOUBLE,
                gathered_buf.data(), recv_counts.data(), displs.data(), MPI_DOUBLE,
                0, comm);

    // Reconstruct full vector<Point> from gathered buffer
    vector<Point> all_points;
    if (rank == 0)
    {
        all_points.resize(total_points);
        for (int i = 0; i < total_points; ++i)
        {
            vector<double> c(dims, 0.0);
            for (int d = 0; d < dims; d++)
            {
                c[d] = gathered_buf[i * (dims + 1) + d];
            }
            all_points[i].coords = c;
            all_points[i].cluster = (int)gathered_buf[i * (dims + 1) + dims];
        }
    }
    return all_points;
}

// Main K-means iteration
void run_kmeans(vector<Point> &local_points, vector<Centroid> &centroids, int k, int max_iters, MPI_Comm comm)
{
    int rank;
    MPI_Comm_rank(comm, &rank);

    for (int iter = 0; iter < max_iters; ++iter)
    {
        vector<Centroid> prev_centroids = centroids;

        assign_clusters(local_points, centroids);
        update_centroids(local_points, centroids, k, comm);

        bool local_converged = has_converged(prev_centroids, centroids);
        int global_converged = 0;
        int local_value = local_converged ? 1 : 0;

        // Check if all processes see convergence
        MPI_Allreduce(&local_value, &global_converged, 1, MPI_INT, MPI_LAND, comm);

        if (global_converged)
        {
            if (rank == 0)
            {
                ;
                // cout << "Converged after " << iter + 1 << " iterations." << endl;
            }
            break;
        }

        if (iter == max_iters - 1 && rank == 0)
        {
            ;
            // cout << "Did not converge within max iterations." << endl;
        }
    }
}

// Print how many points fall into each cluster (for debugging)
void print_debug_summary(const vector<Point> &points, int k)
{
    vector<int> count(k, 0);
    cout << "\n--- Cluster Counts ---\n";
    for (const auto &p : points)
    {
        count[p.cluster]++;
    }
    for (int i = 0; i < k; ++i)
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

    if (argc < 2)
    {
        if (rank == 0)
            cerr << "Usage: mpirun -np <n> ./kmeans-mpi <filename>\n";
        MPI_Finalize();
        return 1;
    }

    char hostname[256];
    gethostname(hostname, sizeof(hostname));

    /*

    cout << "Rank " << rank << "/" << size
         << " running on " << hostname
         << ", PID " << getpid()
         << endl;
    */

    string filename = argv[1];
    int k = extract_k_from_filename(filename);

    // Broadcast k to all processes
    MPI_Bcast(&k, 1, MPI_INT, 0, MPI_COMM_WORLD);

    vector<Point> full_data;
    vector<double> send_buf; // only used by rank 0

    int total_points = 0;
    if (rank == 0)
    {
        // Load CSV into full_data; global dims is set here
        full_data = load_csv(filename);
        total_points = (int)full_data.size();

        // Flatten the entire dataset to send: each row has dims elements
        send_buf.resize(total_points * dims);
        for (int i = 0; i < total_points; ++i)
        {
            for (int d = 0; d < dims; d++)
            {
                send_buf[i * dims + d] = full_data[i].coords[d];
            }
        }
    }

    // Broadcast total_points and dims
    MPI_Bcast(&total_points, 1, MPI_INT, 0, MPI_COMM_WORLD);
    // We also need to broadcast dims so that all processes know how many columns there are
    MPI_Bcast(&dims, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int base = total_points / size;
    int remain = total_points % size;
    int local_n = base + (rank < remain ? 1 : 0);

    // Prepare counts/displacements for Scatterv
    vector<int> counts(size), displs(size);
    if (rank == 0)
    {
        int prev = 0;
        for (int i = 0; i < size; ++i)
        {
            int n = base + (i < remain ? 1 : 0);
            counts[i] = n * dims;
            displs[i] = prev;
            prev += counts[i];
        }
    }

    // recv_buf for local portion
    vector<double> recv_buf(local_n * dims);

    // Scatter the flattened data
    MPI_Scatterv(send_buf.data(), counts.data(), displs.data(), MPI_DOUBLE,
                 recv_buf.data(), (int)recv_buf.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Convert to local Point array
    vector<Point> local_data(local_n);
    for (int i = 0; i < local_n; ++i)
    {
        vector<double> tmp(dims);
        for (int d = 0; d < dims; d++)
        {
            tmp[d] = recv_buf[i * dims + d];
        }
        local_data[i].coords = tmp;
        local_data[i].cluster = -1;
    }

    // Backup of original data for repeated trials
    vector<Point> original_local_data = local_data;

    // Number of trials to run
    const int trials = 100;
    double total_time = 0.0;
    vector<double> trial_times;

    for (int r = 0; r < trials; r++)
    {
        // Reset local_data each run
        local_data = original_local_data;

        // Initialize centroids
        vector<Centroid> centroids(k);
        srand(42 + r);
        if (rank == 0)
        {
            // srand(42 + r);
            for (int i = 0; i < k; ++i)
            {
                int idx = rand() % total_points;
                centroids[i].coords.resize(dims);
                for (int d = 0; d < dims; d++)
                {
                    centroids[i].coords[d] = send_buf[idx * dims + d];
                }
            }
        }
        // Broadcast the centroids to all processes
        for (int i = 0; i < k; i++)
        {
            if (rank == 0 && (int)centroids[i].coords.size() == 0)
            {
                centroids[i].coords.resize(dims, 0.0);
            }
        }
        // We broadcast each centroid's data
        for (int i = 0; i < k; i++)
        {
            if ((int)centroids[i].coords.size() == 0)
            {
                centroids[i].coords.resize(dims, 0.0);
            }
            MPI_Bcast(centroids[i].coords.data(), dims, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        }

        MPI_Barrier(MPI_COMM_WORLD);
        double start_time = MPI_Wtime();

        // Run K-means
        run_kmeans(local_data, centroids, k, 100, MPI_COMM_WORLD);

        // Gather results on the final trial
        vector<Point> all_points;
        if (r == 0)
        {
            all_points = gather_results_to_rank0(local_data, total_points, rank, size, MPI_COMM_WORLD);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        double end_time = MPI_Wtime();

        double run_time = (end_time - start_time) * 1000.0; // ms
        total_time += run_time;

        if (rank == 0)
        {
            trial_times.push_back(run_time);
            // cout << "Run " << r + 1 << " execution time: " << run_time << " ms" << endl;
            if ((r == 0) && (all_points.size() <= 500))
            {
                print_debug_summary(all_points, k);
            }
        }
    }

    if (rank == 0)
    {
        cout << "Average time over " << trials << " runs: "
             << (total_time / trials) << " ms" << endl;
        save_execution_times(trial_times, "mpi");
    }

    MPI_Finalize();
    return 0;
}
