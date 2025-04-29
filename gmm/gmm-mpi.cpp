/* MPI version of GMM */
#include <mpi.h>
#include <vector>
#include <string>
#include <regex>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <unistd.h>
#include <iomanip>
#include <sys/stat.h>
#include <sys/types.h>

using namespace std;
using std::string;
using std::vector;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

double norm_factor = 1.0;
constexpr double EPS = 1e-9;

struct Point
{
    vector<double> coords;
    explicit Point(int dim = 0) : coords(dim, 0.0) {}
    explicit Point(const vector<double> &v) : coords(v) {}
};
struct Gaussian
{
    vector<double> mean, var;
    double weight{};
};

int extract_k(const string &f)
{
    std::smatch m;
    std::regex r("k(\\d+)");
    if (std::regex_search(f, m, r))
        return std::stoi(m[1]);
    std::cerr << "Cannot extract k from filename\n";
    MPI_Abort(MPI_COMM_WORLD, 1);
    return 0;
}

vector<Point> load_csv(const string &fn)
{
    std::ifstream file(fn);
    if (!file)
    {
        std::cerr << "open " << fn << " failed\n";
        exit(1);
    }
    vector<Point> pts;
    string line;
    int dim = -1;
    while (getline(file, line))
    {
        if (line.empty())
            continue;
        std::stringstream ss(line);
        string tok;
        vector<double> c;
        while (getline(ss, tok, ','))
        {
            c.push_back(std::stod(tok));
        }
        if (dim < 0)
        {
            dim = (int)c.size();
        }
        else if ((int)c.size() != dim)
        {
            std::cerr << "dim mismatch\n";
            exit(1);
        }
        pts.emplace_back(c);
    }
    return pts;
}

// Compute the probability density of a point for a diagonal Gaussian.
// Precompute 1/variance to avoid slow divisions.
inline double gaussian_pdf_diag(const Point &p, const Gaussian &g)
{
    double e = 0.0, inv_denom = 1.0;
    for (size_t d = 0; d < p.coords.size(); ++d)
    {
        double var = g.var[d] + EPS;
        double inv_var = 1.0 / var;
        double diff = p.coords[d] - g.mean[d];
        e += diff * diff * inv_var;
        inv_denom *= inv_var;
    }
    if (inv_denom < 1e-300)
        inv_denom = 1e-300;
    double norm = norm_factor * std::sqrt(inv_denom);
    return norm * std::exp(-0.5 * e);
}

// Perform the E-step: compute and normalize responsibilities for each point.
double expectation_step(const vector<Point> &pts,
                        const vector<Gaussian> &comps,
                        vector<vector<double>> &resp)
{
    int n = pts.size(), k = comps.size();
    double ll = 0.0;
    for (int i = 0; i < n; ++i)
    {
        double denom = 0.0;
        for (int c = 0; c < k; ++c)
        {
            resp[i][c] = comps[c].weight * gaussian_pdf_diag(pts[i], comps[c]);
            denom += resp[i][c];
        }
        if (denom < 1e-20)
            denom = 1e-20;
        double inv_denom = 1.0 / denom;
        for (int c = 0; c < k; ++c)
        {
            resp[i][c] *= inv_denom;
        }
        ll += std::log(denom);
    }
    return ll;
}

void maximization_step(const vector<Point> &pts,
                       vector<Gaussian> &comps,
                       const vector<vector<double>> &resp,
                       MPI_Comm comm)
{
    int n = pts.size(), k = comps.size(), dim = pts[0].coords.size();
    vector<double> Nk(k, 0.), sum_mean(k * dim, 0.), sum_var(k * dim, 0.);

    // accumulate Nk and Σ γ x
    for (int i = 0; i < n; ++i)
    {
        for (int c = 0; c < k; ++c)
        {
            double g = resp[i][c];
            Nk[c] += g;
            for (int d = 0; d < dim; ++d)
            {
                sum_mean[c * dim + d] += g * pts[i].coords[d];
            }
        }
    }

    // global reductions for Nk and Σ γ x
    MPI_Allreduce(MPI_IN_PLACE, Nk.data(), k, MPI_DOUBLE, MPI_SUM, comm);
    MPI_Allreduce(MPI_IN_PLACE, sum_mean.data(), k * dim, MPI_DOUBLE, MPI_SUM, comm);

    // update means
    for (int c = 0; c < k; ++c)
    {
        for (int d = 0; d < dim; ++d)
        {
            comps[c].mean[d] = sum_mean[c * dim + d] / (Nk[c] + EPS);
        }
    }

    // accumulate Σ γ (x - μ)^2
    std::fill(sum_var.begin(), sum_var.end(), 0.);
    for (int i = 0; i < n; ++i)
    {
        for (int c = 0; c < k; ++c)
        {
            double g = resp[i][c];
            for (int d = 0; d < dim; ++d)
            {
                double diff = pts[i].coords[d] - comps[c].mean[d];
                sum_var[c * dim + d] += g * diff * diff;
            }
        }
    }
    // global reduction for Σ γ (x - μ)^2
    MPI_Allreduce(MPI_IN_PLACE, sum_var.data(), k * dim, MPI_DOUBLE, MPI_SUM, comm);

    // get total sample count
    int local_n = n, global_n = 0;
    MPI_Allreduce(&local_n, &global_n, 1, MPI_INT, MPI_SUM, comm);

    // update weights and variances
    for (int c = 0; c < k; ++c)
    {
        comps[c].weight = Nk[c] / global_n;
        for (int d = 0; d < dim; ++d)
        {
            comps[c].var[d] = sum_var[c * dim + d] / (Nk[c] + EPS) + EPS;
        }
    }

    // Make sure all component weights add up to 1.0,
    // so the model remains a valid probability distribution.
    double sum_w = 0.0;
    for (int c = 0; c < k; ++c)
        sum_w += comps[c].weight;
    for (int c = 0; c < k; ++c)
        comps[c].weight /= sum_w;
}

void run_gmm_mpi(const vector<Point> &pts,
                 vector<Gaussian> &comps,
                 vector<vector<double>> &resp,
                 MPI_Comm comm,
                 int max_iter = 200, double tol = 1e-4)
{
    int rank;
    MPI_Comm_rank(comm, &rank);
    double prev_ll = -std::numeric_limits<double>::infinity();

    for (int it = 0; it < max_iter; ++it)
    {
        double ll_local = expectation_step(pts, comps, resp), ll_global;
        MPI_Allreduce(&ll_local, &ll_global, 1, MPI_DOUBLE, MPI_SUM, comm);

        maximization_step(pts, comps, resp, comm);

        // check convergence
        int stop = (std::fabs(ll_global - prev_ll) < tol * std::fabs(prev_ll));
        MPI_Bcast(&stop, 1, MPI_INT, 0, comm);
        if (stop)
            break;
        prev_ll = ll_global;
    }
}

void print_debug_global(const vector<Point> &pts,
                        const vector<vector<double>> &resp,
                        int k, int rank, MPI_Comm comm)
{
    vector<int> cnt_local(k, 0), cnt_global(k, 0);
    for (size_t i = 0; i < pts.size(); ++i)
    {
        int best = int(std::max_element(resp[i].begin(), resp[i].end()) - resp[i].begin());
        cnt_local[best]++;
    }
    MPI_Reduce(cnt_local.data(), cnt_global.data(), k, MPI_INT, MPI_SUM, 0, comm);
    if (rank == 0)
    {
        std::cout << "\n--- Cluster counts (global) ---\n";
        for (int c = 0; c < k; ++c)
            std::cout << "Cluster " << c << ": " << cnt_global[c] << "\n";
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

void save_execution_times(const vector<double> &times, const string &test_name, const string &version_name)
{
    string results_dir = "results";
    string runtime_csv_dir = results_dir + "/runtime_csv";
    string test_dir = runtime_csv_dir + "/" + test_name;

    create_dir_if_not_exists(results_dir);
    create_dir_if_not_exists(runtime_csv_dir);
    create_dir_if_not_exists(test_dir);

    string output_file = test_dir + "/execution_times_" + version_name + ".csv";
    cout << "Saving result to " << output_file << endl;
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
int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 2)
    {
        if (rank == 0)
            std::cerr << "Usage: mpirun -n P ./gmm_mpi <csv>\n";
        MPI_Finalize();
        return 0;
    }

    char hostname[256];
    gethostname(hostname, sizeof(hostname));

    /*
    std::cout << "Rank " << rank << "/" << size
              << " running on " << hostname
              << ", PID " << getpid()
              << std::endl;
    */

    string filename = argv[1];
    int k = extract_k(filename);

    // rank 0 loads the CSV
    vector<double> flat_all;
    int global_n = 0, dim = 0;
    if (rank == 0)
    {
        auto full = load_csv(filename);
        global_n = full.size();
        dim = full[0].coords.size();
        flat_all.resize(global_n * dim);
        for (int i = 0; i < global_n; ++i)
        {
            std::copy(full[i].coords.begin(), full[i].coords.end(),
                      flat_all.begin() + i * dim);
        }
    }
    MPI_Bcast(&dim, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&global_n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    norm_factor = std::pow(2.0 * M_PI, -0.5 * dim);

    // distribute data with Scatterv
    vector<int> cnt(size), disp(size);
    int base = global_n / size, rem = global_n % size;
    for (int r = 0; r < size; ++r)
    {
        cnt[r] = (r < rem ? base + 1 : base) * dim;
        disp[r] = (r == 0 ? 0 : disp[r - 1] + cnt[r - 1]);
    }
    int local_n = cnt[rank] / dim;
    vector<double> local_flat(cnt[rank]);
    MPI_Scatterv(flat_all.empty() ? nullptr : flat_all.data(),
                 cnt.data(), disp.data(), MPI_DOUBLE,
                 local_flat.data(), cnt[rank], MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    vector<Point> local_pts(local_n, Point(dim));
    for (int i = 0; i < local_n; ++i)
    {
        for (int d = 0; d < dim; ++d)
        {
            local_pts[i].coords[d] = local_flat[i * dim + d];
        }
    }

    const int trials = 100;
    double acc_ms = 0.0;
    vector<double> trial_times;

    for (int t = 0; t < trials; ++t)
    {
        // rank 0 selects random μ/var/weight and broadcasts
        srand(42 + t);
        vector<double> mu(k * dim), var(k * dim, 1.0), weight(k, 1.0 / k);
        if (rank == 0)
        {
            for (int c = 0; c < k; ++c)
            {
                int idx = rand() % global_n;
                for (int d = 0; d < dim; ++d)
                    mu[c * dim + d] = flat_all[idx * dim + d];
            }
        }
        MPI_Bcast(mu.data(), k * dim, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(var.data(), k * dim, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(weight.data(), k, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // build Gaussian components
        vector<Gaussian> comps(k);
        for (int c = 0; c < k; ++c)
        {
            comps[c].mean.assign(mu.begin() + c * dim, mu.begin() + (c + 1) * dim);
            comps[c].var.assign(var.begin() + c * dim, var.begin() + (c + 1) * dim);
            comps[c].weight = weight[c];
        }
        vector<vector<double>> resp(local_n, vector<double>(k));

        MPI_Barrier(MPI_COMM_WORLD);
        auto t0 = std::chrono::high_resolution_clock::now();

        run_gmm_mpi(local_pts, comps, resp, MPI_COMM_WORLD);

        MPI_Barrier(MPI_COMM_WORLD);
        auto t1 = std::chrono::high_resolution_clock::now();

        double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        if (rank == 0)
        {
            acc_ms += elapsed_ms;
            trial_times.push_back(elapsed_ms);
        }

        if (t == 0 && global_n <= 500)
            print_debug_global(local_pts, resp, k, rank, MPI_COMM_WORLD);
    }

    if (rank == 0)
    {
        std::cout << "Average time over " << trials << " runs: "
                  << acc_ms / trials << " ms\n";

        if (save_result)
        {
            string test_name = get_clean_test_name(filename);
            save_execution_times(trial_times, test_name, "gmm_mpi");
        }
    }

    MPI_Finalize();
    return 0;
}
