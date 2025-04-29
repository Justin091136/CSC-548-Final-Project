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
#include <iomanip>
#include <chrono>
#include <sys/stat.h>
#include <sys/types.h>

using namespace std;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

double norm_factor = 1.0;

/* ---------- Data Structures ---------- */
struct Point
{
    vector<double> coords;
    Point() = default;
    explicit Point(int dim) : coords(dim, 0.0) {}
    explicit Point(const vector<double> &c) : coords(c) {}
};

struct GaussianComponent
{
    vector<double> mean; // μ_k
    vector<double> var;  // σ_k^2  (diagonal covariance)
    double weight = 1.0; // π_k
};

/* ---------- Utility (unchanged) ---------- */
int extract_k_from_filename(const string &filename)
{
    smatch match;
    regex pattern("k(\\d+)");
    if (regex_search(filename, match, pattern))
        return stoi(match[1]);
    cerr << "Error: Cannot extract k from filename: " << filename << endl;
    exit(1);
}

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
            continue;
        stringstream ss(line);
        string val;
        vector<double> coords;
        while (getline(ss, val, ','))
            coords.push_back(stod(val));
        if (dimension < 0)
            dimension = (int)coords.size();
        else if ((int)coords.size() != dimension)
        {
            cerr << "Error: Inconsistent column count in CSV.\n";
            exit(1);
        }
        points.emplace_back(coords);
    }
    return points;
}

// Compute the probability density of a point for a diagonal Gaussian.
// Precompute 1/variance to avoid slow divisions.
inline double gaussian_pdf_diag(const Point &p, const GaussianComponent &g)
{
    // After maximization_step(), g.var[d] already stores inv_var (1/σ²)
    double expn = 0.0, inv_denom = 1.0;

    for (size_t d = 0; d < p.coords.size(); ++d)
    {
        double inv_var = g.var[d];
        double diff = p.coords[d] - g.mean[d];
        expn += diff * diff * inv_var;
        inv_denom *= inv_var;
    }

    if (inv_denom < 1e-300)
        inv_denom = 1e-300;

    double norm = norm_factor * sqrt(inv_denom);
    return norm * exp(-0.5 * expn);
}

void initialize_components(vector<GaussianComponent> &comps,
                           const vector<Point> &points)
{
    int k = (int)comps.size();
    int n = (int)points.size();
    int dim = (int)points[0].coords.size();
    srand(42);
    for (int i = 0; i < k; ++i)
    {
        int idx = rand() % n;
        comps[i].mean = points[idx].coords;
        comps[i].var.assign(dim, 1.0); // start with unit variance
        comps[i].weight = 1.0 / k;
    }
}

// Perform the E-step: compute and normalize responsibilities for each point.
double expectation_step(const vector<Point> &points, const vector<GaussianComponent> &comps, vector<vector<double>> &resp)
{
    int n = static_cast<int>(points.size());
    int k = static_cast<int>(comps.size());
    double log_likelihood = 0.0;

    for (int i = 0; i < n; ++i)
    {
        double denom = 0.0;
        for (int c = 0; c < k; ++c)
        {
            resp[i][c] = comps[c].weight * gaussian_pdf_diag(points[i], comps[c]);
            denom += resp[i][c];
        }
        if (denom < 1e-20)
            denom = 1e-20;
        double inv_denom = 1.0 / denom;
        for (int c = 0; c < k; ++c)
            resp[i][c] *= inv_denom;
        log_likelihood += log(denom);
    }
    return log_likelihood;
}

/* === Maximization Step === */
void maximization_step(const vector<Point> &points,
                       vector<GaussianComponent> &comps,
                       const vector<vector<double>> &resp)
{
    const double EPS = 1e-6;
    int n = (int)points.size();
    int k = (int)comps.size();
    int dim = (int)points[0].coords.size();

    // Temporary accumulators
    vector<double> Nk(k, 0.0);
    vector<vector<double>> sum_mean(k, vector<double>(dim, 0.0));
    vector<vector<double>> sum_var(k, vector<double>(dim, 0.0));

    // PARALLEL CANDIDATE: loop over points
    for (int i = 0; i < n; ++i)
    {
        for (int c = 0; c < k; ++c)
        {
            double gamma = resp[i][c];
            Nk[c] += gamma;
            for (int d = 0; d < dim; ++d)
            {
                double x = points[i].coords[d];
                sum_mean[c][d] += gamma * x;
            }
        }
    }

    // Update means
    for (int c = 0; c < k; ++c)
        for (int d = 0; d < dim; ++d)
            comps[c].mean[d] = sum_mean[c][d] / (Nk[c] + EPS);

    // Re‑accumulate for variance
    // PARALLEL CANDIDATE: loop over points again
    for (int i = 0; i < n; ++i)
    {
        for (int c = 0; c < k; ++c)
        {
            double gamma = resp[i][c];
            for (int d = 0; d < dim; ++d)
            {
                double diff = points[i].coords[d] - comps[c].mean[d];
                sum_var[c][d] += gamma * diff * diff;
            }
        }
    }

    // Update variances and weights
    for (int c = 0; c < k; ++c)
    {
        comps[c].weight = Nk[c] / n;
        for (int d = 0; d < dim; ++d)
        {
            comps[c].var[d] = sum_var[c][d] / (Nk[c] + EPS) + EPS;
        }
    }

    // Make sure all component weights add up to 1.0,
    // so the model remains a valid probability distribution.
    double sum_w = 0.0;
    for (int c = 0; c < k; ++c)
        sum_w += comps[c].weight;
    for (int c = 0; c < k; ++c)
        comps[c].weight /= sum_w;

    // Precompute inv_var = 1/σ² after updating variance
    for (int c = 0; c < k; ++c)
        for (int d = 0; d < dim; ++d)
            comps[c].var[d] = 1.0 / comps[c].var[d];
}

void print_debug_summary(const vector<Point> &points, const vector<vector<double>> &resp, int k)
{
    vector<int> count(k, 0);
    for (size_t i = 0; i < points.size(); ++i)
    {
        int best = 0;
        double best_gamma = resp[i][0];
        for (int c = 1; c < k; ++c)
        {
            if (resp[i][c] > best_gamma)
            {
                best_gamma = resp[i][c];
                best = c;
            }
        }
        count[best]++;
    }

    cout << "\n--- Final Cluster Counts (arg‑max γik) ---\n";
    for (int c = 0; c < k; ++c)
    {
        cout << "Cluster " << c << ": " << count[c] << " points\n";
    }
}

/* === EM Driver === */
/* ================================================================
 *  GMM driver — uses pre‑allocated responsibility matrix (resp)
 *  points : input samples            (size n)
 *  resp   : n × k responsibility     (modified in‑place)
 *  comps  : Gaussian components      (modified in‑place)
 * ================================================================ */
void run_gmm(const vector<Point> &points,
             vector<vector<double>> &resp,
             vector<GaussianComponent> &comps,
             int max_iters = 200, double tol = 1e-4)
{
    if (points.empty())
        return;

    initialize_components(comps, points);

    double prev_ll = 0.0; // 初值隨便，iter==0 不檢查
    for (int iter = 0; iter < max_iters; ++iter)
    {
        double ll = expectation_step(points, comps, resp);
        maximization_step(points, comps, resp);

        if (iter > 0 && fabs(ll - prev_ll) < tol * fabs(prev_ll))
            break;
        prev_ll = ll;
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

/* ================================================================
 *  main – time only run_gmm(), then print hard‑cluster counts
 * ================================================================ */
bool save_result = false;
int main(int argc, char *argv[])
{
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    if (argc < 2)
    {
        cerr << "Usage: ./gmm <csv_filename>\n";
        return 1;
    }
    string filename = argv[1];
    int k = extract_k_from_filename(filename);

    /* ---- load data only once ---- */
    const vector<Point> master_points = load_csv(filename);
    int n = static_cast<int>(master_points.size());
    int dim = master_points[0].coords.size();
    norm_factor = pow(2.0 * M_PI, -0.5 * dim);

    const int trials = 100;
    double total_ms = 0.0;
    vector<double> trial_times;

    for (int t = 0; t < trials; ++t)
    {
        /* make a fresh copy so every trial starts identical */
        vector<Point> points = master_points;

        /* allocate working buffers */
        vector<vector<double>> resp(n, vector<double>(k));
        vector<GaussianComponent> comps(k);

        srand(42 + t);

        auto t0 = chrono::high_resolution_clock::now();
        run_gmm(points, resp, comps); //  ← timed region
        auto t1 = chrono::high_resolution_clock::now();

        double elapsed = chrono::duration<double, milli>(t1 - t0).count();
        total_ms += elapsed;
        trial_times.push_back(elapsed);

        if (t == 0 && n <= 500)
            print_debug_summary(points, resp, k);
    }
    cout << "Average time over " << trials << " runs: "
         << (total_ms / trials) << " ms\n";

    if (save_result)
    {
        string test_name = get_clean_test_name(filename);
        save_execution_times(trial_times, test_name, "gmm_seq");
    }

    return 0;
}
