/* OpenMp version of GMM */
#include <algorithm>
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
#include <omp.h>
#include <iomanip>
#include <sys/stat.h>
#include <sys/types.h>

using namespace std;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

double norm_factor = 1.0;

/* ---------------------------- Data -------------------------------- */
struct Point
{
    vector<double> coords;
    Point() = default;
    explicit Point(int d) : coords(d, 0.0) {}
    explicit Point(const vector<double> &c) : coords(c) {}
};
struct GaussianComponent
{
    vector<double> mean; // μ_k
    vector<double> var;  // σ_k² (diagonal)
    double weight = 1.0; // π_k
};

/* ------------------------ CSV & helper ---------------------------- */
int extract_k_from_filename(const string &f)
{
    smatch m;
    regex r("k(\\d+)");
    if (regex_search(f, m, r))
        return stoi(m[1]);
    cerr << "Cannot extract k\n";
    exit(1);
}
vector<Point> load_csv(const string &f)
{
    ifstream file(f);
    if (!file)
    {
        cerr << "open fail\n";
        exit(1);
    }
    vector<Point> pts;
    string line;
    int dim = -1;
    while (getline(file, line))
    {
        if (line.empty())
            continue;
        stringstream ss(line);
        string v;
        vector<double> c;
        while (getline(ss, v, ','))
            c.push_back(stod(v));
        if (dim < 0)
            dim = c.size();
        else if ((int)c.size() != dim)
        {
            cerr << "dim mismatch\n";
            exit(1);
        }
        pts.emplace_back(c);
    }
    return pts;
}

// Compute the probability density of a point for a diagonal Gaussian.
// Precompute 1/variance to avoid slow divisions.
inline double gauss_pdf_diag(const Point &p, const GaussianComponent &g)
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

/* -------------------- initialization ------------------------------ */
void init_comps(vector<GaussianComponent> &comps, const vector<Point> &pts)
{
    int k = comps.size(), n = pts.size(), dim = pts[0].coords.size();
    for (int c = 0; c < k; ++c)
    {
        int idx = rand() % n;
        comps[c].mean = pts[idx].coords;
        comps[c].var.assign(dim, 1.0);
        comps[c].weight = 1.0 / k;
    }
}

// Perform the E-step: compute and normalize responsibilities for each point.
double expectation_step(const vector<Point> &pts,
                        const vector<GaussianComponent> &comps,
                        vector<vector<double>> &resp)
{
    int n = pts.size(), k = comps.size();
    double ll = 0.0;
#pragma omp parallel for reduction(+ : ll) schedule(static)
    for (int i = 0; i < n; ++i)
    {
        double denom = 0.0;
        for (int c = 0; c < k; ++c)
        {
            resp[i][c] = comps[c].weight * gauss_pdf_diag(pts[i], comps[c]);
            denom += resp[i][c];
        }
        denom = max(denom, 1e-20);
        double inv_denom = 1.0 / denom;
        for (int c = 0; c < k; ++c)
            resp[i][c] *= inv_denom;
        ll += log(denom);
    }
    return ll;
}

/* --------------------- M‑step (no critical) ----------------------- */
void maximization_step(const vector<Point> &pts,
                       vector<GaussianComponent> &comps,
                       const vector<vector<double>> &resp)
{
    constexpr double EPS = 1e-6;
    int n = pts.size(), k = comps.size(), dim = pts[0].coords.size();

    /* 1‑D flattened accumulators */
    vector<double> Nk(k, 0.0);
    vector<double> sum_mean(k * dim, 0.0);
    vector<double> sum_var(k * dim, 0.0);

    double *Nk_a = Nk.data();
    double *mean_a = sum_mean.data();
    double *var_a = sum_var.data();

/* pass‑1 : accumulate Nk and mean numerators */
#pragma omp parallel for reduction(+ : Nk_a[ : k], mean_a[ : k * dim]) schedule(static)
    for (int i = 0; i < n; ++i)
    {
        for (int c = 0; c < k; ++c)
        {
            double g = resp[i][c];
            Nk_a[c] += g;
            const double *pc = pts[i].coords.data();
            for (int d = 0; d < dim; ++d)
                mean_a[c * dim + d] += g * pc[d];
        }
    }

    // Update means
    for (int c = 0; c < k; ++c)
        for (int d = 0; d < dim; ++d)
            comps[c].mean[d] = mean_a[c * dim + d] / (Nk_a[c] + EPS);

/* pass‑2 : accumulate variance numerators */
#pragma omp parallel for reduction(+ : var_a[ : k * dim]) schedule(static)
    for (int i = 0; i < n; ++i)
    {
        for (int c = 0; c < k; ++c)
        {
            double g = resp[i][c];
            const double *pc = pts[i].coords.data();
            for (int d = 0; d < dim; ++d)
            {
                double diff = pc[d] - comps[c].mean[d];
                var_a[c * dim + d] += g * diff * diff;
            }
        }
    }

    // Update variances and weights
    for (int c = 0; c < k; ++c)
    {
        comps[c].weight = Nk_a[c] / n;
        for (int d = 0; d < dim; ++d)
            comps[c].var[d] = var_a[c * dim + d] / (Nk_a[c] + EPS) + EPS;
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

/* --------------------- EM driver --------------------------------- */
void run_gmm(const vector<Point> &pts,
             vector<vector<double>> &resp,
             vector<GaussianComponent> &comps,
             int max_iter = 200, double tol = 1e-4)
{
    if (pts.empty())
        return;
    init_comps(comps, pts);

    double prev_ll = 0.0;
    for (int it = 0; it < max_iter; ++it)
    {
        double ll = expectation_step(pts, comps, resp);
        maximization_step(pts, comps, resp);
        if (it > 0 && fabs(ll - prev_ll) < tol * fabs(prev_ll))
            break;
        prev_ll = ll;
    }
}

/* --------------------- Debug ------------------------------------- */
void print_debug(const vector<Point> &pts, const vector<vector<double>> &resp, int k)
{
    vector<int> cnt(k, 0);
    for (size_t i = 0; i < pts.size(); ++i)
    {
        int best = int(max_element(resp[i].begin(), resp[i].end()) - resp[i].begin());
        cnt[best]++;
    }
    cout << "\n--- Cluster counts ---\n";
    for (int c = 0; c < k; ++c)
        cout << "Cluster " << c << ": " << cnt[c] << "\n";
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

/* --------------------- main -------------------------------------- */
int main(int argc, char *argv[])
{
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    if (argc < 2)
    {
        cerr << "Usage: ./gmm_omp <csv>\n";
        return 1;
    }
    string filename = argv[1];
    int k = extract_k_from_filename(filename);

    const vector<Point> master = load_csv(filename);
    int n = master.size();
    int dim = master[0].coords.size();
    norm_factor = pow(2.0 * M_PI, -0.5 * dim);

    const int trials = 100;
    double total_ms = 0.0;
    vector<double> trial_times;

    for (int t = 0; t < trials; ++t)
    {
        vector<Point> pts = master;
        vector<vector<double>> resp(n, vector<double>(k));
        vector<GaussianComponent> comps(k);

        srand(42 + t);

        auto t0 = chrono::high_resolution_clock::now();
        run_gmm(pts, resp, comps);
        auto t1 = chrono::high_resolution_clock::now();
        double elapsed = chrono::duration<double, milli>(t1 - t0).count();
        total_ms += elapsed;
        trial_times.push_back(elapsed);

        if (t == 0 && n <= 500)
            print_debug(pts, resp, k);
    }
    cout << "Average time over " << trials << " runs: " << total_ms / trials << " ms\n";
    string test_name = get_clean_test_name(filename);
    save_execution_times(trial_times, test_name, "gmm_openmp");
    return 0;
}
