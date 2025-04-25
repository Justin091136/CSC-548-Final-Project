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

using namespace std;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

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

/* === Gaussian PDF for diagonal Σ === */
inline double gaussian_pdf_diag(const Point &p,
                                const GaussianComponent &comp)
{
    const double EPS = 1e-9; // avoid division by zero
    double exponent = 0.0, denom = 1.0;
    for (size_t d = 0; d < p.coords.size(); ++d)
    {
        double var = comp.var[d] + EPS;
        double diff = p.coords[d] - comp.mean[d];
        exponent += (diff * diff) / var;
        denom *= var;
    }
    if (denom < 1e-300)
        denom = 1e-300;
    double norm_const = pow(2.0 * M_PI, -0.5 * p.coords.size()) *
                        pow(denom, -0.5);
    return norm_const * exp(-0.5 * exponent);
}

/* === Initialization: K‑Means++ style (simplified) === */
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

/* === Expectation Step === */
double expectation_step(const vector<Point> &points,
                        const vector<GaussianComponent> &comps,
                        vector<vector<double>> &resp)
{
    int n = (int)points.size();
    int k = (int)comps.size();
    double log_likelihood = 0.0;

    // PARALLEL CANDIDATE: loop over points
    for (int i = 0; i < n; ++i)
    {
        double denom = 0.0;
        for (int c = 0; c < k; ++c)
        {
            resp[i][c] = comps[c].weight *
                         gaussian_pdf_diag(points[i], comps[c]);
            denom += resp[i][c];
        }
        // Normalize γ_ik
        if (denom < 1e-20)
            denom = 1e-20;
        for (int c = 0; c < k; ++c)
            resp[i][c] /= denom;
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
             int max_iters = 100, double tol = 1e-4)
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

/* ================================================================
 *  main – time only run_gmm(), then print hard‑cluster counts
 * ================================================================ */
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

    const int trials = 50;
    double total_ms = 0.0;

    for (int t = 0; t < trials; ++t)
    {
        /* make a fresh copy so every trial starts identical */
        vector<Point> points = master_points;

        /* allocate working buffers */
        vector<vector<double>> resp(n, vector<double>(k));
        vector<GaussianComponent> comps(k);

        srand(42 + t); // new seed each trial

        auto t0 = chrono::high_resolution_clock::now();
        run_gmm(points, resp, comps); //  ← timed region
        auto t1 = chrono::high_resolution_clock::now();

        total_ms += chrono::duration<double, milli>(t1 - t0).count();

        if (t == 0 && n <= 500)
            print_debug_summary(points, resp, k);
    }
    cout << "Average time over " << trials << " runs: "
         << (total_ms / trials) << " ms\n";
}
