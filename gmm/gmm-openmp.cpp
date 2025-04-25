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

using namespace std;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

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

/* ---------------------- Gaussian PDF ------------------------------ */
inline double gauss_pdf_diag(const Point &p, const GaussianComponent &g)
{
    constexpr double EPS = 1e-9;
    double expn = 0.0, den = 1.0;
    for (size_t d = 0; d < p.coords.size(); ++d)
    {
        double var = g.var[d] + EPS, diff = p.coords[d] - g.mean[d];
        expn += diff * diff / var;
        den *= var;
    }
    double norm = pow(2.0 * M_PI, -0.5 * p.coords.size()) * pow(den, -0.5);
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

/* --------------------- E‑step (parallel) -------------------------- */
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
        for (int c = 0; c < k; ++c)
            resp[i][c] /= denom;
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

/* pass‑1 : Nk & mean numerator */
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
    for (int c = 0; c < k; ++c)
        for (int d = 0; d < dim; ++d)
            comps[c].mean[d] = mean_a[c * dim + d] / (Nk_a[c] + EPS);

/* pass‑2 : variance numerator */
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
    for (int c = 0; c < k; ++c)
    {
        comps[c].weight = Nk_a[c] / n;
        for (int d = 0; d < dim; ++d)
            comps[c].var[d] = var_a[c * dim + d] / (Nk_a[c] + EPS) + EPS;
    }
}

/* --------------------- EM driver --------------------------------- */
void run_gmm(const vector<Point> &pts,
             vector<vector<double>> &resp,
             vector<GaussianComponent> &comps,
             int max_iter = 100, double tol = 1e-4)
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
    string file = argv[1];
    int k = extract_k_from_filename(file);

    const vector<Point> master = load_csv(file);
    int n = master.size();

    const int trials = 50;
    double total_ms = 0.0;
    for (int t = 0; t < trials; ++t)
    {
        vector<Point> pts = master;
        vector<vector<double>> resp(n, vector<double>(k));
        vector<GaussianComponent> comps(k);

        srand(42 + t);

        auto t0 = chrono::high_resolution_clock::now();
        run_gmm(pts, resp, comps);
        auto t1 = chrono::high_resolution_clock::now();
        total_ms += chrono::duration<double, milli>(t1 - t0).count();

        if (t == 0 && n <= 500)
            print_debug(pts, resp, k);
    }
    cout << "Average time over " << trials << " runs: " << total_ms / trials << " ms\n";
    return 0;
}
