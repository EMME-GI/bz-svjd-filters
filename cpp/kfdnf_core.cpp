#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>

using namespace Eigen;

// Helper for the FO filter
bool compute_P0_from_kronecker_FO(
    const Eigen::Ref<const MatrixXd>& Phi1,
    const Eigen::Matrix3d& Sigma_0,
    MatrixXd& P0_out
) {
    // Zero heap malloc (fixed dimensions)
    Eigen::Matrix<double, 9, 9> I9_minus_kron = Eigen::Matrix<double, 9, 9>::Identity();

    for (int j = 0; j < 3; ++j) {
        for (int i = 0; i < 3; ++i) {
            I9_minus_kron.block<3, 3>(3 * i, 3 * j) -= Phi1(i, j) * Phi1.block<3, 3>(0, 0);
        }
    }

    // Zero-copy vec(Sigma_0): reads column-major buffer of Sigma_0
    Eigen::Map<const Eigen::Matrix<double, 9, 1>> vec_Sigma(Sigma_0.data());

    // Solves for vec(V_{0|0})
    Eigen::Matrix<double, 9, 1> vec_P0 = I9_minus_kron.lu().solve(vec_Sigma);

    // Building P0_out
    P0_out.resize(3, 3);
    Eigen::Map<Eigen::Matrix<double, 9, 1>>(P0_out.data()) = vec_P0;

    // Check for numerical stability and positive diagonal (stationarity)
    if (!P0_out.allFinite() || P0_out(0,0) <= 0.0 || P0_out(1,1) <= 0.0 || P0_out(2,2) <= 0.0) {
        return false;
    }
    return true;
}

// The following class is just an example to fix in mind how the (conditional) filter should work,
// it will not be used in other files
class ConditionalKalmanFilter {
private:
    VectorXd v;
    MatrixXd P;

public:
    ConditionalKalmanFilter() {} 

    void init_state(Eigen::Ref<const VectorXd> v0, Eigen::Ref<const MatrixXd> P0) {
        v = v0;
        P = P0;
    }

    // Simplified Kalman Step (M_mat = 0 because option errors are indipendent)
    double step(Eigen::Ref<const VectorXd> Phi0, Eigen::Ref<const MatrixXd> Phi1, Eigen::Ref<const MatrixXd> Sigma,
                Eigen::Ref<const VectorXd> z, Eigen::Ref<const VectorXd> Gamma0, Eigen::Ref<const MatrixXd> Gamma1, 
                Eigen::Ref<const MatrixXd> Omega) 
    {
        // Predict
        VectorXd v_pred = Phi0 + Phi1 * v;
        v_pred = v_pred.cwiseMax(1e-8); 

        VectorXd z_pred = Gamma0 + Gamma1 * v_pred;

        MatrixXd P_pred = Phi1 * P * Phi1.transpose() + Sigma;
        
        // Total uncertainty
        MatrixXd S = Gamma1 * P_pred * Gamma1.transpose() + Omega;
        MatrixXd Cov_XZ = P_pred * Gamma1.transpose(); 

        VectorXd y_res = z - z_pred;
        MatrixXd K = Cov_XZ * S.inverse();

        // Update
        v = v_pred + K * y_res;
        v = v.cwiseMax(1e-8); 
        
        P = P_pred - K * S * K.transpose();

        double det_S = S.determinant();
        if (det_S <= 0.0) return 0.0; 

        int dim_z = z.rows();
        double mahalanobis = (y_res.transpose() * S.inverse() * y_res).value();
        
        return std::exp(-0.5 * mahalanobis) / std::sqrt(std::pow(2.0 * M_PI, dim_z) * det_S);
    }

    VectorXd get_v() const { return v; }
    MatrixXd get_P() const { return P; }
};

// This is the actual KF-DNF
class DiscreteNonlinearFilter {
private:
    VectorXd v_est;
    MatrixXd P_est;
    double log_likelihood_total;
    MatrixXd v_history;

public:
    DiscreteNonlinearFilter() { log_likelihood_total = 0.0; }
    
    MatrixXd get_v_history() const { return v_history; }

    double filter_series(
        Eigen::Ref<const MatrixXd> Z_series,
        Eigen::Ref<const MatrixXd> Phi0_mat,
        Eigen::Ref<const MatrixXd> Phi1,
        Eigen::Ref<const VectorXd> Gamma0,
        Eigen::Ref<const MatrixXd> Gamma1,
        Eigen::Ref<const VectorXd> node_probs_static,
        Eigen::Ref<const MatrixXd> Omega_mat,
        double sigma1, double sigma2, double dt,
        double c0_m, double c1_m, double c2_m, double c3_m 
    ) {
        int T = Z_series.rows();
        int obs_dim = Z_series.cols();
        int state_dim = 3;
        int n_nodes = Phi0_mat.cols();
        log_likelihood_total = 0.0;
        v_history.resize(T, state_dim);

        // --- DNF INIT (O(1) due to diagonal Phi1 and Sigma_t) ---
        v_est.resize(state_dim);
        for (int i = 0; i < state_dim; ++i) {
            double k_dt = 1.0 - Phi1(i, i); // equivalent to (-K1)_{i,i}
            if (k_dt <= 0.0) return -1e9;   // stationarity guaranteed
            v_est(i) = std::max(Phi0_mat(i, 0) / k_dt, 1e-8);
        }

        MatrixXd Sigma_t = MatrixXd::Zero(state_dim, state_dim);
        Sigma_t(0,0) = sigma1 * sigma1 * v_est(0) * dt + 1e-8;
        Sigma_t(1,1) = sigma2 * sigma2 * v_est(1) * dt + 1e-8;
        Sigma_t(2,2) = 1e-8;

        P_est = MatrixXd::Zero(state_dim, state_dim);
        for (int i = 0; i < state_dim; ++i) {
            double denom = 1.0 - Phi1(i, i) * Phi1(i, i); // (I_9 - Phi1 (x) Phi1) on the diagonal
            if (denom <= 0.0) return -1e9;
            P_est(i, i) = Sigma_t(i, i) / denom;
        }

        // Pre-cycle allocations
        std::vector<VectorXd> v_updated_nodes(n_nodes, VectorXd::Zero(state_dim));
        VectorXd prob_nodes_new = VectorXd::Zero(n_nodes);
        VectorXd log_lik_nodes = VectorXd::Zero(n_nodes);
        VectorXd dyn_node_probs = VectorXd::Zero(n_nodes);
        
        // Identity
        MatrixXd I_obs = MatrixXd::Identity(obs_dim, obs_dim);

        for (int t = 0; t < T; ++t) {
            VectorXd z_t = Z_series.row(t).transpose();
            
            // State at time t-1 (already collapsed)
            double v1_prev = std::max(v_est(0), 1e-8);
            double v2_prev = std::max(v_est(1), 1e-8);
            double v3_prev = std::max(v_est(2), 1e-8);

            // 1. ARRIVAL PROBABILITY (SVJD dynamics following Eq. 9)
            double c_minus_t = c0_m + (c1_m * v1_prev) + (c2_m * v2_prev) + (c3_m * v3_prev);
            double prob_arrival = 1.0 - std::exp(-c_minus_t * dt); // Exact formula, sec 2.3
            double prob_no_arrival = 1.0 - prob_arrival;

            // node_probs_static has dimension K+1 (0 index ignored, 1..K spatial weights)
            // n_nodes will be (K+1)^2
            int K_grid = node_probs_static.size() - 1; 

            // SECURITY CHECK: we check that gridn dimensions are correct
            if (n_nodes != (K_grid + 1) * (K_grid + 1)) {
                return -1e9; // Refuses parameters and avoids crashes
            }

            for (int i = 0; i <= K_grid; ++i) {
                // Marginal probability for the x jump
                double p_x = (i == 0) ? prob_no_arrival : (prob_arrival * node_probs_static(i));
                
                for (int j = 0; j <= K_grid; ++j) {
                    // Marginal probability for the y jump
                    double p_y = (j == 0) ? prob_no_arrival : (prob_arrival * node_probs_static(j));
                    
                    // Linear index for the flattened matrix
                    int idx = i * (K_grid + 1) + j;
                    
                    // Joint probability (Indipendence: P(x,y) = P(x)*P(y))
                    dyn_node_probs(idx) = p_x * p_y;
                }
            }

            // Baseline diffusive variance (the same for every node at this step)
            Sigma_t(0,0) = sigma1 * sigma1 * v1_prev * dt + 1e-8;
            Sigma_t(1,1) = sigma2 * sigma2 * v2_prev * dt + 1e-8;

            MatrixXd P_pred_base = Phi1 * P_est * Phi1.transpose() + Sigma_t;
            MatrixXd S = Gamma1 * P_pred_base * Gamma1.transpose() + Omega_mat; // M in the paper, S is more common in literature
            
            LLT<MatrixXd> llt_S(S);
            if (llt_S.info() != Eigen::Success) return -1e9;
            
            // Numericaly stable inverse computation using LLT
            MatrixXd S_inv = llt_S.solve(I_obs);
            
            double det_S = S.determinant(); 
            double log_det_S = std::log(std::max(det_S, 1e-300));

            MatrixXd P_GammaT = P_pred_base * Gamma1.transpose();
            MatrixXd K_gain = P_GammaT * S_inv;

            MatrixXd P_update_base = P_pred_base - K_gain * S * K_gain.transpose();

            // 3. CONDITIONAL KALMAN FILTER (On the K nodes)
            for (int j = 0; j < n_nodes; ++j) {
                // PREDICT (The drift Phi0_mat.col(j) contains the exact impact of the k-th jump)
                VectorXd v_pred = Phi0_mat.col(j) + Phi1 * v_est;
                v_pred = v_pred.cwiseMax(1e-8);

                // UPDATE
                VectorXd z_pred = Gamma0 + Gamma1 * v_pred;
                VectorXd y_res = z_t - z_pred;

                v_updated_nodes[j] = v_pred + K_gain * y_res;
                v_updated_nodes[j] = v_updated_nodes[j].cwiseMax(1e-8);

                // LIKELIHOOD OF THE NODE
                double mahalanobis = (y_res.transpose() * S_inv * y_res).value();
                double ll_j = -0.5 * obs_dim * std::log(2.0 * M_PI) - 0.5 * log_det_S - 0.5 * mahalanobis;
                
                // Summing the a-priori log-probability of the jump (avoiding zero)
                log_lik_nodes(j) = std::log(std::max(dyn_node_probs(j), 1e-300)) + ll_j;
            }

            double max_ll = log_lik_nodes.maxCoeff();
            double sum_exp = 0.0;
            
            for(int j = 0; j < n_nodes; ++j) {
                double exp_val = std::exp(log_lik_nodes(j) - max_ll);
                prob_nodes_new(j) = exp_val;
                sum_exp += exp_val;
            }
            
            // Total Likelihood
            double log_lik_t_total = max_ll + std::log(sum_exp);
            log_likelihood_total += log_lik_t_total;
            
            // Probability Normalization
            prob_nodes_new /= sum_exp; 

            // 4. COLLAPSE (Conditional Moment Matching - IMM Approach)
            VectorXd v_collapsed = VectorXd::Zero(state_dim);
            for (int k = 0; k < n_nodes; ++k) {
                v_collapsed += prob_nodes_new(k) * v_updated_nodes[k];
            }
            
            MatrixXd P_collapsed = MatrixXd::Zero(state_dim, state_dim);
            
            for (int k = 0; k < n_nodes; ++k) {
                VectorXd diff = v_updated_nodes[k] - v_collapsed;
                P_collapsed += prob_nodes_new(k) * (P_update_base + diff * diff.transpose());
            }
            
            // State update for the next step
            v_est = v_collapsed;
            P_est = P_collapsed;
            
            v_history.row(t) = v_collapsed.transpose();
        }
        return log_likelihood_total;
    }
};

// ======================================================================
// 1. FEUNOU & OKOU FILTER (Moment-Matching EKF with Dynamic SVJD )
// ======================================================================

class FeunouOkouKF {
private:
    VectorXd v_est;
    MatrixXd P_est;
    double log_likelihood_total;
    MatrixXd v_history; 

public:
    FeunouOkouKF() { log_likelihood_total = 0.0; }
    
    MatrixXd get_v_history() const { return v_history; } 

    double filter_series(
        Eigen::Ref<const MatrixXd> Z_series,     
        Eigen::Ref<const VectorXd> Phi0,       // Must be Eq 27 of FO (including base jumps intensities c0)
        Eigen::Ref<const MatrixXd> Phi1,       // Must be Eq 27 of FO (including dynamical jump intensities c1, c2, c3)
        Eigen::Ref<const VectorXd> Gamma0,   
        Eigen::Ref<const MatrixXd> Gamma1,     
        Eigen::Ref<const MatrixXd> Omega_mat, 
        double sigma1, double sigma2, double dt,
        double c0_m, double c1_m, double c2_m, double c3_m,
        double lambda_minus, double mu1, double mu3, double rho3 
    ) {
        int T = Z_series.rows();
        int state_dim = 3;
        int obs_dim = Z_series.cols();
        log_likelihood_total = 0.0;
        v_history.resize(T, state_dim);

        MatrixXd Sigma_total = MatrixXd::Zero(state_dim, state_dim);

        // 0. PRE-COMPUTATION OF JUMP MOMENTS (Static)
        // E[X^2] is 2/lambda^2 for the exponential distribution 
        // Due to the fact that the jump contributes to factors like X^2, the variance of the Poisson generated jump
        // needs E[(X^2)^2] = E[X^4] = 24/lambda^4.
        double E_x4 = 24.0 / std::pow(lambda_minus, 4);

        // --- FO INIT ---
        Eigen::Matrix3d minus_K1 = Eigen::Matrix3d::Identity() - Phi1.block<3, 3>(0, 0);
        if (std::abs(minus_K1.determinant()) < 1e-15) return -1e9;

        Eigen::Vector3d v0_vec = minus_K1.lu().solve(Phi0.head<3>());
        if (!v0_vec.allFinite()) return -1e9;
        v_est = v0_vec.cwiseMax(1e-8);

        double v1_0 = v_est(0);
        double v2_0 = v_est(1);
        double v3_0 = v_est(2);

        Eigen::Matrix3d Sigma_0 = Eigen::Matrix3d::Zero();
        Sigma_0(0,0) = sigma1 * sigma1 * v1_0 * dt + 1e-8;
        Sigma_0(1,1) = sigma2 * sigma2 * v2_0 * dt + 1e-8;
        Sigma_0(2,2) = 1e-8;

        double c_minus_0 = c0_m + c1_m * v1_0 + c2_m * v2_0 + c3_m * v3_0;
        double var_jump_poisson_0 = c_minus_0 * dt * E_x4;

        Sigma_0(0,0) += mu1 * mu1 * var_jump_poisson_0;
        Sigma_0(2,2) += mu3 * mu3 * (std::pow(1.0 - rho3, 2) + std::pow(rho3, 2)) * var_jump_poisson_0;
        double cov_1_3_0 = mu1 * mu3 * (1.0 - rho3) * var_jump_poisson_0;
        Sigma_0(0,2) += cov_1_3_0;
        Sigma_0(2,0) += cov_1_3_0;

        if (!compute_P0_from_kronecker_FO(Phi1, Sigma_0, P_est)) {
            return -1e9;
        }

        // Filtering
        for (int t = 0; t < T; ++t) {
            VectorXd z_t = Z_series.row(t).transpose();
            
            // Volatilities must be positive
            double v1_prev = std::max(v_est(0), 1e-8);
            double v2_prev = std::max(v_est(1), 1e-8);
            double v3_prev = std::max(v_est(2), 1e-8);

            // 1. TRANSICTION VARIANCE COMPUTATION (Sigma_total)
            // Eq 28 OF Feunou & Okou 2018
            Sigma_total.setZero();
            
            // FO variance is linear wrt v_{t-1}. 

            // A) Base Diffusive Variance
            Sigma_total(0,0) = sigma1 * sigma1 * v1_prev * dt + 1e-8;
            Sigma_total(1,1) = sigma2 * sigma2 * v2_prev * dt + 1e-8;
            Sigma_total(2,2) = 1e-8; // The 3rd factor has no diffusive term

            // B) Jump Variances (Linearly dependent from the states)
            double c_minus_t = c0_m + c1_m * v1_prev + c2_m * v2_prev + c3_m * v3_prev;
            
            // Generated variance in the Compound Poisson process in the period dt
            double var_jump_poisson = c_minus_t * dt * E_x4;
            
            Sigma_total(0,0) += mu1 * mu1 * var_jump_poisson;
            Sigma_total(2,2) += mu3 * mu3 * (std::pow(1.0 - rho3, 2) + std::pow(rho3, 2)) * var_jump_poisson;
            
            double cov_1_3 = mu1 * mu3 * (1.0 - rho3) * var_jump_poisson;
            Sigma_total(0,2) += cov_1_3;
            Sigma_total(2,0) += cov_1_3;

            // 2. PREDICT
            VectorXd v_pred = Phi0 + Phi1 * v_est;
            v_pred = v_pred.cwiseMax(1e-8); 
            
            MatrixXd P_pred = Phi1 * P_est * Phi1.transpose() + Sigma_total;

            // 3. UPDATE (Cholesky)
            VectorXd z_pred = Gamma0 + Gamma1 * v_pred;
            MatrixXd S = Gamma1 * P_pred * Gamma1.transpose() + Omega_mat;
            
            LLT<MatrixXd> llt_S(S);
            if (llt_S.info() != Eigen::Success) return -1e9;

            MatrixXd P_GammaT = P_pred * Gamma1.transpose();
            MatrixXd K = llt_S.solve(P_GammaT.transpose()).transpose();

            VectorXd y_res = z_t - z_pred;
            v_est = v_pred + K * y_res;
            v_est = v_est.cwiseMax(1e-8); 
            P_est = P_pred - K * S * K.transpose();
            
            v_history.row(t) = v_est.transpose(); 

            // 4. LOG-LIKELIHOOD 
            double det_S = S.determinant();
            if (det_S <= 0.0) return -1e9; 
            
            double mahalanobis = (y_res.transpose() * llt_S.solve(y_res)).value();
            
            double log_lik_t = -0.5 * obs_dim * std::log(2.0 * M_PI) - 0.5 * std::log(det_S) - 0.5 * mahalanobis;
            log_likelihood_total += log_lik_t;
        }
        return log_likelihood_total;
    }
};

// ======================================================================
// 2. PARTICLE FILTER (APF algorithm)
// ======================================================================

#include <random> 

class ParticleFilter {
private:
    double log_likelihood_total;
    MatrixXd v_history; 
    std::mt19937 generator; 

public:
    ParticleFilter() { 
        log_likelihood_total = 0.0; 
        std::random_device rd;
        generator.seed(rd()); 
    }
    
    MatrixXd get_v_history() const { return v_history; } 

    double filter_series(
        Eigen::Ref<const MatrixXd> Z_series,     
        Eigen::Ref<const VectorXd> base_Phi0,  // Base drift [k1*vbar1*dt, k2*vbar2*dt, 0]
        Eigen::Ref<const MatrixXd> Phi1,         
        Eigen::Ref<const VectorXd> Gamma0,   
        Eigen::Ref<const MatrixXd> Gamma1,     
        Eigen::Ref<const MatrixXd> Omega_mat, 
        double sigma1, double sigma2, double dt,
        double c0_m, double c1_m, double c2_m, double c3_m,
        double lambda_minus, double mu1, double mu3, double rho3, 
        double init_v1, double init_v2, double init_v3,
        int n_particles
    ) {
        int T = Z_series.rows();
        int dim_z = Z_series.cols();
        int state_dim = 3;
        log_likelihood_total = 0.0;
        
        v_history.resize(T, state_dim); 

        MatrixXd particles = MatrixXd::Zero(state_dim, n_particles);
        for(int i=0; i<n_particles; ++i) {
            particles(0, i) = init_v1;  // Takes vbar1 from Python file
            particles(1, i) = init_v2;  // Takes vbar2 from Python file
            particles(2, i) = init_v3;  // Takes 1e-4 from Python file
        }

        std::normal_distribution<double> std_norm(0.0, 1.0);          // Brownian noise
        std::exponential_distribution<double> exp_dist(lambda_minus); // Jumps
        std::uniform_real_distribution<double> unif_dist(0.0, 1.0);   // Bernoulli

        MatrixXd Omega_inv = Omega_mat.inverse();
        double det_Omega = Omega_mat.determinant();
        double const_term = std::pow(2.0 * M_PI, dim_z) * det_Omega;

        std::vector<double> weights(n_particles, 1.0 / n_particles);

        // 0. AVERAGE EXPECTED JUMP IMPACT
        double exp_x2 = 2.0 / (lambda_minus * lambda_minus);
        
        VectorXd E_jump_x = VectorXd::Zero(state_dim);
        E_jump_x(0) = mu1 * exp_x2;
        E_jump_x(2) = mu3 * (1.0 - rho3) * exp_x2;
        
        VectorXd E_jump_y = VectorXd::Zero(state_dim);
        E_jump_y(2) = mu3 * rho3 * exp_x2;

        for (int t = 0; t < T; ++t) {
            VectorXd z_t = Z_series.row(t).transpose();
            VectorXd mean_state = VectorXd::Zero(state_dim);
            
            std::vector<double> log_weights(n_particles, 0.0);
            double max_log_weight = -1e9; 

            for(int i = 0; i < n_particles; ++i) {
                VectorXd p_prev = particles.col(i);
                
                // A-priori arrival probability (valid for both X and Y, since they are independent)
                double c_i = c0_m + c1_m * p_prev(0) + c2_m * p_prev(1) + c3_m * p_prev(2);
                double p_arr = 1.0 - std::exp(-c_i * dt);
                p_arr = std::max(1e-8, std::min(p_arr, 0.999999)); 
                
                // A-priori joint probabilities for the 4 scenaries 
                double prior_00 = (1.0 - p_arr) * (1.0 - p_arr); // No jumps
                double prior_10 = p_arr * (1.0 - p_arr);         // Only X jump
                double prior_01 = (1.0 - p_arr) * p_arr;         // Only Y jump
                double prior_11 = p_arr * p_arr;                 // Both jumps

                // A) PREDICTIVE LIKELIHOOD FOR THE 4 SCENARIES
                VectorXd v_pred_00 = base_Phi0 + Phi1 * p_prev;
                VectorXd v_pred_10 = v_pred_00 + E_jump_x;
                VectorXd v_pred_01 = v_pred_00 + E_jump_y;
                VectorXd v_pred_11 = v_pred_00 + E_jump_x + E_jump_y;

                auto get_log_L = [&](const VectorXd& v_pred_case) {
                    VectorXd z_pred_case = Gamma0 + Gamma1 * v_pred_case;
                    VectorXd res_case = z_t - z_pred_case;
                    return -0.5 * (res_case.transpose() * Omega_inv * res_case).value(); //no P0 in PF
                };

                double log_L_00 = get_log_L(v_pred_00);
                double log_L_10 = get_log_L(v_pred_10);
                double log_L_01 = get_log_L(v_pred_01);
                double log_L_11 = get_log_L(v_pred_11);

                // B) SAFE BAYES (Scale-invariant Proposal)
                double max_log_L = std::max({log_L_00, log_L_10, log_L_01, log_L_11});
                
                double L_00_safe = std::exp(log_L_00 - max_log_L);
                double L_10_safe = std::exp(log_L_10 - max_log_L);
                double L_01_safe = std::exp(log_L_01 - max_log_L);
                double L_11_safe = std::exp(log_L_11 - max_log_L);
                
                double evidence_safe = (prior_00 * L_00_safe) + (prior_10 * L_10_safe) + 
                                       (prior_01 * L_01_safe) + (prior_11 * L_11_safe) + 1e-300;
                
                // Probabilities case by case (normalized in [0, 1])
                double prop_00 = (prior_00 * L_00_safe) / evidence_safe;
                double prop_10 = (prior_10 * L_10_safe) / evidence_safe;
                double prop_01 = (prior_01 * L_01_safe) / evidence_safe;
                double prop_11 = (prior_11 * L_11_safe) / evidence_safe;

                // 2. SIMULATION (Sampling from the categorigal distribution)
                double u_jump = unif_dist(generator); // Recall that is between 0 and 1
                int jump_case = 0;
                
                // Think of probs like they're stacked: 
                // from 0 to prop_00 is the no jump case, from prop_00 to prop_00+prop_10 is the x jump case, ...
                if (u_jump < prop_00) {
                    jump_case = 0; // 00
                } else if (u_jump < prop_00 + prop_10) {
                    jump_case = 1; // 10
                } else if (u_jump < prop_00 + prop_10 + prop_01) {
                    jump_case = 2; // 01
                } else {
                    jump_case = 3; // 11
                }

                VectorXd jump_effect = VectorXd::Zero(state_dim);
                double prior_prob_case = prior_00;
                double prop_prob_case = prop_00;
                
                // Applying jumps impact depending on the sorted case
                if (jump_case == 1 || jump_case == 3) {
                    double x_minus = exp_dist(generator);
                    jump_effect(0) += mu1 * (x_minus * x_minus);
                    jump_effect(2) += mu3 * (1.0 - rho3) * (x_minus * x_minus);
                }
                if (jump_case == 2 || jump_case == 3) {
                    double y_indep = exp_dist(generator);
                    jump_effect(2) += mu3 * rho3 * (y_indep * y_indep);
                }

                // Matching probabilities for the Importance Ratio computation
                if (jump_case == 1) { prior_prob_case = prior_10; prop_prob_case = prop_10; }
                else if (jump_case == 2) { prior_prob_case = prior_01; prop_prob_case = prop_01; }
                else if (jump_case == 3) { prior_prob_case = prior_11; prop_prob_case = prop_11; }

                // 3. NOISE & UPDATE 
                VectorXd noise = VectorXd::Zero(state_dim);
                noise(0) = std_norm(generator) * sigma1 * std::sqrt(std::max(p_prev(0), 1e-8) * dt);
                noise(1) = std_norm(generator) * sigma2 * std::sqrt(std::max(p_prev(1), 1e-8) * dt);

                VectorXd v_new = base_Phi0 + Phi1 * p_prev + jump_effect + noise;
                v_new = v_new.cwiseMax(1e-8); 
                particles.col(i) = v_new;

                // 4. EXACT LIKELIHOOD WEIGHTING
                VectorXd z_pred_actual = Gamma0 + Gamma1 * v_new;
                VectorXd res_actual = z_t - z_pred_actual;
                double log_L_actual = -0.5 * (res_actual.transpose() * Omega_inv * res_actual).value();

                double importance_ratio = prior_prob_case / std::max(prop_prob_case, 1e-300);

                log_weights[i] = log_L_actual + std::log(std::max(importance_ratio, 1e-300));
                
                if (log_weights[i] > max_log_weight) {
                    max_log_weight = log_weights[i];
                }
            }

            // LOG-SUM-EXP TO AVOID UNDERFLOW 
            double sum_weights = 0.0;
            for(int i = 0; i < n_particles; ++i) {
                weights[i] = std::exp(log_weights[i] - max_log_weight);
                sum_weights += weights[i];
            }

            if (sum_weights <= 0.0 || std::isnan(sum_weights)) return -1e9;

            // Computing average state
            for(int i = 0; i < n_particles; ++i) {
                mean_state += particles.col(i) * (weights[i] / sum_weights);
            }
            v_history.row(t) = mean_state.transpose();

            // Update Likelihood 
            // log( avg_of_true_weights ) = max_log_weight + log(sum_weights / n_particles)
            double log_lik_t = max_log_weight + std::log(sum_weights / n_particles) - 0.5 * std::log(const_term);
            log_likelihood_total += log_lik_t;

            // RESAMPLING
            std::vector<double> cdf(n_particles);
            cdf[0] = weights[0] / sum_weights;
            for(int i = 1; i < n_particles; ++i) {
                cdf[i] = cdf[i-1] + (weights[i] / sum_weights);
            }

            double u = unif_dist(generator) / n_particles; 
            MatrixXd new_particles = MatrixXd::Zero(state_dim, n_particles);
            int idx = 0;
            for(int j = 0; j < n_particles; ++j) {
                while(u > cdf[idx] && idx < n_particles - 1) { idx++; }
                new_particles.col(j) = particles.col(idx);
                u += (1.0 / n_particles);
            }
            particles = new_particles; 
        }

        return log_likelihood_total;
    }
};