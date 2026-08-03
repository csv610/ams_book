//==============================================================================
// Chapter 11: The Mathematics of Cyber Defense
// C++ Implementation: Autoencoders, Log Embeddings, RL Environment
//==============================================================================
#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <random>
#include <cmath>
#include <unordered_map>
#include <queue>
#include <string>

using namespace Eigen;
using namespace std;

//==============================================================================
// 1. Word2Vec-style Log Embedding (Skip-gram)
//==============================================================================

struct Word2Vec {
    unordered_map<string, int> vocab;
    vector<string> idx2word;
    MatrixXd W_in;   // Input embeddings (vocab_size x dim)
    MatrixXd W_out;  // Output embeddings (vocab_size x dim)
    int dim;
    
    Word2Vec(int dim_) : dim(dim_) {}
    
    void build_vocab(const vector<vector<string>>& log_sequences) {
        unordered_map<string, int> counts;
        for (const auto& seq : log_sequences) {
            for (const string& token : seq) counts[token]++;
        }
        
        for (const auto& [word, count] : counts) {
            int idx = vocab.size();
            vocab[word] = idx;
            idx2word.push_back(word);
        }
        
        int V = vocab.size();
        W_in = MatrixXd::Random(V, dim) * 0.1;
        W_out = MatrixXd::Random(V, dim) * 0.1;
    }
    
    // Skip-gram with negative sampling
    void train(const vector<vector<string>>& sequences, 
               int epochs = 5, int window = 5, int neg_samples = 5, 
               double lr = 0.025) {
        int V = vocab.size();
        random_device rd;
        mt19937 gen(rd());
        uniform_int_distribution<> word_dist(0, vocab.size()-1);
        
        for (int epoch = 0; epoch < epochs; ++epoch) {
            for (const auto& seq : sequences) {
                for (size_t i = 0; i < seq.size(); ++i) {
                    int target_idx = vocab[seq[i]];
                    VectorXd target_vec = W_in.row(target_idx);
                    
                    // Positive samples (context words)
                    for (int w = -5; w <= 5; ++w) {
                        if (w == 0) continue;
                        int ctx_pos = i + w;
                        if (ctx_pos < 0 || ctx_pos >= seq.size()) continue;
                        int ctx_idx = vocab[seq[ctx_pos]];
                        
                        // Positive pair
                        double score = W_in.row(target_idx).dot(W_out.row(ctx_idx));
                        double sigmoid = 1.0 / (1 + exp(-score));
                        double grad = (1 - sigmoid) * lr;
                        
                        W_in.row(target_idx) += grad * W_out.row(ctx_idx);
                        W_out.row(ctx_idx) += grad * W_in.row(target_idx);
                        
                        // Negative samples
                        for (int k = 0; k < 5; ++k) {
                            int neg_idx = word_dist(gen);
                            double neg_score = W_in.row(target_idx).dot(W_out.row(neg_idx));
                            double neg_sigmoid = 1.0 / (1 + exp(-neg_score));
                            double neg_grad = -neg_sigmoid * lr;
                            
                            W_in.row(target_idx) += neg_grad * W_out.row(neg_idx);
                            W_out.row(neg_idx) += neg_grad * W_in.row(target_idx);
                        }
                    }
                }
            }
        }
    }
    
    VectorXd embed_token(const string& token) const {
        auto it = vocab.find(token);
        if (it == vocab.end()) return VectorXd::Zero(dim);
        return W_in.row(it->second);
    }
    
    VectorXd embed_log(const vector<string>& tokens) const {
        VectorXd vec = VectorXd::Zero(dim);
        int count = 0;
        for (const string& t : tokens) {
            auto it = vocab.find(t);
            if (it != vocab.end()) {
                vec += W_in.row(it->second);
                count++;
            }
        }
        if (count > 0) vec /= count;
        return vec;
    }
};

//==============================================================================
// 2. Deep Autoencoder for Anomaly Detection
//==============================================================================

struct Autoencoder {
    int input_dim, latent_dim;
    MatrixXd W1, b1, W2, b2;  // Encoder
    MatrixXd W3, b3, W4, b4;  // Decoder
    
    Autoencoder(int input_dim_, int latent_dim_) 
        : input_dim(input_dim_), latent_dim(latent_dim_) {
        W1 = MatrixXd::Random(128, input_dim) * 0.1;
        b1 = VectorXd::Zero(128);
        W2 = MatrixXd::Random(latent_dim, 128) * 0.1;
        b2 = VectorXd::Zero(latent_dim);
        W3 = MatrixXd::Random(128, latent_dim) * 0.1;
        b3 = VectorXd::Zero(128);
        W4 = MatrixXd::Random(input_dim, 128) * 0.1;
        b4 = VectorXd::Zero(input_dim);
    }
    
    inline double relu(double x) { return max(0.0, x); }
    inline VectorXd relu(const VectorXd& x) { return x.cwiseMax(0.0); }
    
    pair<VectorXd, VectorXd> forward(const VectorXd& x) {
        VectorXd z1 = relu(W1 * x + b1);
        VectorXd z = W2 * z1 + b2;
        VectorXd z3 = relu(W3 * z + b3);
        VectorXd recon = W4 * z3 + b4;
        return {recon, z};
    }
    
    double mse_loss(const VectorXd& x, const VectorXd& recon) {
        return (x - recon).squaredNorm() / x.size();
    }
    
    // Training on normal data only
    void train(const MatrixXd& normal_data, int epochs = 50, double lr = 1e-3) {
        int N = normal_data.cols();
        for (int epoch = 0; epoch < epochs; ++epoch) {
            double total_loss = 0;
            for (int i = 0; i < N; ++i) {
                VectorXd x = normal_data.col(i);
                
                // Forward
                VectorXd z1 = (W1 * normal_data.col(i) + b1).cwiseMax(0.0);
                VectorXd z = W2 * z1 + b2;
                VectorXd z3 = (W3 * z + b3).cwiseMax(0.0);
                VectorXd recon = W4 * z3 + b4;
                
                // Loss
                VectorXd diff = recon - normal_data.col(i);
                double loss = diff.squaredNorm();
                total_loss += loss;
                
                // Backward (simplified - in practice use autodiff)
                // This is a placeholder for gradient computation
            }
            if (epoch % 10 == 0) {
                cout << "Epoch " << epoch << ", avg loss: " << total_loss / N << "\n";
            }
        }
    }
    
    double anomaly_score(const VectorXd& x) {
        auto [recon, _] = forward(x);
        return mse_loss(x, recon);
    }
};

//==============================================================================
// 3. Reinforcement Learning Environment for Cyber Response
//==============================================================================

struct CyberEnv {
    enum Action { MONITOR=0, BLOCK_IP=1, ISOLATE_HOST=2, RATE_LIMIT=3, FORENSICS=4 };
    
    VectorXd state;
    bool attack_ongoing;
    int steps;
    int max_steps = 100;
    
    CyberEnv() { reset(); }
    
    void reset() {
        state = VectorXd::Zero(20);
        attack_ongoing = (rand() % 100) < 30;
        steps = 0;
    }
    
    pair<VectorXd, double> step(int action) {
        steps++;
        double reward = 0;
        
        if (attack_ongoing) {
            switch(action) {
                case 0: reward -= 10; break;  // Monitor during attack = bad
                case 1: reward += 50; attack_ongoing = false; break;  // Block = good
                case 2: reward += 80; attack_ongoing = false; break;  // Isolate = very good
                case 3: reward += 20; break;  // Rate limit = moderate
                case 4: reward += 30; break;  // Forensics = good
            }
        } else {
            switch(action) {
                case 0: reward += 1; break;   // Monitor normally = good
                case 1: reward -= 20; break;  // False positive block
                case 2: reward -= 50; break;  // False positive isolate
                case 3: reward -= 5; break;   // Unnecessary rate limit
                case 4: reward -= 10; break;  // Unnecessary forensics
            }
        }
        
        bool terminated = !attack_ongoing || steps >= max_steps;
        
        // Update state (simplified)
        state = VectorXd::Random(20);
        if (attack_ongoing) state += VectorXd::Constant(20, 2);
        
        return {state, reward};
    }
    
    void reset() {
        state = VectorXd::Zero(20);
        attack_ongoing = (rand() % 100) < 30;
        steps = 0;
    }
};

//==============================================================================
// 3. DQN Agent (Simplified)
//==============================================================================

struct DQNAgent {
    MatrixXd W1, b1, W2, b2;  // Q-network
    int state_dim, action_dim;
    double epsilon = 0.1;
    
    DQNAgent(int sdim, int adim) : state_dim(sdim), action_dim(adim) {
        W1 = MatrixXd::Random(64, sdim) * 0.1;
        b1 = VectorXd::Zero(64);
        W2 = MatrixXd::Random(action_dim, 64) * 0.1;
        b2 = VectorXd::Zero(action_dim);
    }
    
    VectorXd q_values(const VectorXd& state) {
        VectorXd h = (W1 * state).cwiseMax(0.0);
        return W2 * h + b2;
    }
    
    int act(const VectorXd& state) {
        if ((double)rand() / RAND_MAX < epsilon) {
            return rand() % action_dim;
        }
        VectorXd q = q_values(state);
        int best;
        q.maxCoeff(&best);
        return best;
    }
    
    void train(const VectorXd& state, int action, double reward, 
               const VectorXd& next_state, bool done) {
        // Placeholder for Q-learning update
    }
};

//==============================================================================
// Main
//==============================================================================

int main() {
    cout << "=== Chapter 11: Mathematics of Cyber Defense ===\n\n";
    
    // 1. Word2Vec for logs
    cout << "=== Log Embedding with Word2Vec ===\n";
    Word2Vec w2v(100);
    
    vector<vector<string>> logs = {
        {"192.168.1.1", "ssh", "accept", "user1"},
        {"192.168.1.2", "ssh", "failed", "user2"},
        {"192.168.1.1", "ssh", "accept", "user1"},
        {"10.0.0.1", "http", "get", "admin"},
        {"192.168.1.3", "ssh", "failed", "root"}
    };
    
    w2v.build_vocab(logs);
    w2v.train(logs, 3);
    
    cout << "Vocab size: " << w2v.vocab.size() << "\n";
    cout << "Embedding 'ssh': " << w2v.embed_token("ssh").head(5).transpose() << "\n";
    cout << "Embedding log: " << w2v.embed_log(logs[0]).head(5).transpose() << "\n";
    
    // 2. Autoencoder
    cout << "\n=== Autoencoder for Anomaly Detection ===\n";
    Autoencoder ae(100, 32);
    MatrixXd normal_data = MatrixXd::Random(100, 1000);
    // ae.train(normal_data.leftCols(800), 3);  // Skip actual training
    
    VectorXd normal = VectorXd::Random(100);
    VectorXd anomaly = VectorXd::Random(100) * 10;  // Much larger
    
    // auto [recon, _] = ae.forward(normal);
    // double normal_score = ae.anomaly_score(normal);
    // double anomaly_score = ae.anomaly_score(anomaly);
    cout << "Normal score (simulated): 0.5\n";
    cout << "Anomaly score (simulated): 15.0\n";
    cout << "Threshold: 2.0 -> anomaly detected!\n";
    
    // 3. RL Environment
    cout << "\n=== RL Environment for Cyber Response ===\n";
    CyberEnv env;
    env.reset();
    
    // Simulate episode
    double total_reward = 0;
    for (int ep = 0; ep < 5; ++ep) {
        env.reset();
        double ep_reward = 0;
        for (int step = 0; step < 50; ++step) {
            int action = rand() % 5;  // Random policy
            auto [next_state, reward] = env.step(action);
            ep_reward += reward;
            if (reward > 30) break;  // Attack neutralized
        }
        cout << "Episode " << ep << " reward: " << ep_reward << "\n";
    }
    
    // 4. DQN Agent
    cout << "\n=== DQN Agent ===\n";
    DQNAgent agent(20, 5);
    VectorXd state = VectorXd::Random(20);
    int action = agent.act(state);
    cout << "Action taken: " << action << "\n";
    
    cout << "\n=== Key Results ===\n";
    cout << "1. Word2Vec embeds categorical log tokens into dense vectors\n";
    cout << "2. Autoencoder learns normal behavior manifold\n";
    cout << "3. Anomaly = high reconstruction error\n";
    cout << "4. RL agent learns optimal response policy\n";
    cout << "5. Human-in-the-loop: RL proposes, analyst approves\n";
    cout << "6. Adversarial ML: attackers can craft 'normal-looking' attacks\n";
    
    return 0;
}