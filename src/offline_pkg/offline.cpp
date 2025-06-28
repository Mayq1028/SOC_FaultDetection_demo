    #include <iostream>
    #include <vector>
    #include <fstream>
    #include <sstream>
    #include <random>
    #include <Eigen/Dense>
    #include <Eigen/QR>
    #include <algorithm>
    #include <iomanip>
    #include <fcntl.h>
    #include <unistd.h>
    #include <termios.h>
    #include <sys/ioctl.h>
    #include <chrono>
    #include <thread>
    #include <cstring>


    using namespace Eigen;
    using namespace std;

    //======================================= 参数 ============================================//

    //电池参数
    struct BatteryParam {
        double R0   = 0.072;        // Ω
        double Rp   = 0.03;         // Ω
        double Cp   = 1000.0;       // F
        double Qn   = 2.0 * 3600.0; // 容量 (As)
        double deltaT   = 1.0;          // 采样周期 (s)
        double eta  = 1.0;          // 库仑效率
    };

    /* 状态空间系数矩阵（线性） */
    struct CoeffMatrix {
        Matrix2d A;
        Vector2d B;
        double   D;
    };

    //真实检测数据
    struct Battery {
        VectorXd Cur;
        VectorXd Vol;
        Battery() = default;
        Battery(VectorXd Cur, VectorXd Vol) : Cur(Cur), Vol(Vol){}
    };

    /* —— 噪声参数 —— */
    struct NoiseParam {
        double    ww = 0.0001;   // 过程噪声均匀上界
        double    vv = 0.001;   // 测量噪声均匀上界
        MatrixXd  Gw = Matrix2d::Identity() * 1e-3;   // EKF 用
        double    Gv = 0.01;   // EKF 用
    };

    /* —— 用于求 EKF 增益矩阵—— */
    struct EKFGainParam {
        MatrixXd Q = Matrix2d::Identity() * 1e-8;
        double   R = 0.01;
        MatrixXd P0 = Matrix2d::Zero();  // 初始协方差矩阵
        EKFGainParam() {
            P0(0,0) = 1.0;
            P0(1,1) = 1.0;
        }
    };

    /* —— 状态估计的初始值 —— */
    struct InitParam {
        Vector2d x1 = {0.01, 0.8};  // 初始状态
        Vector2d x1_ = {-0.05, 0.7}; // 其他初始值
        MatrixXd BG1_diag = Matrix2d::Zero(); // 边界生成器初值
        InitParam() {
            BG1_diag(0, 0) = 0.01;
            BG1_diag(1, 1) = 0.001;
        }
    };

    /* —— EKF 输出结果打包 —— */
    struct EstimateOutput {
        MatrixXd x_hat, x_low, x_up;
        VectorXd y_hat, y_low, y_up;// 离线
    };

    /* —— 真实系统仿真模型—— */
    struct SimSystem {
        MatrixXd x;   // 2×N 真实状态序列
        VectorXd y;   // (N-1)×1 真实输出序列
    };   
    
    // 维度结构体
    struct Dimension {
        int s;   // 状态维度
        int n;    // 输出维度
        
        Dimension(int s, int n) : s(s), n(n) {} // 通过构造函数初始化
    };

    //==================================== 函数 ===============================================//

    inline CoeffMatrix makeParaMatrix(const BatteryParam& bp)
    {
        CoeffMatrix matrix;                      // 待填充的返回值

        double e = exp(-bp.deltaT / (bp.Rp * bp.Cp));
        matrix.A <<  e , 0, 0 , 1;
        matrix.B <<  bp.Rp * (1 - e), -bp.eta * bp.deltaT / bp.Qn;
        matrix.D  = -bp.R0;

        return matrix;                           // Return-by-value（RVO 会消除拷贝）
    }

    // 从文件读取向量数据
    VectorXd load_data(const string& filename) {
        ifstream file(filename);
        if (!file.is_open()) {
            cerr << "错误：无法打开文件 " << filename << endl;
            exit(1);
        }
        
        vector<double> data;
        string line;
        while (getline(file, line)) {
            if (!line.empty()) {
                stringstream ss(line);
                double value;
                while (ss >> value) {
                    data.push_back(value);
                }
            }
        }
        file.close();
        
        VectorXd vec(data.size());
        for (int i = 0; i < data.size(); i++) {
            vec(i) = data[i];
        }
        return vec;
    }
    
    /* ---------------------------------------------------------
    *  生成真实状态 & 输出
    *  输入：
    *      - 系数矩阵  coeff         (A, B, D)
    *      - 电流序列  Cur           (N×1)
    *      - 初始状态  init.x1
    *      - OCV 拟合  pfit
    *      - 噪声参数  noise.ww / noise.vv
    *  输出：
    *      - 返回 SimSystem{x, y}
    * --------------------------------------------------------- */
    inline SimSystem generate_state(const CoeffMatrix& matrix, const VectorXd& Cur,
                                    const InitParam& init, const VectorXd& pfit,
                                    const NoiseParam& noise)
    {
        const int N = Cur.size();

        SimSystem sim;
        sim.x.resize(2, N);
        sim.y.resize(N - 1);
        sim.x.col(0) = init.x1;          // 初始真值

        random_device rd;
        mt19937       gen(rd());
        uniform_real_distribution<> dis_w(-noise.ww, noise.ww);
        uniform_real_distribution<> dis_v(-noise.vv, noise.vv);

        for (int k = 0; k < N - 1; ++k)
        {
            Vector2d w(dis_w(gen), dis_w(gen));
            sim.x.col(k + 1) = matrix.A * sim.x.col(k) + matrix.B * Cur(k) + w;
            double soc = sim.x(1, k);        // k 时刻 SOC
            double OCV = 0.0;
            for (int i = 0; i < pfit.size(); ++i)
                OCV += pfit[i] * pow(soc, pfit.size() - 1 - i);
            double v_noise = dis_v(gen);
            sim.y(k) = OCV - sim.x(0, k) + matrix.D * Cur(k) + v_noise;
        }

        return sim;  
    }


    // 降阶函数
    MatrixXd jiangjie(const MatrixXd& G, int s) {
        int n_z = G.rows(); // 获取行数
        int r = G.cols();   // 获取列数
        
        // 步骤1: 计算各列的欧氏范数并排序
        VectorXd Gn(r);
        for (int j = 0; j < r; ++j) {
            Gn(j) = G.col(j).norm(); // 计算欧氏范数
        }

        // 创建索引并降序排序
        vector<int> indices(r);
        for (int i = 0; i < r; ++i) indices[i] = i;
        sort(indices.begin(), indices.end(), [&](int a, int b) {
            return Gn(a) > Gn(b);
        });

        // 按新顺序排列矩阵
        MatrixXd Gx0(n_z, r);
        for (int j = 0; j < r; ++j) {
            Gx0.col(j) = G.col(indices[j]);
        }

        // 情况1: r <= s 直接返回
        if (r <= s) {
            return Gx0;
        }
        // 情况2: r > s
        else {
            // 错误检查
            if (s < n_z) {
                throw invalid_argument("error! s must be >= n_z");
            }
            
            int k = s - n_z; // 计算要保留的列数
            
            // 分割矩阵为两部分
            MatrixXd Gx01 = Gx0.leftCols(k);            // 前k列
            MatrixXd Gx02 = Gx0.rightCols(r - k);        // 剩余列
            
            // 计算Q矩阵 (对角线为行绝对值和)
            MatrixXd Q = MatrixXd::Zero(n_z, n_z);
            for (int i = 0; i < n_z; ++i) {
                Q(i, i) = Gx02.row(i).cwiseAbs().sum(); // 计算每行绝对值和
            }
            
            // 水平拼接矩阵
            MatrixXd G_j(Gx01.rows(), Gx01.cols() + Q.cols());
            G_j << Gx01, Q;
            
            return G_j;
        }
    }

    // 多项式求值辅助函数
    double polyval(const VectorXd& coeffs, double x) {
        double result = 0.0;
        for (int i = 0; i < coeffs.size(); ++i) {
            result = result * x + coeffs[coeffs.size() - 1 - i];
        }
        return result;
    }

    // 多项式导数系数计算
    VectorXd polyder(const VectorXd& coeffs) {
        if (coeffs.size() <= 1) {
            return VectorXd::Zero(1);
        }
        
        VectorXd deriv(coeffs.size() - 1);
        for (int i = 0; i < deriv.size(); ++i) {
            deriv[i] = coeffs[i] * (coeffs.size() - 1 - i);
        }
        return deriv;
    }

    void save_data_to_csv(const EstimateOutput& est, const SimSystem& sim, int N)
    {
        ofstream file("data/output/offline.csv");
        
        file << "k,Up_est,Up_low,Up_up,Up_true,SOC_est,SOC_low,SOC_up,SOC_true,Vol,y_est,y_low,y_up,y_true\n";
        
        for (int i = 0; i < N - 1; i++) {
            file << i << ","
                << est.x_hat(0, i) << ","
                << est.x_low(0, i) << ","
                << est.x_up(0, i) << ","
                << sim.x(0, i) << ","
                << est.x_hat(1, i) << ","
                << est.x_low(1, i) << ","
                << est.x_up(1, i) << ","
                << sim.x(1, i) << ","
                << sim.y(i) << ","
                << est.y_hat(i) << ","
                << est.y_low(i) << ","
                << est.y_up(i) << ","
                << sim.y(i) << "\n";
        }
        
        file.close();
    }

    void is_fault(EstimateOutput output, Battery input, std::vector<int>& fault_status) {
        int N = input.Vol.size();
        
        // 遍历每个时刻 k
        for (int k = 0; k < N - 1; ++k) {
            int fault = (input.Vol[k] < output.y_low[k] || input.Vol[k] > output.y_up[k]) ? 1 : 0;
            fault_status.push_back(fault);
            
            // 打印状态
            std::cout << "k=" << k << ", 状态：";
            std::cout << "x_hat=" << output.x_hat(0, k) << ", ";
            std::cout << "x_low=" << output.x_low(0, k) << ", ";
            std::cout << "x_up=" << output.x_up(0, k) << "; ";

            // 打印输出
            std::cout << "输出：";
            std::cout << "y_hat=" << output.y_hat(k) << ", ";
            std::cout << "y_low=" << output.y_low(k) << ", ";
            std::cout << "y_up=" << output.y_up(k) << "; ";

            //打印故障
            std::cout << "f=" << fault << std::endl;
        }
    }  
    
    double compute_error(float target, float est) {
        float error = (target - est > 0) ? (target - est) : (est - target);
        error = error / target;
        return error;
    }

    void print_error(SimSystem& sim, Battery& input, EstimateOutput& output) {
        int N = sim.x.size();
        std::vector<float> e_x1;
        std::vector<float> e_x2;
        std::vector<float> e_y;
        std::vector<float> e_fit;
        // 拟合误差
        std::cout << "y_fit_error:" << std::endl;
        for (int k = 0; k < N; k++) {
            e_fit.push_back(compute_error(sim.y(k), input.Vol(k)));// 
            std::cout << "k=" << k << ", " << e_fit[k] << std::endl;
        }
        // 估计误差
        std::cout << "Up_hat_error:" << std::endl;
        for (int k = 0; k < N; k++) {
            e_x1.push_back(compute_error(sim.x(0, k), output.x_hat(0, k)));// 
            std::cout << "k=" << k << ", " << e_x1[k] << std::endl;
        }
        std::cout << "SOC_hat_error:" << std::endl;
        for (int k = 0; k < N; k++) {
            e_x2.push_back(compute_error(sim.x(1, k), output.x_hat(1, k)));// 
            std::cout << "k=" << k << ", " << e_x2[k] << std::endl;
        }
        std::cout << "U_hat_error:" << std::endl;
        for (int k = 0; k < N; k++) {
            e_y.push_back(compute_error(sim.y(k), output.y_hat(k)));// 
            std::cout << "k=" << k << ", " << e_y[k] << std::endl;
        }
    }

    // 离线数据集EKF实现
    inline EstimateOutput EKF_offline(const CoeffMatrix& cm, 
                int s, 
                const VectorXd& pfit, 
                const Battery& input, 
                const InitParam& init, 
                const NoiseParam& noise, 
                EKFGainParam& gain)
    {
        int N = input.Cur.size();
        int n_state = init.x1.size();
        
        // 计算多项式导数
        VectorXd pder = polyder(pfit);
        
        // 初始化输出变量
        MatrixXd x_(n_state, N), x_l(n_state, N), x_u(n_state, N);
        VectorXd y_(N - 1), y_l(N - 1), y_u(N - 1);
        
        // 初始化边界生成器
        MatrixXd BG_current = init.BG1_diag;
        
        // 存储中间结果
        VectorXd Uoc = VectorXd::Zero(N - 1);
        VectorXd Up = VectorXd::Zero(N - 1);
        
        // 设置初始状态
        x_.col(0) = init.x1;
        x_l.col(0) = init.x1 - init.BG1_diag.cwiseAbs().rowwise().sum();
        x_u.col(0) = init.x1 + init.BG1_diag.cwiseAbs().rowwise().sum();
        
        // 主循环
        for (int k = 0; k < N - 1; ++k) {
            // 预测步骤
            VectorXd x_pre = cm.A * x_.col(k) + cm.B * input.Cur[k];
            MatrixXd BG_pre = cm.A * BG_current;
            BG_pre.conservativeResize(NoChange, BG_pre.cols() + noise.Gw.cols());
            BG_pre.rightCols(noise.Gw.cols()) = noise.Gw;
            
            // 计算测量矩阵
            RowVectorXd C(2);
            C << -1, polyval(pder, x_pre[1]);
            
            // 估计输出
            Uoc[k] = polyval(pfit, x_pre[1]);
            Up[k] = x_pre[0];
            y_[k] = Uoc[k] - Up[k] + cm.D * input.Cur[k];
            
            // 计算输出边界
            MatrixXd BG_y = C * BG_current;
            BG_y.conservativeResize(NoChange, BG_y.cols() + 1);
            BG_y(BG_y.cols() - 1) = noise.Gv;
            BG_y = jiangjie(BG_y, s);
            
            y_l[k] = y_[k] - BG_y.cwiseAbs().sum();
            y_u[k] = y_[k] + BG_y.cwiseAbs().sum();
            
            // 卡尔曼增益和状态更新
            MatrixXd P_pred = cm.A * gain.P0 * cm.A.transpose() + gain.Q;
            double S = (C * P_pred * C.transpose())(0, 0) + gain.R; // 标量计算
            MatrixXd K = P_pred * C.transpose() * (1.0 / S);        // 用除法替代逆
            MatrixXd temp = (MatrixXd::Identity(n_state, n_state) - K * C);
            gain.P0 = temp * P_pred;
            
            // 状态更新
            x_.col(k + 1) = x_pre + K * (input.Vol[k] - y_[k]);
            
            // 更新边界生成器
            MatrixXd BG(BG_pre);
            BG.conservativeResize(NoChange, BG.cols() + (K * (-BG_y)).cols());
            BG.rightCols((K * (-BG_y)).cols()) = K * (-BG_y);
            BG_current = 0.5 * jiangjie(BG, s);
            
            // 计算状态边界
            VectorXd abs_sum = BG_current.cwiseAbs().rowwise().sum();
            x_l.col(k + 1) = x_.col(k + 1) - abs_sum;
            x_u.col(k + 1) = x_.col(k + 1) + abs_sum;
        }

        // 输出结果
        EstimateOutput result;
        result.x_hat = x_;
        result.x_low = x_l;
        result.x_up = x_u;
        result.y_hat = y_;
        result.y_low = y_l;
        result.y_up = y_u;

        return result;
    }

    int main() {
        // 电池参数
        BatteryParam bp;   
        //系数矩阵
        CoeffMatrix cm = makeParaMatrix(bp);  
        // 加载拟合的SOC非线性观测系数C
        VectorXd pfit = load_data("data/pre_data/pfit_offline.txt"); 
        // 维数
        Dimension dim(12, 2); 
        // 初始值
        InitParam init; 
        // 噪声
        NoiseParam noise; 
        //ekf增益矩阵参数
        EKFGainParam gain; 

        //加载数据集原始数据
        VectorXd voltage = load_data("data/pre_data/voltage.txt");
        VectorXd current = load_data("data/pre_data/current.txt");

        // 电流故障
        // VectorXd f_current = -current;
        // f_current.segment(499, 501) = VectorXd::Constant(501, 10); // 输入电流发生阶跃跳变故障
        Battery input(-current, voltage.array() + 0.0001 * VectorXd::Random(current.size()).array());

        // 生产soc仿真模型
        SimSystem sim = generate_state(cm, input.Cur, init, pfit, noise);

        // EKF状态估计
        EstimateOutput output = EKF_offline(cm, dim.s, pfit, input, init, noise, gain);

        std::vector<int> fault_status; // 用来存储每个时刻的故障状态
        is_fault(output, input, fault_status); //判断并打印故障状态,输出数据
        // print_error(sim, input, output);
        
        // 保存output数据
        save_data_to_csv(output, sim, current.size());
        
        return 0;
    }
